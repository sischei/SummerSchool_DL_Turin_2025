import json
import os
import numpy as np
import tensorflow as tf

import scipy.io as sio
import simulation_KS as KS
import simulation_Davila as Davila
import simulation_DavilaAS as DavilaAS
import simulation_JFV as JFV

EPSILON = 1e-3
DTYPE = "float64"
tf.keras.backend.set_floatx(DTYPE)
if DTYPE == "float64":
    NP_DTYPE = np.float64
elif DTYPE == "float32":
    NP_DTYPE = np.float32
else:
    raise ValueError("Unknown dtype.")


class NumpyEncoder(json.JSONEncoder):
    def default(self, o):  # pylint: disable=E0202
        if isinstance(o, np.ndarray):
            return o.tolist()
        return json.JSONEncoder.default(self, o)


class BasicDataSet():
    def __init__(self, datadict=None):
        self.datadict, self.keys = None, None
        self.size, self.idx_in_epoch, self.epoch_used = None, None, None
        if datadict:
            self.update_datadict(datadict)

    def update_datadict(self, datadict):
        self.datadict = datadict
        self.keys = datadict.keys()
        size_list = [datadict[k].shape[0] for k in self.keys]
        for i in range(1, len(size_list)):
            assert size_list[i] == size_list[0], "The size does not match."
        self.size = size_list[0]
        self.shuffle()
        self.epoch_used = 0

    def shuffle(self):
        idx = np.arange(0, self.size)
        np.random.shuffle(idx)
        self.datadict = dict((k, self.datadict[k][idx]) for k in self.keys)
        self.idx_in_epoch = 0

    def next_batch(self, batch_size):
        if self.idx_in_epoch + batch_size > self.size:
            self.shuffle()
            self.epoch_used += 1
        idx = slice(self.idx_in_epoch, self.idx_in_epoch+batch_size)
        self.idx_in_epoch += batch_size
        return dict((k, self.datadict[k][idx]) for k in self.keys)


class DataSetwithStats(BasicDataSet):
    def __init__(self, stats_keys, datadict=None):
        super().__init__(datadict)
        self.stats_keys = stats_keys
        self.stats_dict, self.stats_dict_tf = {}, {}
        for k in stats_keys:
            self.stats_dict[k] = None
            self.stats_dict_tf[k] = None

    def update_stats(self, data, key, ma):
        # data can be of shape B * d or B * n_agt * d
        axis_for_mean = tuple(list(range(len(data.shape)-1)))
        if self.stats_dict[key] is None:
            mean, std = data.mean(axis=axis_for_mean), data.std(axis=axis_for_mean)
        else:
            mean_new, std_new = data.mean(axis=axis_for_mean), data.std(axis=axis_for_mean)
            mean, std = self.stats_dict[key]
            mean = mean * ma + mean_new * (1-ma)
            std = std * ma + std_new * (1-ma)
        self.stats_dict[key] = (mean, std)
        self.stats_dict_tf[key] = (tf.constant(mean, dtype=DTYPE), tf.constant(std, dtype=DTYPE))

    def normalize_data(self, data, key, withtf=False):
        if withtf:
            mean, std = self.stats_dict_tf[key]
        else:
            mean, std = self.stats_dict[key]
        return (data - mean) / std

    def unnormalize_data(self, data, key, withtf=False):
        if withtf:
            mean, std = self.stats_dict_tf[key]
        else:
            mean, std = self.stats_dict[key]
        return data * std + mean

    def save_stats(self, path):
        with open(os.path.join(path, "stats.json"), "w") as fp:
            json.dump(self.stats_dict, fp, cls=NumpyEncoder)

    def load_stats(self, path):
        with open(os.path.join(path, "stats.json"), "r") as fp:
            saved_stats = json.load(fp)
        for key in saved_stats:
            assert key in self.stats_dict, "The key of stats_dict does not match!"
            mean, std = saved_stats[key]
            mean, std = np.asarray(mean).astype(NP_DTYPE), np.asarray(std).astype(NP_DTYPE)
            self.stats_dict[key] = (mean, std)
            self.stats_dict_tf[key] = (tf.constant(mean, dtype=DTYPE), tf.constant(std, dtype=DTYPE))


class InitDataSet(DataSetwithStats):
    def __init__(self, mparam, config):
        super().__init__(stats_keys=["basic_s", "agt_s", "value"])
        self.mparam = mparam
        self.config = config
        self.n_basic = config["n_basic"]
        self.n_fm = config["n_fm"]  # fixed moments
        self.n_path = config["dataset_config"]["n_path"]
        self.t_burn = config["dataset_config"]["t_burn"]
        self.c_policy_const_share = lambda *args: config["init_const_share"]
        if not config["init_with_bchmk"]:
            assert config["policy_config"]["update_init"], \
                "Must update init data during learning if bchmk policy is not used for sampling init"

    def update_with_burn(self, policy, policy_type, t_burn=None, state_init=None, use_last=False):
        if t_burn is None:
            t_burn = self.t_burn
        if state_init is None:
            state_init = self.datadict
        #TODO func='last' is only for KS and Davila
        if use_last:
            simul_data = self.simul_k_func(
                self.n_path, t_burn, self.mparam,
                policy, policy_type, state_init=state_init, func='last'
            )
        else:
            simul_data = self.simul_k_func(
                self.n_path, t_burn, self.mparam,
                policy, policy_type, state_init=state_init
            )
        self.update_from_simul(simul_data, use_last)

    def update_from_simul(self, simul_data, use_last=False):
        if use_last:
            init_datadict = dict((k, simul_data[k].copy()) for k in self.keys)
        else:
            init_datadict = dict((k, simul_data[k][..., -1].copy()) for k in self.keys)
        for k in self.keys:
            if len(init_datadict[k].shape) == 1:
                init_datadict[k] = init_datadict[k][:, None] # for macro init state like N in JFV
        notnan = ~(np.isnan(init_datadict["k_cross"]).any(axis=1))
        if np.sum(~notnan) > 0:
            num_nan = np.sum(~notnan)
            num_total = notnan.shape[0]
            print("Warning: {} of {} init samples are nan!".format(num_nan, num_total))
            idx = np.where(notnan)[0]
            idx = np.concatenate([idx, idx[:num_nan]])
            for k in self.keys:
                init_datadict[k] = init_datadict[k][idx]
        self.update_datadict(init_datadict)

    def process_vdatadict(self, v_datadict, norm=True):
        idx_nan = np.logical_or(
            np.isnan(v_datadict["basic_s"]).any(axis=(1, 2)),
            np.isnan(v_datadict["value"]).any(axis=(1, 2))
        )
        ma = self.config["dataset_config"]["moving_average"]
        for key, array in v_datadict.items():
            array = array[~idx_nan].astype(NP_DTYPE)
            self.update_stats(array, key, ma)
            if norm:
                v_datadict[key] = self.normalize_data(array, key)
            else:
                v_datadict[key] = array
        print("Average of total utility %f." % (self.stats_dict["value"][0][0]))

        valid_size = self.config["value_config"]["valid_size"]
        n_sample = v_datadict["value"].shape[0]
        if valid_size > 0.2*n_sample:
            valid_size = int(0.2*n_sample)
            print("Valid size is reduced to %d according to small data size!" % valid_size)
        print("The dataset has %d samples in total." % n_sample)

        dataset = tf.data.Dataset.from_tensor_slices(v_datadict)
        dataset = dataset.shuffle(n_sample)
        train_size = n_sample - valid_size
        train_vdataset = dataset.skip(valid_size).shuffle(train_size, reshuffle_each_iteration=True)
        valid_vdataset = dataset.take(valid_size).batch(valid_size)
        return train_vdataset, valid_vdataset

    def get_policydataset(self, policy, policy_type, update_init=False, random_sampling=False):
        policy_config = self.config["policy_config"]
        simul_data = self.simul_k_func(
            self.n_path, policy_config["T"], self.mparam, policy, policy_type,
            state_init=self.datadict
        )
        if update_init:
            self.update_from_simul(simul_data)
        p_datadict = {}
        idx_nan = False
        for k in self.keys:
            arr = simul_data[k].astype(NP_DTYPE)
            arr = arr[..., slice(-policy_config["t_sample"], -1, policy_config["t_skip"])]
            if len(arr.shape) == 3:
                arr = np.swapaxes(arr, 1, 2)
                arr = np.reshape(arr, (-1, self.mparam.n_agt))
                if k != "ishock":
                    idx_nan = np.logical_or(idx_nan, np.isnan(arr).any(axis=1))
            else:
                arr = np.reshape(arr, (-1, 1))
                if k != "ashock":
                    idx_nan = np.logical_or(idx_nan, np.isnan(arr[:, 0]))
            p_datadict[k] = arr
        for k in self.keys:
            p_datadict[k] = p_datadict[k][~idx_nan]
        if policy_config["opt_type"] == "game":
            p_datadict = crazyshuffle(p_datadict)

        if random_sampling:
            # TODO Improve sampling
            p_replacement=0.3
            log10_min=-2.0
            log10_max=2.0
            idx = np.random.uniform(0, 1, p_datadict["k_cross"].shape) < p_replacement
            p_datadict["k_cross"][idx] = np.power(10.0, np.random.uniform(log10_min, log10_max, idx.sum())).astype(NP_DTYPE)

        policy_ds = BasicDataSet(p_datadict)
        return policy_ds

    def simul_k_func(self, n_sample, T, mparam, c_policy, policy_type, state_init=None, shocks=None):
        raise NotImplementedError

    def simul_k_func(self, n_sample, T, mparam, c_policy, policy_type, state_init=None, shocks=None, func=None):
        raise NotImplementedError


class KSInitDataSet(InitDataSet):
    def __init__(self, mparam, config):
        super().__init__(mparam, config)
        mats = sio.loadmat(mparam.mats_path)
        self.splines = KS.construct_bspl(mats)
        self.keys = ["k_cross", "ashock", "ishock"]
        self.k_policy_bchmk = lambda k_cross, ashock, ishock: KS.k_policy_bspl(k_cross, ashock, ishock, self.splines)
        # the first burn for initialization
        self.update_with_burn(self.k_policy_bchmk, "pde", use_last=True)

    def get_valuedataset(self, policy, policy_type, update_init=False):
        value_config = self.config["value_config"]
        t_count = value_config["t_count"]
        t_skip = value_config["t_skip"]
        simul_data = self.simul_k_func(
            self.n_path, value_config["T"], self.mparam, policy, policy_type,
            state_init=self.datadict
        )
        if update_init:
            self.update_from_simul(simul_data)

        ashock, ishock = simul_data["ashock"], simul_data["ishock"]
        k_cross, csmp = simul_data["k_cross"], simul_data["csmp"]
        k_mean = np.mean(k_cross, axis=1, keepdims=True)
        # k_fm = self.compute_fm(k_cross) # n_path*n_fm*T
        discount = np.power(self.mparam.beta, np.arange(t_count))
        util = np.log(csmp)

        basic_s = np.zeros(shape=[0, self.mparam.n_agt, self.n_basic+1])
        agt_s = np.zeros(shape=[0, self.mparam.n_agt, 1])
        value = np.zeros(shape=[0, self.mparam.n_agt, 1])
        t_idx = 0
        while t_idx + t_count < value_config["T"]-1:
            k_tmp = k_cross[:, :, t_idx:t_idx+1]
            i_tmp = ishock[:, :, t_idx:t_idx+1]
            k_mean_tmp = np.repeat(k_mean[:, :, t_idx:t_idx+1], self.mparam.n_agt, axis=1)
            a_tmp = np.repeat(ashock[:, None, t_idx:t_idx+1], self.mparam.n_agt, axis=1)
            basic_s_tmp = np.concatenate([k_tmp, k_mean_tmp, a_tmp, i_tmp], axis=-1)
            v_tmp = np.sum(util[..., t_idx:t_idx+t_count]*discount, axis=-1, keepdims=True)

            basic_s = np.concatenate([basic_s, basic_s_tmp], axis=0)
            agt_s = np.concatenate([agt_s, k_tmp], axis=0)
            value = np.concatenate([value, v_tmp], axis=0)
            t_idx += t_skip

        v_datadict = {"basic_s": basic_s, "agt_s": agt_s, "value": value}
        train_vdataset, valid_vdataset = self.process_vdatadict(v_datadict)
        return train_vdataset, valid_vdataset

    def simul_k_func(self, n_sample, T, mparam, c_policy, policy_type, state_init=None, shocks=None, func=None):
        return KS.simul_k(n_sample, T, mparam, c_policy, policy_type, state_init, shocks, func)


class DavilaInitDataSet(InitDataSet):
    def __init__(self, mparam, config):
        super().__init__(mparam, config)
        mats = sio.loadmat(mparam.mats_path)
        self.splines = Davila.construct_spl(mats)
        self.keys = ["k_cross", "ashock", "ishock"]
        # state_init = {"k_cross": mparam.k_ss}
        self.k_policy_bchmk = lambda k_cross, ishock: Davila.k_policy_spl(k_cross, ishock, self.splines)
        # the first burn for initialization
        self.update_with_burn(self.k_policy_bchmk, "pde", use_last=True)

    def get_valuedataset(self, policy, policy_type, update_init=False):
        value_config = self.config["value_config"]
        t_count = value_config["t_count"]
        t_skip = value_config["t_skip"]
        simul_data = self.simul_k_func(
            self.n_path, value_config["T"], self.mparam, policy, policy_type,
            state_init=self.datadict
        )
        if update_init:
            self.update_from_simul(simul_data)

        ishock = simul_data["ishock"]
        k_cross, csmp = simul_data["k_cross"], simul_data["csmp"]
        k_mean = np.mean(k_cross, axis=1, keepdims=True)
        # k_fm = self.compute_fm(k_cross) # n_path*n_fm*T
        discount = np.power(self.mparam.beta, np.arange(t_count))
        util = 1 - 1/csmp

        basic_s = np.zeros(shape=[0, self.mparam.n_agt, self.n_basic+1])
        agt_s = np.zeros(shape=[0, self.mparam.n_agt, 1])
        value = np.zeros(shape=[0, self.mparam.n_agt, 1])
        t_idx = 0
        while t_idx + t_count < value_config["T"]-1:
            k_tmp = k_cross[:, :, t_idx:t_idx+1]
            i_tmp = ishock[:, :, t_idx:t_idx+1]
            k_mean_tmp = np.repeat(k_mean[:, :, t_idx:t_idx+1], self.mparam.n_agt, axis=1)
            # a_tmp = np.repeat(ashock[:, None, t_idx:t_idx+1], self.mparam.n_agt, axis=1)
            basic_s_tmp = np.concatenate([k_tmp, k_mean_tmp, i_tmp], axis=-1)
            v_tmp = np.sum(util[..., t_idx:t_idx+t_count]*discount, axis=-1, keepdims=True)

            basic_s = np.concatenate([basic_s, basic_s_tmp], axis=0)
            agt_s = np.concatenate([agt_s, k_tmp], axis=0)
            value = np.concatenate([value, v_tmp], axis=0)
            t_idx += t_skip

        v_datadict = {"basic_s": basic_s, "agt_s": agt_s, "value": value}
        train_vdataset, valid_vdataset = self.process_vdatadict(v_datadict, norm=False) # use log instead of norm in Davila's problem
        return train_vdataset, valid_vdataset

    def simul_k_func(self, n_sample, T, mparam, c_policy, policy_type, state_init=None, shocks=None, func=None):
        return Davila.simul_k(n_sample, T, mparam, c_policy, policy_type, state_init, shocks, func)


class DavilaASInitDataSet(InitDataSet):
    def __init__(self, mparam, config):
        super().__init__(mparam, config)
        mats = sio.loadmat(mparam.mats_path)
        policy_config = self.config["policy_config"]
        self.splines = DavilaAS.construct_spl(mats, policy_config)
        self.keys = ["k_cross", "ashock", "ishock"]
        self.k_policy_bchmk = \
            lambda k_cross, ashock, ishock: DavilaAS.k_policy_spl(k_cross, ashock, ishock, self.splines, policy_config)
        # the first burn for initialization
        self.update_with_burn(self.k_policy_bchmk, "pde")

    def get_valuedataset(self, policy, policy_type, update_init=False):
        value_config = self.config["value_config"]
        t_count = value_config["t_count"]
        t_skip = value_config["t_skip"]
        simul_data = self.simul_k_func(
            self.n_path, value_config["T"], self.mparam, policy, policy_type,
            state_init=self.datadict
        )
        ashock, ishock = simul_data["ashock"], simul_data["ishock"]
        if update_init:
            self.update_from_simul(simul_data)

        k_cross, csmp = simul_data["k_cross"], simul_data["csmp"]
        k_mean = np.mean(k_cross, axis=1, keepdims=True)
        discount = np.power(self.mparam.beta, np.arange(t_count))
        util = 1 - 1/csmp

        basic_s = np.zeros(shape=[0, self.mparam.n_agt, self.n_basic+1])
        agt_s = np.zeros(shape=[0, self.mparam.n_agt, 1])
        value = np.zeros(shape=[0, self.mparam.n_agt, 1])
        t_idx = 0
        while t_idx + t_count < value_config["T"]-1:
            k_tmp = k_cross[:, :, t_idx:t_idx+1]
            i_tmp = ishock[:, :, t_idx:t_idx+1]
            k_mean_tmp = np.repeat(k_mean[:, :, t_idx:t_idx+1], self.mparam.n_agt, axis=1)
            a_tmp = np.repeat(ashock[:, None, t_idx:t_idx+1], self.mparam.n_agt, axis=1)
            basic_s_tmp = np.concatenate([k_tmp, k_mean_tmp, a_tmp, i_tmp], axis=-1)
            v_tmp = np.sum(util[..., t_idx:t_idx+t_count]*discount, axis=-1, keepdims=True)

            basic_s = np.concatenate([basic_s, basic_s_tmp], axis=0)
            agt_s = np.concatenate([agt_s, k_tmp], axis=0)
            value = np.concatenate([value, v_tmp], axis=0)
            t_idx += t_skip

        v_datadict = {"basic_s": basic_s, "agt_s": agt_s, "value": value}
        train_vdataset, valid_vdataset = self.process_vdatadict(v_datadict)
        return train_vdataset, valid_vdataset

    def simul_k_func(self, n_sample, T, mparam, c_policy, policy_type, state_init=None, shocks=None):
        return DavilaAS.simul_k(n_sample, T, mparam, c_policy, policy_type, state_init, shocks)


class JFVInitDataSet(InitDataSet):
    def __init__(self, mparam, config):
        super().__init__(mparam, config)
        self.with_ashock = mparam.with_ashock
        self.keys = ["k_cross", "N", "ishock"]
        mats = sio.loadmat(mparam.mats_path)
        if self.with_ashock:
            self.splines = JFV.construct_spl_SSS(mats, 'c')
            self.c_policy_bchmk = lambda k_cross, N, ishock: JFV.c_policy_spl_SSS(k_cross, N, ishock, self.splines)
            # state_init = {"k_cross": mparam.B_sss, "N": mparam.N_sss}
        else:
            self.splines = JFV.construct_spl_DSS(mats, 'c')
            self.c_policy_bchmk = lambda k_cross, N, ishock: JFV.c_policy_spl_DSS(k_cross, N, ishock, self.splines)
            # state_init = {"k_cross": mparam.k_dss, "N": mparam.N_dss}
        # the first burn for initialization
        self.update_with_burn(self.c_policy_bchmk, "pde")

    def get_valuedataset(self, policy, policy_type, update_init=False):
        value_config = self.config["value_config"]
        t_count = value_config["t_count"]
        t_skip = value_config["t_skip"]
        simul_data = self.simul_k_func(
            self.n_path, value_config["T"], self.mparam, policy, policy_type,
            state_init=self.datadict
        )
        if update_init:
            self.update_from_simul(simul_data)

        ishock = simul_data["ishock"]
        k_cross, csmp = simul_data["k_cross"], simul_data["csmp"]
        k_mean = np.mean(k_cross, axis=1, keepdims=True)
        discount = np.power(self.mparam.beta, np.arange(t_count))
        util = 1 - 1/csmp

        basic_s = np.zeros(shape=[0, self.mparam.n_agt, self.n_basic+1])
        agt_s = np.zeros(shape=[0, self.mparam.n_agt, 1])
        value = np.zeros(shape=[0, self.mparam.n_agt, 1])
        t_idx = 0
        while t_idx + t_count < value_config["T"]-1:
            k_tmp = k_cross[:, :, t_idx:t_idx+1]
            i_tmp = ishock[:, :, t_idx:t_idx+1]
            k_mean_tmp = np.repeat(k_mean[:, :, t_idx:t_idx+1], self.mparam.n_agt, axis=1)
            N_tmp = np.repeat(simul_data["N"][:, None, t_idx:t_idx+1], self.mparam.n_agt, axis=1)
            basic_s_tmp = np.concatenate([k_tmp, k_mean_tmp, N_tmp, i_tmp], axis=-1)
            v_tmp = np.sum(util[..., t_idx:t_idx+t_count]*discount, axis=-1, keepdims=True) * self.mparam.dt

            basic_s = np.concatenate([basic_s, basic_s_tmp], axis=0)
            agt_s = np.concatenate([agt_s, k_tmp], axis=0)
            value = np.concatenate([value, v_tmp], axis=0)
            t_idx += t_skip

        v_datadict = {"basic_s": basic_s, "agt_s": agt_s, "value": value}
        train_vdataset, valid_vdataset = self.process_vdatadict(v_datadict)
        return train_vdataset, valid_vdataset

    # def get_policydataset(self, policy, policy_type, update_init=False):
    #     # it only include k_cross and N for policy
    #     policy_config = self.config["policy_config"]
    #     simul_data = self.simul_k_func(
    #         self.n_path, policy_config["T"], self.mparam, policy, policy_type,
    #         state_init=self.datadict
    #     )
    #     if update_init:
    #         self.update_from_simul(simul_data)

    #     k_cross = simul_data["k_cross"].astype(NP_DTYPE)
    #     k_cross = np.swapaxes(k_cross[..., slice(-policy_config["t_sample"], -1, policy_config["t_skip"])], 1, 2)
    #     k_cross = np.reshape(k_cross, (-1, self.mparam.n_agt))
    #     if policy_config["opt_type"] == "game":
    #         k_cross = crazyshuffle(k_cross)
    #     N = simul_data["N"][:, slice(-policy_config["t_sample"], -1, policy_config["t_skip"])].astype(NP_DTYPE)
    #     N = np.reshape(N, (-1, 1))
    #     idx_nan = np.isnan(k_cross).any(axis=1)
    #     idx_nan = np.logical_or(idx_nan, np.isnan(N[:, 0]))
    #     p_datadict = {"k_cross": k_cross[~idx_nan], "N": N[~idx_nan]}
    #     policy_ds = BasicDataSet(p_datadict)
    #     return policy_ds

    def simul_k_func(self, n_sample, T, mparam, c_policy, policy_type, state_init=None, shocks=None):
        return JFV.simul_k(n_sample, T, mparam, c_policy, policy_type, state_init, shocks)


def crazyshuffle(data):
    assert data["k_cross"].shape == data["ishock"].shape
    x, y = data["k_cross"].shape
    rows = np.indices((x, y))[0]
    cols = [np.random.permutation(y) for _ in range(x)]
    data["k_cross"] = data["k_cross"][rows, cols]
    data["ishock"] = data["ishock"][rows, cols]
    return data
