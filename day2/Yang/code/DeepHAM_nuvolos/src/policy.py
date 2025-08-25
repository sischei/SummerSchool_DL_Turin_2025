import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import util
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
log_10 = tf.cast(tf.math.log(10.0), DTYPE)
eps = tf.cast(1e-8, DTYPE)


class PolicyTrainer():
    def __init__(self, vtrainers, init_ds, policy_path=None):
        self.config = init_ds.config
        self.policy_config = self.config["policy_config"]
        self.t_unroll = self.policy_config["t_unroll"]
        self.vtrainers = vtrainers
        self.valid_size = self.policy_config["valid_size"]
        self.sgm_scale = self.policy_config["sgm_scale"] # scaling param in sigmoid
        self.init_ds = init_ds
        self.value_sampling = self.config["dataset_config"]["value_sampling"]
        self.num_vnet = len(vtrainers)
        self.mparam = init_ds.mparam
        self.model = util.FeedforwardModel(vtrainers[0].d_in, 1, self.policy_config, name="p_net")
        if self.config["n_gm"] > 0:
            # TODO generalize to multi-dimensional agt_s
            self.gm_model = util.GeneralizedMomModel(1, self.config["n_gm"], self.config["gm_config"], name="p_gm")
        self.train_vars = None
        if policy_path is not None:
            self.model.load_weights_after_init(policy_path)
            if self.config["n_gm"] > 0:
                self.gm_model.load_weights_after_init(policy_path.replace(".weights.h5", "_gm.weights.h5"))
            self.init_ds.load_stats(os.path.dirname(policy_path))
        self.discount = np.power(self.mparam.beta, np.arange(self.t_unroll))
        # to be generated in the child class
        self.policy_ds = None
        self.use_log_k = self.config.get("use_log_k", False)
        self.grad_penalty = self.policy_config.get("grad_penalty", False)
        self.train_loss_metric = tf.keras.metrics.Mean('train_loss', dtype=log_10.dtype)
        self.valid_loss_metric = tf.keras.metrics.Mean('valid_loss', dtype=log_10.dtype)
        train_log_dir = os.path.join(self.config["model_path"], 'logs/', self.config["current_time"], 'pnet_train')
        valid_log_dir = os.path.join(self.config["model_path"], 'logs/', self.config["current_time"], 'pnet_valid')
        self.train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        self.valid_summary_writer = tf.summary.create_file_writer(valid_log_dir)

    @tf.function
    def prepare_state(self, input_data):
        if self.use_log_k:
            log_k = tf.math.log(tf.cast(input_data["basic_s"][..., 0:1], DTYPE) + eps)/log_10
            log_k_mean = tf.math.reduce_mean(log_k, axis=1 ,keepdims=True)
            log_k_mean = tf.tile(log_k_mean, [1, input_data["basic_s"].shape[1], 1])
            basic_s = tf.concat([log_k, log_k_mean, tf.cast(input_data["basic_s"][..., 2:], DTYPE)], axis=-1)
            agt_s = tf.math.log(tf.cast(input_data["agt_s"], DTYPE))/log_10
        else:
            basic_s = tf.cast(input_data["basic_s"], DTYPE)
            agt_s = tf.cast(input_data["agt_s"], DTYPE)
        if self.config.get("full_state", False):
            state = tf.concat(
                [basic_s[..., 0:1], basic_s[..., 2:],
                 tf.repeat(tf.transpose(tf.cast(input_data["agt_s"], DTYPE), perm=[0, 2, 1]), self.config["n_agt"], axis=-2)],
                 axis=-1
            )
        elif self.config["n_fm"] == 2:
            k_var = tf.math.reduce_variance(agt_s, axis=-2, keepdims=True)
            k_var = tf.tile(k_var, [1, agt_s.shape[1], 1])
            state = tf.concat([basic_s, k_var], axis=-1)
        elif self.config["n_fm"] == 0:
            state = tf.concat([basic_s[..., 0:1], basic_s[..., 2:]], axis=-1)
        elif self.config["n_fm"] == 1:  # so far always add k_mean in the basic_state
            state = basic_s
        if self.config["n_gm"] > 0:
            gm = self.gm_model(agt_s)
            state = tf.concat([state, gm], axis=-1)
        return state

    @tf.function
    def policy_fn(self, input_data):
        state = self.prepare_state(input_data)
        policy = tf.sigmoid(self.sgm_scale*self.model(state))
        return policy

    @tf.function
    def loss(self, input_data):
        raise NotImplementedError

    def grad(self, input_data):
        with tf.GradientTape(persistent=True) as tape:
            output_dict = self.loss(input_data)
            if self.grad_penalty:
                total_loss = output_dict["m_util"] + 0.1 * output_dict["gp_loss"] # TODO the weight of gp_loss can be modified
            else:
                total_loss = output_dict["m_util"]
        self.train_loss_metric(-output_dict["m_util"])
        train_vars = self.model.trainable_variables
        if self.config["n_gm"] > 0:
            train_vars += self.gm_model.trainable_variables
        self.train_vars = train_vars
        grad = tape.gradient(
            total_loss,
            train_vars,
            unconnected_gradients=tf.UnconnectedGradients.ZERO,
        )
        del tape
        return grad, output_dict["k_end"]

    @tf.function
    def train_step(self, train_data):
        grad, k_end = self.grad(train_data)
        self.optimizer.apply_gradients(
            zip(grad, self.train_vars)
        )
        return k_end

    def train(self, num_step=None, batch_size=None):
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            self.policy_config["lr_beg"],
            decay_steps=num_step,
            decay_rate=self.policy_config["lr_end"] / self.policy_config["lr_beg"],
            staircase=False,
        )
        assert batch_size <= self.valid_size, "The valid size should be no smaller than batch_size."
        self.optimizer = tf.keras.optimizers.Adam(  # pylint: disable=W0201
            learning_rate=lr_schedule, epsilon=1e-8,
            beta_1=0.99, beta_2=0.99
        )

        # TODO: currenly assuming valid_size = n_path in self.init_ds
        valid_data = dict((k, self.init_ds.datadict[k].astype(NP_DTYPE)) for k in self.init_ds.keys)
        ashock, ishock = self.simul_shocks(
            self.valid_size, self.t_unroll, self.mparam,
            state_init=self.init_ds.datadict
        )
        valid_data["ashock"] = ashock.astype(NP_DTYPE)
        valid_data["ishock"] = ishock.astype(NP_DTYPE)

        freq_valid = self.policy_config["freq_valid"]
        n_epoch = num_step // freq_valid
        update_init = False
        if tf.config.list_physical_devices('GPU'):
            print(tf.config.experimental.get_memory_info('GPU:0'))
        for n in range(n_epoch):
            for step in tqdm(range(freq_valid)):
                train_data = self.sampler(batch_size, update_init)
                k_end = self.train_step(train_data)
                n_step = n*freq_valid + step
                if self.value_sampling != "bchmk" and n_step % self.policy_config["freq_update_v"] == 0 and n_step > 0:
                    update_init = self.policy_config["update_init"]
                    train_vds, valid_vds = self.get_valuedataset(update_init)
                    for vtr in self.vtrainers:
                        vtr.train(
                            train_vds, valid_vds,
                            self.config["value_config"]["num_epoch"],
                            self.config["value_config"]["batch_size"],
                        )
                    if tf.config.list_physical_devices('GPU'):
                        print(tf.config.experimental.get_memory_info('GPU:0'))
                with self.train_summary_writer.as_default():
                    tf.summary.scalar('loss', self.train_loss_metric.result(), step=n_step)
                self.train_loss_metric.reset_state()
            val_output = self.loss(valid_data)
            print(
                "Step: %d, valid util: %g, k_end: %g" %
                (freq_valid*(n+1), -val_output["m_util"], k_end)
            )
            self.valid_loss_metric(-val_output["m_util"])
            with self.valid_summary_writer.as_default():
                tf.summary.scalar('loss', self.valid_loss_metric.result(), step=n_step)
            self.valid_loss_metric.reset_state()

    def save_model(self, path="policy_model"):
        self.model.save_weights(path)
        self.init_ds.save_stats(os.path.dirname(path))
        if self.config["n_gm"] > 0:
            self.gm_model.save_weights(path.replace(".weights.h5", "_gm.weights.h5"))

    def simul_shocks(self, n_sample, T, mparam, state_init):
        raise NotImplementedError

    def sampler(self, batch_size, update_init=False):
        train_data = self.policy_ds.next_batch(batch_size)
        ashock, ishock = self.simul_shocks(batch_size, self.t_unroll, self.mparam, train_data)
        train_data["ashock"] = ashock.astype(NP_DTYPE)
        train_data["ishock"] = ishock.astype(NP_DTYPE)
        # TODO test the effect of epoch_resample
        if self.policy_ds.epoch_used > self.policy_config["epoch_resample"]:
            self.update_policydataset(update_init)
        return train_data

    def update_policydataset(self, update_init=False):
        raise NotImplementedError

    def get_valuedataset(self, update_init=False):
        raise NotImplementedError


class KSPolicyTrainer(PolicyTrainer):
    def __init__(self, vtrainers, init_ds, policy_path=None):
        super().__init__(vtrainers, init_ds, policy_path)
        if self.config["init_with_bchmk"]:
            init_policy = self.init_ds.k_policy_bchmk
            policy_type = "pde"
        else:
            init_policy = self.init_ds.c_policy_const_share
            policy_type = "nn_share"
        self.policy_ds = self.init_ds.get_policydataset(init_policy, policy_type, update_init=False)

    @tf.function
    def loss(self, input_data):
        k_cross = input_data["k_cross"]
        ashock, ishock = input_data["ashock"], input_data["ishock"]
        util_sum = 0

        for t in range(self.t_unroll):
            k_mean = tf.reduce_mean(k_cross, axis=1, keepdims=True)
            k_mean_tmp = tf.tile(k_mean, [1, self.mparam.n_agt])
            k_mean_tmp = tf.expand_dims(k_mean_tmp, axis=-1)
            i_tmp = ishock[:, :, t:t+1] # n_path*n_agt*1
            a_tmp = tf.tile(ashock[:, t:t+1], [1, self.mparam.n_agt])
            a_tmp = tf.expand_dims(a_tmp, axis=2) # n_path*n_agt*1
            basic_s_tmp = tf.concat([tf.expand_dims(k_cross, axis=-1), k_mean_tmp, a_tmp, i_tmp], axis=-1)
            basic_s_tmp = self.init_ds.normalize_data(basic_s_tmp, key="basic_s", withtf=True)
            full_state_dict = {
                "basic_s": basic_s_tmp,
                "agt_s": self.init_ds.normalize_data(tf.expand_dims(k_cross, axis=-1), key="agt_s", withtf=True)
            }
            if t == self.t_unroll - 1:
                value = 0
                for vtr in self.vtrainers:
                    value += self.init_ds.unnormalize_data(
                        vtr.value_fn(full_state_dict)[..., 0], key="value", withtf=True)
                value /= self.num_vnet
                util_sum += self.discount[t]*value
                continue

            c_share = self.policy_fn(full_state_dict)[..., 0]
            if self.policy_config["opt_type"] == "game":
                # optimizing agent 0 only
                c_share = tf.concat([c_share[:, 0:1], tf.stop_gradient(c_share[:, 1:])], axis=1)
            # labor tax rate - depend on ashock
            tau = tf.where(ashock[:, t:t+1] < 1, self.mparam.tau_b, self.mparam.tau_g)
            # total labor supply - depend on ashock
            emp = tf.where(
                ashock[:, t:t+1] < 1,
                self.mparam.l_bar*self.mparam.er_b,
                self.mparam.l_bar*self.mparam.er_g
            )
            tau, emp = tf.cast(tau, DTYPE), tf.cast(emp, DTYPE)
            R = 1 - self.mparam.delta + ashock[:, t:t+1] * self.mparam.alpha*(k_mean / emp)**(self.mparam.alpha-1)
            wage = ashock[:, t:t+1]*(1-self.mparam.alpha)*(k_mean / emp)**(self.mparam.alpha)
            wealth = R * k_cross + (1-tau)*wage*self.mparam.l_bar*ishock[:, :, t] + \
                self.mparam.mu*wage*(1-ishock[:, :, t])
            csmp = tf.clip_by_value(c_share * wealth, EPSILON, wealth-EPSILON)
            k_cross = wealth - csmp
            util_sum += self.discount[t] * tf.math.log(csmp)

        if self.policy_config["opt_type"] == "socialplanner":
            output_dict = {"m_util": -tf.reduce_mean(util_sum), "k_end": tf.reduce_mean(k_cross)}
        elif self.policy_config["opt_type"] == "game":
            # optimizing agent 0 only
            output_dict = {"m_util": -tf.reduce_mean(util_sum[:, 0]), "k_end": tf.reduce_mean(k_cross)}
        return output_dict

    def update_policydataset(self, update_init=False):
        self.policy_ds = self.init_ds.get_policydataset(self.current_c_policy, "nn_share", update_init)

    def get_valuedataset(self, update_init=False):
        return self.init_ds.get_valuedataset(self.current_c_policy, "nn_share", update_init)

    def current_c_policy(self, k_cross, ashock, ishock):
        k_mean = np.mean(k_cross, axis=1, keepdims=True)
        k_mean = np.repeat(k_mean, self.mparam.n_agt, axis=1)
        ashock = np.repeat(ashock, self.mparam.n_agt, axis=1)
        basic_s = np.stack([k_cross, k_mean, ashock, ishock], axis=-1)
        basic_s = self.init_ds.normalize_data(basic_s, key="basic_s")
        basic_s = basic_s.astype(NP_DTYPE)
        full_state_dict = {
            "basic_s": basic_s,
            "agt_s": self.init_ds.normalize_data(k_cross[:, :, None], key="agt_s")
        }
        c_share = self.policy_fn(full_state_dict)[..., 0]
        return c_share

    def simul_shocks(self, n_sample, T, mparam, state_init):
        return KS.simul_shocks(n_sample, T, mparam, state_init)


class DavilaPolicyTrainer(PolicyTrainer):
    def __init__(self, vtrainers, init_ds, policy_path=None):
        super().__init__(vtrainers, init_ds, policy_path)
        if self.config["init_with_bchmk"]:
            init_policy = self.init_ds.k_policy_bchmk
            policy_type = "pde"
        else:
            init_policy = self.init_ds.c_policy_const_share
            policy_type = "nn_share"
        self.policy_ds = self.init_ds.get_policydataset(init_policy, policy_type, update_init=False, random_sampling=True)

    @tf.function
    def loss(self, input_data):
        k_cross = input_data["k_cross"]
        #k_cross = tf.constant(input_data["k_cross"])
        ishock = input_data["ishock"]
        util_sum = 0
        gp_loss = 0

        for t in range(self.t_unroll):
            with tf.GradientTape() as tape:
                tape.watch(k_cross)
                k_mean = tf.reduce_mean(k_cross, axis=1, keepdims=True)
                k_mean_tmp = tf.tile(k_mean, [1, self.mparam.n_agt])
                k_mean_tmp = tf.expand_dims(k_mean_tmp, axis=-1)
                i_tmp = ishock[:, :, t:t+1] # n_path*n_agt*1
                # a_tmp = tf.tile(ashock[:, t:t+1], [1, self.mparam.n_agt])
                # a_tmp = tf.expand_dims(a_tmp, axis=2) # n_path*n_agt*1
                basic_s_tmp = tf.concat([tf.expand_dims(k_cross, axis=-1), k_mean_tmp, i_tmp], axis=-1)
                #basic_s_tmp = self.init_ds.normalize_data(basic_s_tmp, key="basic_s", withtf=True)
                full_state_dict = {
                    "basic_s": basic_s_tmp,
                    "agt_s": tf.expand_dims(k_cross, axis=-1)
                    #"agt_s": self.init_ds.normalize_data(tf.expand_dims(k_cross, axis=-1), key="agt_s", withtf=True)
                }
                if t == self.t_unroll - 1:
                    value = 0
                    for vtr in self.vtrainers:
                        value += self.init_ds.unnormalize_data(
                            vtr.value_fn(full_state_dict)[..., 0], key="value", withtf=True)
                    value /= self.num_vnet
                    util_sum += self.discount[t]*value
                    continue
                c_share = self.policy_fn(full_state_dict)[..., 0]
                if self.policy_config["opt_type"] == "game":
                    # optimizing agent 0 only
                    c_share = tf.concat([c_share[:, 0:1], tf.stop_gradient(c_share[:, 1:])], axis=1)
                # total labor supply, emp_g = emp_b
                emp = tf.cast(self.mparam.emp_g, DTYPE)
                R = 1 - self.mparam.delta + self.mparam.alpha*(k_mean / emp)**(self.mparam.alpha-1)
                wage = (1-self.mparam.alpha)*(k_mean / emp)**(self.mparam.alpha)
                wealth_gp = tf.stop_gradient(R) * k_cross + tf.stop_gradient(wage) * (
                    self.mparam.epsilon_0*(1-ishock[:, :, t])*(2-ishock[:, :, t])/2 + \
                    self.mparam.epsilon_1*ishock[:, :, t]*(2-ishock[:, :, t]) + \
                    self.mparam.epsilon_2*ishock[:, :, t]*(ishock[:, :, t]-1)/2
                )
                csmp_gp = c_share * wealth_gp
            gradients = tape.gradient(csmp_gp, k_cross) * log_10 * k_cross
            gp_loss += tf.keras.activations.relu(-gradients)
            wealth = R * k_cross + wage * (
                self.mparam.epsilon_0*(1-ishock[:, :, t])*(2-ishock[:, :, t])/2 + \
                self.mparam.epsilon_1*ishock[:, :, t]*(2-ishock[:, :, t]) + \
                self.mparam.epsilon_2*ishock[:, :, t]*(ishock[:, :, t]-1)/2
            )
            csmp = c_share * wealth
            k_cross = wealth - csmp
            util_sum += self.discount[t] * (1 - 1/csmp)

        if self.policy_config["opt_type"] == "socialplanner":
            output_dict = {"m_util": -tf.reduce_mean(util_sum), "gp_loss": tf.reduce_mean(gp_loss), "k_end": tf.reduce_mean(k_cross)}
        elif self.policy_config["opt_type"] == "game":
            # optimizing agent 0 only
            # TODO game gp_loss ???
            output_dict = {"m_util": -tf.reduce_mean(util_sum[:, 0]), "gp_loss": tf.reduce_mean(gp_loss), "k_end": tf.reduce_mean(k_cross)}
        return output_dict

    def update_policydataset(self, update_init=True):
        self.policy_ds = self.init_ds.get_policydataset(self.current_c_policy, "nn_share", update_init, random_sampling=True)

    def get_valuedataset(self, update_init=True):
        return self.init_ds.get_valuedataset(self.current_c_policy, "nn_share", update_init)

    def current_c_policy(self, k_cross, ishock):
        k_mean = np.mean(k_cross, axis=1, keepdims=True)
        k_mean = np.repeat(k_mean, self.mparam.n_agt, axis=1)
        # ashock = np.repeat(ashock, self.mparam.n_agt, axis=1)
        basic_s = np.stack([k_cross, k_mean, ishock], axis=-1)
        #basic_s = self.init_ds.normalize_data(basic_s, key="basic_s")
        basic_s = basic_s.astype(NP_DTYPE)
        full_state_dict = {
            "basic_s": basic_s,
            "agt_s": k_cross[:, :, None]
            #"agt_s": self.init_ds.normalize_data(k_cross[:, :, None], key="agt_s")
        }
        c_share = self.policy_fn(full_state_dict)[..., 0]
        return c_share

    def simul_shocks(self, n_sample, T, mparam, state_init):
        return Davila.simul_shocks(n_sample, T, mparam, state_init)


class DavilaASPolicyTrainer(PolicyTrainer):
    def __init__(self, vtrainers, init_ds, policy_path=None):
        super().__init__(vtrainers, init_ds, policy_path)
        if self.config["init_with_bchmk"]:
            init_policy = self.init_ds.k_policy_bchmk
            policy_type = "pde"
        else:
            init_policy = self.init_ds.c_policy_const_share
            policy_type = "nn_share"
        self.policy_ds = self.init_ds.get_policydataset(init_policy, policy_type, update_init=False)

    @tf.function
    def loss(self, input_data):
        k_cross = input_data["k_cross"]
        ashock, ishock = input_data["ashock"], input_data["ishock"]
        util_sum = 0

        for t in range(self.t_unroll):
            k_mean = tf.reduce_mean(k_cross, axis=1, keepdims=True)
            k_mean_tmp = tf.tile(k_mean, [1, self.mparam.n_agt])
            k_mean_tmp = tf.expand_dims(k_mean_tmp, axis=-1)
            i_tmp = ishock[:, :, t:t+1] # n_path*n_agt*1
            a_tmp = tf.tile(ashock[:, t:t+1], [1, self.mparam.n_agt])
            a_tmp = tf.expand_dims(a_tmp, axis=2) # n_path*n_agt*1
            basic_s_tmp = tf.concat([tf.expand_dims(k_cross, axis=-1), k_mean_tmp, a_tmp, i_tmp], axis=-1)
            basic_s_tmp = self.init_ds.normalize_data(basic_s_tmp, key="basic_s", withtf=True)
            full_state_dict = {
                "basic_s": basic_s_tmp,
                "agt_s": self.init_ds.normalize_data(tf.expand_dims(k_cross, axis=-1), key="agt_s", withtf=True)
            }
            if t == self.t_unroll - 1:
                value = 0
                for vtr in self.vtrainers:
                    value += self.init_ds.unnormalize_data(
                        vtr.value_fn(full_state_dict)[..., 0], key="value", withtf=True)
                value /= self.num_vnet
                util_sum += self.discount[t]*value
                continue

            c_share = self.policy_fn(full_state_dict)[..., 0]
            if self.policy_config["opt_type"] == "game":
                # optimizing agent 0 only
                c_share = tf.concat([c_share[:, 0:1], tf.stop_gradient(c_share[:, 1:])], axis=1)
            # total labor supply
            emp = tf.where(ashock[:, t:t+1] < 1, self.mparam.emp_b, self.mparam.emp_g)
            emp = tf.cast(emp, DTYPE)
            R = 1 - self.mparam.delta + ashock[:, t:t+1]*self.mparam.alpha*(k_mean / emp)**(self.mparam.alpha-1)
            wage = ashock[:, t:t+1]*(1-self.mparam.alpha)*(k_mean / emp)**(self.mparam.alpha)
            wealth = R * k_cross + wage * (
                self.mparam.epsilon_0*(1-ishock[:, :, t])*(2-ishock[:, :, t])/2 + \
                self.mparam.epsilon_1*ishock[:, :, t]*(2-ishock[:, :, t]) + \
                self.mparam.epsilon_2*ishock[:, :, t]*(ishock[:, :, t]-1)/2
            )
            csmp = tf.clip_by_value(c_share * wealth, EPSILON, wealth-EPSILON)
            k_cross = wealth - csmp
            util_sum += self.discount[t] * (1 - 1/csmp)

        if self.policy_config["opt_type"] == "socialplanner":
            output_dict = {"m_util": -tf.reduce_mean(util_sum), "k_end": tf.reduce_mean(k_cross)}
        elif self.policy_config["opt_type"] == "game":
            # optimizing agent 0 only
            output_dict = {"m_util": -tf.reduce_mean(util_sum[:, 0]), "k_end": tf.reduce_mean(k_cross)}
        return output_dict

    def update_policydataset(self, update_init=False):
        self.policy_ds = self.init_ds.get_policydataset(self.current_c_policy, "nn_share", update_init)

    def get_valuedataset(self, update_init=False):
        return self.init_ds.get_valuedataset(self.current_c_policy, "nn_share", update_init)

    def current_c_policy(self, k_cross, ashock, ishock):
        k_mean = np.mean(k_cross, axis=1, keepdims=True)
        k_mean = np.repeat(k_mean, self.mparam.n_agt, axis=1)
        ashock = np.repeat(ashock, self.mparam.n_agt, axis=1)
        basic_s = np.stack([k_cross, k_mean, ashock, ishock], axis=-1)
        basic_s = self.init_ds.normalize_data(basic_s, key="basic_s")
        basic_s = basic_s.astype(NP_DTYPE)
        full_state_dict = {
            "basic_s": basic_s,
            "agt_s": self.init_ds.normalize_data(k_cross[:, :, None], key="agt_s")
        }
        c_share = self.policy_fn(full_state_dict)[..., 0]
        return c_share

    def simul_shocks(self, n_sample, T, mparam, state_init):
        return DavilaAS.simul_shocks(n_sample, T, mparam, state_init)


class JFVPolicyTrainer(PolicyTrainer):
    def __init__(self, vtrainers, init_ds, policy_path=None):
        super().__init__(vtrainers, init_ds, policy_path)
        if self.config["init_with_bchmk"]:
            init_policy = self.init_ds.c_policy_bchmk
            policy_type = "pde"
        else:
            init_policy = self.init_ds.c_policy_const_share
            policy_type = "nn_share"
        self.policy_ds = self.init_ds.get_policydataset(init_policy, policy_type, update_init=False)
        self.with_ashock = self.mparam.with_ashock

    @tf.function
    def loss(self, input_data):
        k_cross, N = input_data["k_cross"], input_data["N"]
        ashock, ishock = input_data["ashock"], input_data["ishock"]
        util_sum = 0

        for t in range(self.t_unroll):
            k_mean = tf.reduce_mean(k_cross, axis=1, keepdims=True)
            k_mean_tmp = tf.tile(k_mean, [1, self.mparam.n_agt])
            k_mean_tmp = tf.expand_dims(k_mean_tmp, axis=-1)
            i_tmp = ishock[:, :, t:t+1]
            N_tmp = tf.tile(N, [1, self.mparam.n_agt])
            N_tmp = tf.expand_dims(N_tmp, axis=-1) # n_path*n_agt*1
            basic_s_tmp = tf.concat([tf.expand_dims(k_cross, axis=-1), k_mean_tmp, N_tmp, i_tmp], axis=-1)
            basic_s_tmp = self.init_ds.normalize_data(basic_s_tmp, key="basic_s", withtf=True)
            full_state_dict = {
                "basic_s": basic_s_tmp,
                "agt_s": self.init_ds.normalize_data(tf.expand_dims(k_cross, axis=-1), key="agt_s", withtf=True)
            }
            if t == self.t_unroll - 1:
                value = 0
                for vtr in self.vtrainers:
                    value += self.init_ds.unnormalize_data(
                        vtr.value_fn(full_state_dict)[..., 0], key="value", withtf=True)
                value /= self.num_vnet
                util_sum = util_sum * self.mparam.dt + self.discount[t]*value
                continue

            c_share = self.policy_fn(full_state_dict)[..., 0]
            if self.policy_config["opt_type"] == "game":
                # optimizing agent 0 only
                c_share = tf.concat([c_share[:, 0:1], tf.stop_gradient(c_share[:, 1:])], axis=1)

            K = N + k_mean
            wage_unit = (1 - self.mparam.alpha) * K**self.mparam.alpha
            r = self.mparam.alpha * K**(self.mparam.alpha-1) - self.mparam.delta - self.mparam.sigma2*K/N
            wage = (ishock[:, :, t] * (self.mparam.z2-self.mparam.z1) + self.mparam.z1) * wage_unit  # map 0/1 to z1/z2
            wealth = (1 + r*self.mparam.dt) * k_cross + wage * self.mparam.dt
            csmp = tf.clip_by_value(c_share * wealth / self.mparam.dt, EPSILON, wealth/self.mparam.dt-EPSILON)
            k_cross = wealth - csmp * self.mparam.dt
            dN_drift = self.mparam.dt * (self.mparam.alpha * K**(self.mparam.alpha-1) - self.mparam.delta - \
                self.mparam.rhohat - self.mparam.sigma2*(-k_mean/N)*(K/N))*N
            dN_diff = K * ashock[:, t:t+1]
            N = tf.maximum(N + dN_drift + dN_diff, 0.01)
            util_sum += self.discount[t] * (1 - 1/csmp)

        if self.policy_config["opt_type"] == "socialplanner":
            output_dict = {"m_util": -tf.reduce_mean(util_sum), "k_end": tf.reduce_mean(k_cross)}
        elif self.policy_config["opt_type"] == "game":
            # optimizing agent 0 only
            output_dict = {"m_util": -tf.reduce_mean(util_sum[:, 0]), "k_end": tf.reduce_mean(k_cross)}
        return output_dict

    def update_policydataset(self, update_init=False):
        # self.policy_ds = self.init_ds.get_policydataset(self.init_ds.c_policy_bchmk, "pde", update_init)
        self.policy_ds = self.init_ds.get_policydataset(self.current_c_policy, "nn_share", update_init)

    def get_valuedataset(self, update_init=False):
        return self.init_ds.get_valuedataset(self.current_c_policy, "nn_share", update_init)

    def current_c_policy(self, k_cross, N, ishock):
        k_mean = np.mean(k_cross, axis=1, keepdims=True)
        k_mean = np.repeat(k_mean, self.mparam.n_agt, axis=1)
        N_tmp = np.repeat(N, self.mparam.n_agt, axis=1)
        basic_s = np.stack([k_cross, k_mean, N_tmp, ishock], axis=-1)
        basic_s = self.init_ds.normalize_data(basic_s, key="basic_s")
        basic_s = basic_s.astype(NP_DTYPE)
        full_state_dict = {
            "basic_s": basic_s,
            "agt_s": self.init_ds.normalize_data(k_cross[:, :, None], key="agt_s")
        }
        c_share = self.policy_fn(full_state_dict)[..., 0]
        return c_share

    def simul_shocks(self, n_sample, T, mparam, state_init):
        return JFV.simul_shocks(n_sample, T, mparam, state_init)
