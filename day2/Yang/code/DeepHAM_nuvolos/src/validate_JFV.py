import json
import os
import numpy as np
import scipy.io as sio
from absl import app
from absl import flags
from param import JFVParam
from dataset import JFVInitDataSet
from value import ValueTrainer
from policy import JFVPolicyTrainer
import simulation_JFV as JFV
from util import set_random_seed  # Import the random seed function
import pandas as pd
import re

flags.DEFINE_string("model_path", "../data/simul_results/JFV_DSS/game_nn_n50_test",
                    """The path to load json file.""",
                    short_name='m')
flags.DEFINE_integer('n_agt', -1, "Number of agents in validation simulation")
flags.DEFINE_boolean('save', False, "Save simulation results or not.")
FLAGS = flags.FLAGS

def bellman_err(simul_data, shocks, ptrainer, value_fn, prefix="", nnext=500, nc=10, seed=None, nt=None):
    if seed:
        set_random_seed(seed)  # Use the seed function here
    k_cross, csmp, B, N = simul_data["k_cross"], simul_data["csmp"], simul_data["B"], simul_data["N"]
    if nt:
        nt = min(nt, csmp.shape[-1])
    else:
        nt = csmp.shape[-1]
    mparam = ptrainer.mparam
    # compute error for n_path * n_agt * nt states
    t_idx = np.random.choice(csmp.shape[-1], nt)
    n_agt = csmp.shape[1]
    k_now, k_next = k_cross[:, :, t_idx], k_cross[:, :, t_idx+1]
    knormed_now = ptrainer.init_ds.normalize_data(k_now, key="agt_s")
    knormed_next = ptrainer.init_ds.normalize_data(k_next, key="agt_s")
    knormed_mean_now = np.repeat(np.mean(knormed_now, axis=-2, keepdims=True), n_agt, axis=1)
    knormed_mean_next = np.repeat(np.mean(knormed_next, axis=-2, keepdims=True), n_agt, axis=1)
    B_now, B_next = np.repeat(B[:, None, t_idx], n_agt, axis=1), np.repeat(B[:, None, t_idx+1], n_agt, axis=1)
    N_now, N_next = np.repeat(N[:, None, t_idx], n_agt, axis=1), np.repeat(N[:, None, t_idx+1], n_agt, axis=1)
    c_now = csmp[:, :, t_idx]
    K_now = B_now + N_now
    ashock = shocks[0][:, t_idx]
    ishock = shocks[1][:, :, t_idx]

    ashock_next = mparam.dt**0.5*np.random.normal(0, mparam.sigma, [ashock.shape[0], ashock.shape[1], nnext])
    y_agt = ishock.copy()
    ur_rate = (1 - y_agt) * (1 - mparam.la1 * mparam.dt)
    ur_rate += y_agt * mparam.la2 * mparam.dt
    ishock0 = np.zeros_like(ishock)
    ishock1 = np.ones_like(ishock)

    def gm_fn(knormed):  # k_normalized of shape B * n_agt * T
        knormed = knormed.transpose((0, 2, 1))[:, :, :, None]
        basis = [None] * len(ptrainer.vtrainers)
        gm = [None] * len(ptrainer.vtrainers)
        for i, vtr in enumerate(ptrainer.vtrainers):
            basis[i] = vtr.gm_model.basis_fn(knormed).numpy()
            basis[i] = basis[i].transpose((0, 2, 1, 3))
            gm[i] = np.repeat(np.mean(basis[i], axis=1, keepdims=True), n_agt, axis=1)
        return basis, gm

    basic_s_now = np.stack([k_now, B_now, N_now, ishock], axis=-1)
    if ptrainer.init_ds.config["n_fm"] == 2 and "pde" not in prefix:
        knormed_sqr_mean_now = np.repeat(np.mean(knormed_now**2, axis=1, keepdims=True), n_agt, axis=1)
        knormed_sqr_mean_next = np.repeat(np.mean(knormed_next**2, axis=1, keepdims=True), n_agt, axis=1)
        fm_extra_now = knormed_sqr_mean_now-knormed_mean_now**2
        fm_extra_now = fm_extra_now[:, :, :, None]
    else:
        fm_extra_now = None
    if ptrainer.init_ds.config["n_gm"] > 0 and "pde" not in prefix:
        _, gm_now = gm_fn(knormed_now)
        gm_basis_next, gm_next = gm_fn(knormed_next)
    else:
        gm_now, gm_next = None, None
    v_now = value_fn(basic_s_now, fm_extra_now, gm_now)
    def next_value_fn(c_tmp):
        k_next_tmp = k_next + (c_now - c_tmp) * mparam.dt
        B_next_tmp = B_next + (k_next_tmp - k_next) / n_agt
        knormed_next_tmp = ptrainer.init_ds.normalize_data(k_next_tmp, key="agt_s")
        knormed_mean_next_tmp = knormed_mean_next + (knormed_next_tmp - knormed_next) / n_agt
        basic_s_next_tmp = [k_next_tmp, B_next_tmp]
        if ptrainer.init_ds.config["n_fm"] == 2 and "pde" not in prefix:
            knormed_sqr_mean_next_tmp = knormed_sqr_mean_next + (knormed_next_tmp**2 - knormed_next**2) / n_agt
            fm_extra_next_tmp = knormed_sqr_mean_next_tmp - knormed_mean_next_tmp**2
            fm_extra_next_tmp = fm_extra_next_tmp[:, :, :, None]
        else:
            fm_extra_next_tmp = None
        if ptrainer.init_ds.config["n_gm"] > 0 and "pde" not in prefix:
            gm_basis_next_tmp, _ = gm_fn(knormed_next_tmp)
            gm_next_tmp = [gm_next[i] + (gm_basis_next_tmp[i] - gm_basis_next[i]) / n_agt for i in range(len(gm_next))]
        else:
            gm_next_tmp = None
        v_tmp = np.zeros_like(v_now)
        for j in range(nnext):
            N_next_tmp = N_next - K_now * ashock[:, None, :] + K_now * ashock_next[:, None, :, j]
            basic_s_next0_tmp = np.stack(basic_s_next_tmp + [N_next_tmp, ishock0], axis=-1)
            basic_s_next1_tmp = np.stack(basic_s_next_tmp + [N_next_tmp, ishock1], axis=-1)
            v_next0 = value_fn(basic_s_next0_tmp, fm_extra_next_tmp, gm_next_tmp)
            v_next1 = value_fn(basic_s_next1_tmp, fm_extra_next_tmp, gm_next_tmp)
            v_tmp += mparam.beta*(v_next0 * ur_rate + v_next1 * (1-ur_rate)) + mparam.dt * (1-1/c_tmp)
        v_tmp /= nnext
        return v_tmp
    # Bellman expectation error
    v_next = next_value_fn(c_now)
    err_blmexpct = (v_now - v_next) / mparam.dt
    # Bellman expectation error
    v_next = np.zeros_like(v_now)
    c_max = c_now + np.minimum(k_next/mparam.dt-1e-6, 0.5)  # sampliewise cmax
    c_min = c_now - np.minimum(c_now*0.95, 0.5)  # sampliewise cmin
    dc = (c_max - c_min) / nc
    for i in range(nc+1):
        c_tmp = c_min + dc * i
        v_tmp = next_value_fn(c_tmp)
        v_next = np.maximum(v_tmp, v_next)
    err_blmopt = (v_now - v_next) / mparam.dt
    print("Bellman error of %3s: %.6f (expectation), %.6f (optimality)" % \
        (prefix.upper(), np.abs(err_blmexpct).mean(), np.abs(err_blmopt).mean()))
    return err_blmexpct, err_blmopt


def main(argv):
    del argv
    print("Validating the model from {}".format(FLAGS.model_path))
    with open(os.path.join(FLAGS.model_path, "config.json"), 'r') as f:
        config = json.load(f)
        
    # Extract seed index from FLAGS.model_path using regex pattern
    seed_match = re.search(r'(?:[fg]m)(\d)', FLAGS.model_path)
    seed_index = int(seed_match.group(1)) if seed_match else 0
    
    # Set random seed from config using the extracted index
    random_seed = config.get("random_seed", 0)
    if isinstance(random_seed, list):
        if seed_index < len(random_seed):
            random_seed = random_seed[seed_index]
        else:
            random_seed = random_seed[0]  # Fallback if index is out of bounds
    
    # # Set random seed
    # random_seed = config.get("random_seed", 0)  # Default to 0 if not specified
    # if isinstance(random_seed, list):  # Check if it's a list
    #     random_seed = random_seed[0]  # Use the first element if it's a list
    # random_seed = config.get("random_seed", [0])[0]  # Default to the first seed in the list if not specified
    print(f"Using random seed: {random_seed}")
    set_random_seed(random_seed)

    config["dataset_config"]["n_path"] = config["simul_config"]["n_path"]
    config["init_with_bchmk"] = True
    if FLAGS.n_agt > 0:
        config["n_agt"] = FLAGS.n_agt
    mparam = JFVParam(config["n_agt"], config["dt"], config["mats_path"], config["with_ashock"])

    init_ds = JFVInitDataSet(mparam, config)
    value_config = config["value_config"]
    vtrainers = [ValueTrainer(config) for i in range(value_config["num_vnet"])]
    for i, vtr in enumerate(vtrainers):
        vtr.load_model(os.path.join(FLAGS.model_path, "value{}.weights.h5".format(i)))
    ptrainer = JFVPolicyTrainer(vtrainers, init_ds, os.path.join(FLAGS.model_path, "policy.weights.h5"))

    # long simulation
    simul_config = config["simul_config"]
    n_path = simul_config["n_path"]
    T = simul_config["T"]
    state_init = init_ds.next_batch(n_path)
    shocks = JFV.simul_shocks(n_path, T, mparam, state_init)
    simul_data_bchmk = JFV.simul_k(
        n_path, T, mparam, init_ds.c_policy_bchmk, "pde",
        state_init=state_init, shocks=shocks
    )
    idx = ~np.isnan(simul_data_bchmk["k_cross"]).any(axis=(1, 2))   # exclude nan path
    def remove_nan(simul_data, idx):
        new_dict = {
            "k_cross": simul_data["k_cross"][idx],
            "csmp": simul_data["csmp"][idx],
            "N": simul_data["N"][idx],
            "B": simul_data["B"][idx]
        }
        return new_dict
    simul_data_bchmk = remove_nan(simul_data_bchmk, idx)
    simul_data_nn = JFV.simul_k(
        n_path, T, mparam, ptrainer.current_c_policy, "nn_share",
        state_init=state_init, shocks=shocks
    )
    simul_data_nn = remove_nan(simul_data_nn, idx)
    shocks = (shocks[0][idx], shocks[1][idx])

    # calculate path stats
    def path_stats(simul_data, prefix=""):
        k_mean = np.mean(simul_data["k_cross"], axis=1)
        discount = np.power(mparam.beta, np.arange(simul_data["csmp"].shape[-1]))
        util_sum = np.sum((1-1/simul_data["csmp"])*discount, axis=-1) * mparam.dt
        print(
            "%8s: total utilily: %.5f, mean of k: %.5f, std of k: %.5f, max of k: %.5f, max of K: %.5f" % (
                prefix.upper(), util_sum.mean(), simul_data["k_cross"].mean(), simul_data["k_cross"].std(),
                simul_data["k_cross"].max(), k_mean.max()
            )
        )
    # path_stats(simul_data_bchmk, "pde")
    path_stats(simul_data_nn, "nn")

    # compute Bellman expectation error
    mats = sio.loadmat(mparam.mats_path)
    if mparam.with_ashock:
        v_spline = JFV.construct_spl_SSS(mats, "V")
        value_fn_pde = lambda state, fm_extra, gm: \
            JFV.value_spl_SSS(state[..., 0], state[..., 1], state[..., -2], state[..., -1], v_spline)
    else:
        v_spline = JFV.construct_spl_DSS(mats, "V")
        value_fn_pde = lambda state, fm_extra, gm: \
            JFV.value_spl_DSS(state[..., 0], state[..., 1], state[..., -2], state[..., -1], v_spline)
    def value_fn_nn(basic_s, fm_extra, gm):
        basic_s = ptrainer.init_ds.normalize_data(basic_s, key="basic_s")
        if ptrainer.init_ds.config["n_fm"] == 0:
            basic_s = np.concatenate([basic_s[..., 0:1], basic_s[..., 2:]], axis=-1)
        if fm_extra is not None:
            n_state = basic_s.shape[-1] + fm_extra.shape[-1]
            state_fix = np.concatenate([basic_s, fm_extra], axis=-1)
        else:
            n_state = basic_s.shape[-1]
            state_fix = basic_s
        if gm is not None:
            n_state += gm[0].shape[-1]
            state = [None] * len(vtrainers)
            for i in range(len(vtrainers)):
                state[i] = np.concatenate([state_fix, gm[i]], axis=-1)
                state[i] = state[i].transpose((0, 2, 1, 3)).reshape((-1, config['n_agt'], n_state))
        else:
            state = [state_fix.transpose((0, 2, 1, 3)).reshape((-1, config['n_agt'], n_state))] * len(vtrainers)
        v = 0
        for i, vtr in enumerate(vtrainers):
            v += vtr.model(state[i]).numpy()
        v /= len(vtrainers)
        v = ptrainer.init_ds.unnormalize_data(v, key="value")
        # reshape and transpose back to path * n_agt * time
        v = v.reshape([basic_s.shape[0], basic_s.shape[2], basic_s.shape[1]])
        v = np.transpose(v, (0, 2, 1))
        return v

    err_blm_nn = bellman_err(simul_data_nn, shocks, ptrainer, value_fn_nn, "nn", seed=random_seed, nt=100)

    if config["policy_config"].get("opt_type") == "game":
        # Compute scalar Bellman error
        err_blmopt = err_blm_nn[1]
        bellman_error = np.abs(err_blmopt).mean()
        
        # Prepare row
        row = {
            "model_id": "JFV",
            "random_seed": random_seed,
            "n_fm": config["n_fm"],
            "n_gm": config["n_gm"],
            "bellman_error": bellman_error
        }

        # Define file path
        parent_dir = os.path.dirname(FLAGS.model_path)  # go up one level
        save_path = os.path.join(parent_dir, "JFV_bellman_error.csv")        
        file_exists = os.path.isfile(save_path)

        # Append to CSV
        df_row = pd.DataFrame([row])
        df_row.to_csv(save_path, mode='a', header=not file_exists, index=False)    

        # when "fm" in path_name (only do it once), compute and save PDE Bellman error
        if "fm" in FLAGS.model_path:
            err_blm_bchmk = bellman_err(simul_data_bchmk, shocks, ptrainer, value_fn_pde, "pde", seed=random_seed, nt=100)
            err_blmopt_pde = err_blm_bchmk[1]        
            bellman_error_bchmk = np.abs(err_blmopt_pde).mean()

            row_bchmk = {
                "model_id": "JFV",
                "random_seed": random_seed,
                "n_fm": "benchmark",
                "n_gm": "benchmark",
                "bellman_error": bellman_error_bchmk
            }

            # Reuse same parent_dir and save_path
            save_path_bchmk = os.path.join(parent_dir, "JFV_bellman_error.csv")
            file_exists_bchmk = os.path.isfile(save_path_bchmk)

            df_row_bchmk = pd.DataFrame([row_bchmk])
            df_row_bchmk.to_csv(save_path_bchmk, mode='a', header=not file_exists_bchmk, index=False)
            
    # save data if required
    if FLAGS.save:
        to_save = {
            "config": config,
            "k_cross_bchmk": simul_data_bchmk["k_cross"],
            "k_cross_nn": simul_data_nn["k_cross"],
            "csmp_bchmk": simul_data_bchmk["csmp"],
            "csmp_nn": simul_data_nn["csmp"],
            "N_bchmk": simul_data_bchmk["N"],
            "N_nn": simul_data_nn["N"],
            # "err_blmexpct_bchmk": err_blm_bchmk[0],
            # "err_blmopt_bchmk": err_blm_bchmk[1],
            "err_blmexpct_nn": err_blm_nn[0],
            "err_blmopt_nn": err_blm_nn[1],
            "ashock": shocks[0],
            "ishock": shocks[1],
        }
        np.savez(os.path.join(FLAGS.model_path, "paths.npz"), **to_save)

if __name__ == '__main__':
    app.run(main)
