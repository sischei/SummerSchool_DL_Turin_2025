import json
import os
import time
import numpy as np
from absl import app
from absl import flags
from param import JFVParam
from dataset import JFVInitDataSet
from value import ValueTrainer
from policy import JFVPolicyTrainer
from simulation_JFV import simul_shocks, simul_k, EPSILON
from util import print_elapsedtime

flags.DEFINE_string("model_path", "../data/simul_results/JFV_DSS/game_nn_n50_test",
                    """The path to load json file.""",
                    short_name='m')
flags.DEFINE_integer('n_agt', -1, "Number of agents in simulation")
FLAGS = flags.FLAGS

def main(argv):
    del argv
    print("Evaluate the model from {}".format(FLAGS.model_path))
    with open(os.path.join(FLAGS.model_path, "config.json"), 'r') as f:
        config = json.load(f)
    config["dataset_config"]["n_path"] = config["simul_config"]["n_path"]
    config["init_with_bchmk"] = True
    if FLAGS.n_agt > 0:
        config["n_agt"] = FLAGS.n_agt
    mparam = JFVParam(config["n_agt"], config["dt"], config["mats_path"], config["with_ashock"])
    n_agt = config["n_agt"]
    n_sim1 = 2000 # randomly select n_sim1 samples from T * path simulated data
    n_sim2 = 500 # randomly sample n_sim2 aggregate and idio shocks per sample
    bind_thres = 0.01

    start_time = time.monotonic()
    # load the PDE and NN policy we saved
    init_ds = JFVInitDataSet(mparam, config)
    value_config = config["value_config"]
    vtrainers = [ValueTrainer(config) for i in range(value_config["num_vnet"])]
    for i, vtr in enumerate(vtrainers):
        vtr.load_model(os.path.join(FLAGS.model_path, "value{}.h5".format(i)))
    ptrainer = JFVPolicyTrainer(vtrainers, init_ds, os.path.join(FLAGS.model_path, "policy.h5"))

    # long simulation
    ### Step 1: given policy, simulate many path for n_agt for T periods
    simul_config = config["simul_config"]
    n_path = simul_config["n_path"]
    T = simul_config["T"]
    state_init = init_ds.next_batch(n_path)
    shocks = simul_shocks(n_path, T, mparam, state_init)
    # ashock: n_path * T, ishock: n_path * n_agt * T
    ashock, ishock = shocks
    simul_data_bchmk = simul_k(
        n_path, T, mparam, init_ds.c_policy_bchmk, "pde",
        state_init=state_init, shocks=shocks
    )
    k_cross_bchmk, B_bchmk, N_bchmk = simul_data_bchmk["k_cross"], simul_data_bchmk["B"], simul_data_bchmk["N"]
    # k_cross_bchmk: n_path * n_agt * T, B_bchmk: n_path * T, N_bchmk: n_path * T
    simul_data_nn = simul_k(
        n_path, T, mparam, ptrainer.current_c_policy, "nn_share",
        state_init=state_init, shocks=shocks
    )
    k_cross_nn, B_nn, N_nn = simul_data_nn["k_cross"], simul_data_nn["B"], simul_data_nn["N"]
    # k_cross_nn: n_path * n_agt * T, B_nn: n_path * T, N_nn: n_path * T

    ### Step 1.5: rule out samples that constrained in the next period (rule out if any agent gets constrained)
    # to compare the baseline and NN policy, we rule out samples that constraint in either policy
    idx_bind = np.sum(k_cross_bchmk[:, :, 1:] < bind_thres, axis=1) + \
        np.sum(k_cross_nn[:, :, 1:] < bind_thres, axis=1) # n_path * (T-1)
    idx_bind = idx_bind.reshape(-1,)
    idx_not_bind = (idx_bind == 0)  # 0/1 array
    print('Binding sample number is', idx_not_bind.shape[0]-idx_not_bind.sum(), 'out of', idx_not_bind.shape[0])
    idx_not_bind = np.nonzero(idx_not_bind)[0] # array of non-bind positions
    idx_not_bind = np.random.choice(idx_not_bind, n_sim1, replace=False)

    ### Step 2: randomly draw n_sim1 states (k,y,Z,k^,y^) from the simulated results
    k_cross_sam = k_cross_bchmk[:, :, :-1].transpose(0, 2, 1).reshape(-1, n_agt)[idx_not_bind]
    k_cross_nn_sam = k_cross_nn[:, :, :-1].transpose(0, 2, 1).reshape(-1, n_agt)[idx_not_bind]
    B_bchmk_sam = B_bchmk[:, :-1].reshape(-1, 1)[idx_not_bind]
    B_nn_sam = B_nn[:, :-1].reshape(-1, 1)[idx_not_bind]
    N_bchmk_sam = N_bchmk[:, :-1].reshape(-1, 1)[idx_not_bind]
    N_nn_sam = N_nn[:, :-1].reshape(-1, 1)[idx_not_bind]
    ashock_sam = ashock[:, :-1].reshape(-1, 1)[idx_not_bind]
    ishock_sam = ishock[:, :, :-1].transpose(0, 2, 1).reshape(-1, n_agt)[idx_not_bind]

    ### Step 3: for each state we draw, simulate aggregate and idios shocks for next period
    ### for n_sim2 times, calculate c' and r' in each simulation
    euler_err_pde = np.zeros([n_sim1, n_agt])
    euler_err_nn = np.zeros([n_sim1, n_agt])

    # simulate next step with PDE and NNpolicy, calculate static EE error
    policy_ls = ["nn_share", "pde"]
    for i_sam in range(n_sim1):
        ashock_t = ashock_sam[i_sam:i_sam+1] # 1 * 1
        ishock_t = ishock_sam[i_sam:i_sam+1] # 1 * n_agt
        shock_init = {
            "ashock": np.repeat(ashock_t, n_sim2, axis=0), # n_sim2 * 1
            "ishock": np.repeat(ishock_t, n_sim2, axis=0)  # n_sim2 * n_agt
        }
        # simulta one shock transition through T=2
        ashock_tp, ishock_tp = simul_shocks(n_sim2, 2, mparam, shock_init)
        ashock_tp, ishock_tp = ashock_tp[..., -2:-1], ishock_tp[..., -1]
        for policy_type in policy_ls:
            if policy_type == "pde":
                k_cross_t = k_cross_sam[i_sam:i_sam+1] # 1 * n_agt
                B_t = B_bchmk_sam[i_sam:i_sam+1]
                N_t = N_bchmk_sam[i_sam:i_sam+1]
            if policy_type == "nn_share":
                k_cross_t = k_cross_nn_sam[i_sam:i_sam+1] # 1 * n_agt
                B_t = B_nn_sam[i_sam:i_sam+1]
                N_t = N_nn_sam[i_sam:i_sam+1]

            # K_t, w_t, r_t, wealth_t
            K_t = B_t + N_t # 1 * 1
            wage_unit = (1 - mparam.alpha) * K_t**mparam.alpha # 1 * 1
            wage = (ishock_t * (mparam.z2-mparam.z1) + mparam.z1) * wage_unit  # map 0 to z1 and 1 to z2, 1 * n_agt
            r = mparam.alpha * K_t**(mparam.alpha-1) - mparam.delta - mparam.sigma2*K_t/N_t # 1 * 1
            wealth_t = (1 + r*mparam.dt) * k_cross_t + wage * mparam.dt    # 1 * n_agt
            if policy_type == "pde":
                c_policy = init_ds.c_policy_bchmk
                csmp_t = np.minimum(c_policy(k_cross_t, N_t, ishock_t), wealth_t/mparam.dt-EPSILON) # 1 * n_agt
            if policy_type == "nn_share":
                c_policy = ptrainer.current_c_policy
                csmp_t = c_policy(k_cross_t, N_t, ishock_t)*wealth_t/mparam.dt # 1 * n_agt

            # construct repeated sample as starting point for t+1
            k_cross_tp = wealth_t - csmp_t * mparam.dt # 1 * n_agt
            k_cross_tp = np.repeat(k_cross_tp, n_sim2, axis=0) # n_sim2 * n_agt
            csmp_t = np.repeat(csmp_t, n_sim2, axis=0) # n_sim2 * n_agt
            N_t = np.repeat(N_t, n_sim2, axis=0) # n_sim2 * 1
            B_tp = np.mean(k_cross_tp, axis=1, keepdims=True) # n_sim2 * 1
            dN_drift = mparam.dt*(mparam.alpha * K_t**(mparam.alpha-1) - mparam.delta - mparam.rhohat - \
                mparam.sigma2*(-B_t/N_t)*(K_t/N_t))*N_t
            dN_diff = K_t * ashock_tp
            N_tp = N_t + dN_drift + dN_diff # n_sim2 * 1

            # K_t+1, w_t+1, r_t+1, wealth_t+1
            K_tp = B_tp + N_tp # n_sim2 * 1
            wage_unit_tp = (1 - mparam.alpha) * K_tp**mparam.alpha
            wage_tp = (ishock_tp * (mparam.z2-mparam.z1) + mparam.z1) * wage_unit_tp  # map 0 to z1 and 1 to z2
            r_tp = mparam.alpha * K_tp**(mparam.alpha-1) - mparam.delta - mparam.sigma2*K_tp/N_tp # n_sim2 * 1
            wealth_tp = (1 + r_tp*mparam.dt) * k_cross_tp + wage_tp * mparam.dt
            if policy_type == "pde":
                csmp_tp = np.minimum(c_policy(k_cross_tp, N_tp, ishock_tp), wealth_tp/mparam.dt-EPSILON)
            if policy_type == "nn_share":
                csmp_tp = c_policy(k_cross_tp, N_tp, ishock_tp)*wealth_tp/mparam.dt # n_sim2 * n_agt

            ### Step 4: calculate static Euler equation error for each state we draw, then take average
            # JFV: gamma = 2
            euler_err_tmp = csmp_t**(-mparam.gamma) - \
                mparam.beta*(1+r_tp*mparam.dt)*csmp_tp**(-mparam.gamma) # n_sim2 * n_agt
            if policy_type == "pde":
                euler_err_pde[i_sam] = np.mean(euler_err_tmp, axis=0, keepdims=True)
            if policy_type == "nn_share":
                euler_err_nn[i_sam] = np.mean(euler_err_tmp, axis=0, keepdims=True)

    end_time = time.monotonic()
    print_elapsedtime(end_time - start_time)
    print("The Euler equation error for PDE solution to JFV problem is {}".format(
        np.nanmean(np.abs(euler_err_pde))))
    print("The Euler equation error for NN solution to JFV problem with {} agents is {}".format(
        n_agt, np.nanmean(np.abs(euler_err_nn))))
    em_idx = (ishock_sam == 1)
    print("\nThe Euler equation error for employed HH in PDE solution to JFV problem is {}".format(
        np.nanmean(np.abs(euler_err_pde[em_idx]))))
    print("The Euler equation error for unemployed HH in PDE solution to JFV problem is {}".format(
        np.nanmean(np.abs(euler_err_pde[~em_idx]))))
    print("\nThe Euler equation error for employed HH in NN solution to JFV problem with {} agents is {}".format(
        n_agt, np.nanmean(np.abs(euler_err_nn[em_idx]))))
    print("The Euler equation error for unemployed HH in NN solution to JFV problem with {} agents is {}".format(
        n_agt, np.nanmean(np.abs(euler_err_nn[~em_idx]))))

if __name__ == '__main__':
    app.run(main)
