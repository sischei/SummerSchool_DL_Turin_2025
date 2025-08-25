import json
import os
import time
import numpy as np
from absl import app
from absl import flags
from param import KSParam
from dataset import KSInitDataSet
from value import ValueTrainer
from policy import KSPolicyTrainer
from simulation_KS import simul_shocks, simul_k, EPSILON
from util import print_elapsedtime

flags.DEFINE_string("model_path", "../data/simul_results/KS/game_nn_n50_test",
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
    mparam = KSParam(config["n_agt"], config["beta"], config["mats_path"])
    n_agt = config["n_agt"]
    n_sim1 = 2000 # randomly select n_sim1 samples from T * path simulated data
    n_sim2 = 500 # randomly sample n_sim2 aggregate and idio shocks per sample
    bind_thres = 0.1

    start_time = time.monotonic()
    # load the PDE and NN policy we saved
    init_ds = KSInitDataSet(mparam, config)
    value_config = config["value_config"]
    vtrainers = [ValueTrainer(config) for i in range(value_config["num_vnet"])]
    for i, vtr in enumerate(vtrainers):
        vtr.load_model(os.path.join(FLAGS.model_path, "value{}.h5".format(i)))
    ptrainer = KSPolicyTrainer(vtrainers, init_ds, os.path.join(FLAGS.model_path, "policy.h5"))

    ### Step 1: given policy, simulate many path for n_agt for T periods
    simul_config = config["simul_config"]
    n_path = simul_config["n_path"]
    T = simul_config["T"]
    state_init = init_ds.next_batch(n_path)
    shocks = simul_shocks(n_path, T, mparam, state_init)
    ashock, ishock = shocks
    simul_data_bchmk = simul_k(
        n_path, T, mparam, init_ds.k_policy_bchmk, policy_type="pde",
        state_init=state_init, shocks=shocks
    )
    k_cross_bchmk = simul_data_bchmk["k_cross"]
    simul_data_nn = simul_k(
        n_path, T, mparam, ptrainer.current_c_policy, policy_type="nn_share",
        state_init=state_init, shocks=shocks
    )
    k_cross_nn = simul_data_nn["k_cross"]

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
    ashock_sam = ashock[:, :-1].reshape(-1, 1)[idx_not_bind]
    ishock_sam = ishock[:, :, :-1].transpose(0, 2, 1).reshape(-1, n_agt)[idx_not_bind]

    ### Step 3: for each state we draw, simulate aggregate and idios shocks for next period
    ### for n_sim2 times, calculate c' and r' in each simulation
    euler_err_pde = np.zeros([n_sim1, n_agt])
    euler_err_nn = np.zeros([n_sim1, n_agt])

    def next_wealth(k_cross, ashock, ishock, mparam):
        k_mean = np.mean(k_cross, axis=1, keepdims=True)
        tau = np.where(ashock < 1, mparam.tau_b, mparam.tau_g) # labor tax rate - depend on ashock
        emp = np.where(ashock < 1, mparam.l_bar*mparam.er_b, mparam.l_bar*mparam.er_g)
        r = 1 - mparam.delta + ashock * mparam.alpha*(k_mean / emp)**(mparam.alpha-1)
        wage = ashock*(1-mparam.alpha)*(k_mean / emp)**(mparam.alpha)
        wealth = r * k_cross + (1-tau)*wage*mparam.l_bar*ishock + mparam.mu*wage*(1-ishock)
        return wealth, r

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
                policy = init_ds.k_policy_bchmk
            if policy_type == "nn_share":
                k_cross_t = k_cross_nn_sam[i_sam:i_sam+1] # 1 * n_agt
                policy = ptrainer.current_c_policy
            wealth, _ = next_wealth(k_cross_t, ashock_t, ishock_t, mparam)
            if policy_type == "pde":
                k_cross_tp = policy(k_cross_t, ashock_t, ishock_t)
                csmp_t = wealth - k_cross_tp
            if policy_type == "nn_share":
                csmp_t = policy(k_cross_t, ashock_t, ishock_t) * wealth
                csmp_t = np.clip(csmp_t, EPSILON, wealth-EPSILON)
                k_cross_tp = wealth - csmp_t

            # construct repeated sample as starting point for t+1
            k_cross_tp = np.repeat(k_cross_tp, n_sim2, axis=0) # n_sim2 * n_agt
            wealth_tp, r_tp = next_wealth(k_cross_tp, ashock_tp, ishock_tp, mparam)
            if policy_type == "pde":
                k_cross_tpp = policy(k_cross_tp, ashock_tp, ishock_tp)
                csmp_tp = wealth_tp - k_cross_tpp
            if policy_type == "nn_share":
                csmp_tp = policy(k_cross_tp, ashock_tp, ishock_tp) * wealth_tp
                csmp_tp = np.clip(csmp_tp, EPSILON, wealth_tp-EPSILON)

            ### Step 4: calculate static Euler equation error for each state we draw, then take average
            # MMV: log utility
            euler_err_tmp = 1/csmp_t - mparam.beta*r_tp/csmp_tp
            if policy_type == "pde":
                euler_err_pde[i_sam] = np.mean(euler_err_tmp, axis=0, keepdims=True)
            if policy_type == "nn_share":
                euler_err_nn[i_sam] = np.mean(euler_err_tmp, axis=0, keepdims=True)

    end_time = time.monotonic()
    print_elapsedtime(end_time - start_time)
    print("The Euler equation error for PDE solution to KS problem is {}".format(
        np.nanmean(np.abs(euler_err_pde))))
    print("The Euler equation error for NN solution to KS problem with {} agents is {}".format(
        n_agt, np.nanmean(np.abs(euler_err_nn))))
    em_idx = (ishock_sam == 1)
    print("\nThe Euler equation error for employed HH in PDE solution to KS problem is {}".format(
        np.nanmean(np.abs(euler_err_pde[em_idx]))))
    print("The Euler equation error for unemployed HH in PDE solution to KS problem is {}".format(
        np.nanmean(np.abs(euler_err_pde[~em_idx]))))
    print("\nThe Euler equation error for employed HH in NN solution to KS problem with {} agents is {}".format(
        n_agt, np.nanmean(np.abs(euler_err_nn[em_idx]))))
    print("The Euler equation error for unemployed HH in NN solution to KS problem with {} agents is {}".format(
        n_agt, np.nanmean(np.abs(euler_err_nn[~em_idx]))))

if __name__ == '__main__':
    app.run(main)
