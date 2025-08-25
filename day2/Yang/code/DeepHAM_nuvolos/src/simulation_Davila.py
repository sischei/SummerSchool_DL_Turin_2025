import numpy as np
from scipy.interpolate import interp1d
import quantecon as qe

EPSILON = 1e-3

def simul_shocks(n_sample, T, mparam, state_init=None):
    n_agt = mparam.n_agt
    # ashock is not used
    ashock = np.ones([n_sample, T])
    if state_init:
        ishock_init = state_init["ishock"].astype(int)
    else:
        ishock_init = 2*np.ones([n_sample, n_agt], dtype=int) # must be int to use markov chain function in quantecon
        rand = np.random.uniform(0, 1, size=(n_sample, n_agt))
        ishock_init[rand < mparam.ur + mparam.er1] = 1
        ishock_init[rand < mparam.ur] = 0

    mc_ishock = qe.MarkovChain(mparam.prob_trans)
    ishock = mc_ishock.simulate(ts_length=T, init=ishock_init.reshape(-1,))
    ishock = ishock.reshape((n_sample, n_agt, T))

    return ashock, ishock


def simul_k(n_sample, T, mparam, policy, policy_type, state_init=None, shocks=None, func=None):
    # input:
    #   policy_type: "pde" or "nn_share"
    #   if func is None, save all data created by simul_k
    #   if func is "last", only save the last data
    #   if func is a function of k_cross and csmp, save some statistics computed by this function
    #     e.g. if you want to save the mean value of k_cross at each moment, func = lambda k_cross, csmp: np.mean(k_cross)
    # output:
    #   if func is None, return k_cross [n_sample, n_agt, T], csmp [n_sample, n_agt, T-1], ashock [n_sample, T], ishock [n_sample, n_agt, T]
    #   if func is "last", return k_cross [n_sample, n_agt], csmp [n_sample, n_agt], ashock [n_sample], ishock [n_sample, n_agt]
    #   if func is a function of k_cross and csmp, return func(k_cross, csmp) [..., T]
    assert policy_type in ["pde", "nn_share"], "Invalid policy type"
    n_agt = mparam.n_agt
    if shocks:
        ashock, ishock = shocks
        assert n_sample == ishock.shape[0], "n_sample is inconsistent with given shocks."
        assert T == ishock.shape[2], "T is inconsistent with given shocks."
        if state_init:
            assert np.array_equal(ashock[..., 0:1], state_init["ashock"]) and \
                np.array_equal(ishock[..., 0], state_init["ishock"]), \
                "Shock inputs are inconsistent with state_init"
    else:
        ashock, ishock = simul_shocks(n_sample, T, mparam, state_init)

    cur_k = np.zeros((n_sample, n_agt), dtype='float32')
    if state_init is not None:
        assert n_sample == state_init["k_cross"].shape[0], "n_sample is inconsistent with state_init."
        cur_k[:, :] = state_init["k_cross"]
    else:
        cur_k[:, :] = mparam.k_ss
    cur_csmp = np.zeros((n_sample, n_agt), dtype='float32')
    wealth = np.zeros((n_sample, n_agt), dtype='float32')

    if func:
        if func != 'last':
            res = np.repeat(np.array(func(cur_k, cur_csmp))[None, ...], T, axis=0)
            res[1:, ...] = 0
    else:
        k_cross = np.zeros((T, n_sample, n_agt))
        csmp = np.zeros((T-1, n_sample, n_agt))

    if policy_type == "pde":
        for t in range(1, T):
            next_k = policy(cur_k, ishock[:, :, t-1])
            wealth = next_wealth(cur_k, ishock[:, :, t-1], mparam)
            # avoid csmp being too small or even negative
            next_k = np.clip(next_k, EPSILON, wealth-np.minimum(1.0, 0.8*wealth))
            cur_csmp = wealth - next_k
            cur_k = next_k
            if func:
                if func != 'last':
                    res[t] = func(cur_k, cur_csmp)
            else:
                k_cross[t] = cur_k
                csmp[t-1] = cur_csmp
    if policy_type == "nn_share":
        for t in range(1, T):
            wealth = next_wealth(cur_k, ishock[:, :, t-1], mparam)
            cur_csmp = np.clip(policy(cur_k, ishock[:, :, t-1]) * wealth, EPSILON, wealth-EPSILON)
            cur_k = wealth - cur_csmp
            if func:
                if func != 'last':
                    res[t] = func(cur_k, cur_csmp)
            else:
                k_cross[t] = cur_k
                csmp[t-1] = cur_csmp

    if func:
        if func != 'last':
            return np.transpose(res, list(range(1, len(res.shape)))+[0])
        return {"k_cross": cur_k, "csmp": cur_csmp, "ashock": ashock[:, -1], "ishock": ishock[:, :, -1]}
    else:
        return {"k_cross": np.transpose(k_cross, (1,2,0)), "csmp": np.transpose(csmp, (1,2,0)), "ashock": ashock, "ishock": ishock}


def next_wealth(k_cross, ishock, mparam):
    k_mean = np.mean(k_cross, axis=1, keepdims=True)
    # emp_g = emp_b
    R = 1 - mparam.delta +  mparam.alpha*(k_mean / mparam.emp_g)**(mparam.alpha-1)
    wage = (1-mparam.alpha)*(k_mean / mparam.emp_g)**(mparam.alpha)
    wealth = R * k_cross + wage * (
        mparam.epsilon_0*(ishock == 0) + mparam.epsilon_1*(ishock == 1) + mparam.epsilon_2*(ishock == 2)
    )
    return wealth


def k_policy_spl(k_cross, ishock, splines):
    k_next = np.zeros_like(k_cross)
    for i in range(3):
        idx = (ishock == i)
        k_next[idx] = splines['y{}'.format(i)](k_cross[idx])
    return k_next


def construct_spl(mats):
    # mats is saved in Matlab through
    # "save Aiyagari_model2 c k lambda K K1 Kss r_ss w_ss"
    # basic_spline = lambda i: interp1d(mats['k'][:, i], mats['K1'][:, i], kind='cubic', fill_value="extrapolate")
    basic_spline = lambda i: interp1d(
        mats['k'][:, 0], mats['K1'][:, i], kind='cubic',
        fill_value=(mats['K1'][0, i], mats['K1'][-1, i]), bounds_error=False
    )
    splines = {'y{}'.format(i): basic_spline(i) for i in range(3)}
    return splines
