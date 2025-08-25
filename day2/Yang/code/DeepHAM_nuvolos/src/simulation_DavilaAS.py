import numpy as np
from scipy.interpolate import interp1d
import quantecon as qe
from scipy.interpolate import RectBivariateSpline

EPSILON = 1e-3

def simul_shocks(n_sample, T, mparam, state_init=None):
    n_agt = mparam.n_agt
    if state_init:
        # convert productivity to 0/1 variable
        ashock_init = ((state_init["ashock"] - 1) / mparam.delta_a + 1) / 2
        ashock_init = ashock_init.astype(int)
        ishock_init = state_init["ishock"].astype(int)
    else:
        ashock_init = np.random.binomial(1, 0.5, [n_sample, 1])  # stationary distribution of Z is (0.5, 0.5)
        ishock_init = 2*np.ones([n_sample, n_agt], dtype=int) # must be int to use markov chain function in quantecon
        rand = np.random.uniform(0, 1, size=(n_sample, n_agt))
        ishock_init[rand < mparam.ur + mparam.er1] = 1
        ishock_init[rand < mparam.ur] = 0

    ashock = np.zeros([n_sample, T])
    ashock[:, 0:1] = ashock_init
    for t in range(1, T):
        if_keep = np.random.binomial(1, 0.875, n_sample)  # prob for Z to stay the same is 0.875
        ashock[:, t] = if_keep * ashock[:, t-1] + (1 - if_keep) * (1 - ashock[:, t-1])

    if mparam.ashock_type == "IAS":
        mc_ishock = qe.MarkovChain(mparam.prob_trans)
        ishock = mc_ishock.simulate(ts_length=T, init=ishock_init.reshape(-1,))
        ishock = ishock.reshape((n_sample, n_agt, T))
    elif mparam.ashock_type in ["CIS", "CIS_rare"]:
        ishock = 2 * np.ones([n_sample, n_agt, T], dtype=int)
        ishock[:, :, 0] = ishock_init
        for t in range(1, T):
            a1 = ashock[:, None, t]
            y_agt = ishock[:, :, t - 1]
            # ishock realization depend on ashock
            y_agt_0 = (1 - y_agt) * (2 - y_agt) / 2
            y_agt_1 = y_agt * (2 - y_agt)
            y_agt_2 = y_agt * (y_agt - 1) / 2
            ur_rate_b = (mparam.trans_b[0][0] * y_agt_0 + 
                         mparam.trans_b[1][0] * y_agt_1 + 
                         mparam.trans_b[2][0] * y_agt_2)
            er1_rate_b = (mparam.trans_b[0][1] * y_agt_0 + 
                          mparam.trans_b[1][1] * y_agt_1 + 
                          mparam.trans_b[2][1] * y_agt_2)
            ur_rate_g = (mparam.trans_g[0][0] * y_agt_0 + 
                         mparam.trans_g[1][0] * y_agt_1 + 
                         mparam.trans_g[2][0] * y_agt_2)
            er1_rate_g = (mparam.trans_g[0][1] * y_agt_0 + 
                          mparam.trans_g[1][1] * y_agt_1 + 
                          mparam.trans_g[2][1] * y_agt_2)
            ur_rate = a1 * ur_rate_g + (1 - a1) * ur_rate_b
            er1_rate = a1 * er1_rate_g + (1 - a1) * er1_rate_b
            rand = np.random.uniform(0, 1, size=(n_sample, n_agt))
            ishock[rand < ur_rate + er1_rate, t] = 1
            ishock[rand < ur_rate, t] = 0
    else:
        raise ValueError(f"Unsupported ashock_type: {mparam.ashock_type}")

    ashock = (ashock * 2 - 1) * mparam.delta_a + 1  # convert 0/1 variable to productivity
    return ashock, ishock


def simul_k(n_sample, T, mparam, policy, policy_type, state_init=None, shocks=None):
    # policy_type: "pde" or "nn_share"
    # return k_cross [n_sample, n_agt, T]
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
    k_cross = np.zeros([n_sample, n_agt, T])
    if state_init:
        assert n_sample == state_init["k_cross"].shape[0], "n_sample is inconsistent with state_init."
        k_cross[:, :, 0] = state_init["k_cross"]
    else:
        k_cross[:, :, 0] = mparam.k_ss
    csmp = np.zeros([n_sample, n_agt, T-1])
    wealth = k_cross.copy()

    if policy_type == "pde":
        for t in range(1, T):
            wealth[:, :, t] = next_wealth(k_cross[:, :, t-1], ashock[:, t-1:t], ishock[:, :, t-1], mparam)
            k_cross_t = policy(k_cross[:, :, t-1], ashock[:, t-1:t], ishock[:, :, t-1])
            # avoid csmp being too small or even negative
            k_cross[:, :, t] = np.clip(k_cross_t, EPSILON, wealth[:, :, t]-np.minimum(1.0, 0.8*wealth[:, :, t]))
            csmp[:, :, t-1] = wealth[:, :, t] - k_cross[:, :, t]
    if policy_type == "nn_share":
        for t in range(1, T):
            wealth[:, :, t] = next_wealth(k_cross[:, :, t-1], ashock[:, t-1:t], ishock[:, :, t-1], mparam)
            csmp_t = policy(k_cross[:, :, t-1], ashock[:, t-1:t], ishock[:, :, t-1]) * wealth[:, :, t]
            csmp_t = np.clip(csmp_t, EPSILON, wealth[:, :, t]-EPSILON)
            k_cross[:, :, t] = wealth[:, :, t] - csmp_t
            csmp[:, :, t-1] = csmp_t
    simul_data = {"k_cross": k_cross, "csmp": csmp, "ashock": ashock, "ishock": ishock}
    return simul_data


def next_wealth(k_cross, ashock, ishock, mparam):
    k_mean = np.mean(k_cross, axis=1, keepdims=True)
    # tau = np.where(ashock < 1, mparam.tau_b, mparam.tau_g)  # labor tax rate based on ashock
    emp = np.where(ashock < 1, mparam.emp_b, mparam.emp_g)  # total labor supply based on ashock
    R = 1 - mparam.delta +  ashock*mparam.alpha*(k_mean / emp)**(mparam.alpha-1)
    wage = ashock*(1-mparam.alpha)*(k_mean / emp)**(mparam.alpha)
    wealth = R * k_cross + wage * (
        mparam.epsilon_0*(ishock == 0) + mparam.epsilon_1*(ishock == 1) + mparam.epsilon_2*(ishock == 2)
    )
    return wealth

def k_policy_spl(k_cross, ashock, ishock, splines, policy_config):
    opt_type = policy_config.get("opt_type", "")
    
    if opt_type == "game":
        k_next = np.zeros_like(k_cross)
        k_mean = np.repeat(np.mean(k_cross, axis=1, keepdims=True), k_cross.shape[1], axis=1)

        idx = ((ashock < 1) & (ishock == 0))
        k_next[idx] = splines['00'](k_cross[idx], k_mean[idx], grid=False)

        idx = ((ashock < 1) & (ishock == 1))
        k_next[idx] = splines['01'](k_cross[idx], k_mean[idx], grid=False)

        idx = ((ashock < 1) & (ishock == 2))
        k_next[idx] = splines['02'](k_cross[idx], k_mean[idx], grid=False)

        idx = ((ashock > 1) & (ishock == 0))
        k_next[idx] = splines['10'](k_cross[idx], k_mean[idx], grid=False)

        idx = ((ashock > 1) & (ishock == 1))
        k_next[idx] = splines['11'](k_cross[idx], k_mean[idx], grid=False)

        idx = ((ashock > 1) & (ishock == 2))
        k_next[idx] = splines['12'](k_cross[idx], k_mean[idx], grid=False)

        return k_next

    elif opt_type == "socialplanner":
        k_next = np.zeros_like(k_cross)
        for i in range(3):
            idx = (ishock == i)
            k_next[idx] = splines['y{}'.format(i)](k_cross[idx])
        return k_next

    else:
        raise ValueError(f"Unknown opt_type: {opt_type}")

def construct_spl(mats, policy_config):
    opt_type = policy_config.get("opt_type", "")

    if opt_type == "game":
        splines = {
            '00': RectBivariateSpline(mats['k'], mats['km'], mats['kprime'][:, :, 0, 0]),
            '01': RectBivariateSpline(mats['k'], mats['km'], mats['kprime'][:, :, 0, 1]),
            '02': RectBivariateSpline(mats['k'], mats['km'], mats['kprime'][:, :, 0, 2]),
            '10': RectBivariateSpline(mats['k'], mats['km'], mats['kprime'][:, :, 1, 0]),
            '11': RectBivariateSpline(mats['k'], mats['km'], mats['kprime'][:, :, 1, 1]),
            '12': RectBivariateSpline(mats['k'], mats['km'], mats['kprime'][:, :, 1, 2]),
        }
        return splines

    elif opt_type == "socialplanner":
        basic_spline = lambda i: interp1d(
            mats['k'][:, 0], mats['K1'][:, i], kind='cubic',
            fill_value=(mats['K1'][0, i], mats['K1'][-1, i]), bounds_error=False
        )
        splines = {'y{}'.format(i): basic_spline(i) for i in range(3)}
        return splines

    else:
        raise ValueError(f"Unknown opt_type: {opt_type}")
