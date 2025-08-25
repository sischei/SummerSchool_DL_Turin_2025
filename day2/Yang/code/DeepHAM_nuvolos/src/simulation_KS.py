import numpy as np
from scipy.interpolate import RectBivariateSpline

EPSILON = 1e-3

def simul_shocks(n_sample, T, mparam, state_init=None):
    n_agt = mparam.n_agt
    ashock = np.zeros([n_sample, T])
    ishock = np.ones([n_sample, n_agt, T])
    if state_init:
        # convert productivity to 0/1 variable
        ashock[:, 0:1] = ((state_init["ashock"] - 1) / mparam.delta_a + 1) / 2
        ishock[..., 0] = state_init["ishock"]
    else:
        ashock[:, 0] = np.random.binomial(1, 0.5, n_sample)  # stationary distribution of Z is (0.5, 0.5)
        ur_rate = ashock[:, 0] * mparam.ur_g + (1 - ashock[:, 0]) * mparam.ur_b
        ur_rate = np.repeat(ur_rate[:, None], n_agt, axis=1)
        rand = np.random.uniform(0, 1, size=(n_sample, n_agt))
        ishock[rand < ur_rate, 0] = 0

    for t in range(1, T):
        if_keep = np.random.binomial(1, 0.875, n_sample)  # prob for Z to stay the same is 0.875
        ashock[:, t] = if_keep * ashock[:, t - 1] + (1 - if_keep) * (1 - ashock[:, t - 1])

    for t in range(1, T):
        a0, a1 = ashock[:, None, t - 1], ashock[:, None, t]
        y_agt = ishock[:, :, t - 1]
        ur_rate = (1 - a0) * (1 - a1) * (1 - y_agt) * mparam.p_bb_uu + (1 - a0) * (1 - a1) * y_agt * mparam.p_bb_eu
        ur_rate += (1 - a0) * a1 * (1 - y_agt) * mparam.p_bg_uu + (1 - a0) * a1 * y_agt * mparam.p_bg_eu
        ur_rate += a0 * (1 - a1) * (1 - y_agt) * mparam.p_gb_uu + a0 * (1 - a1) * y_agt * mparam.p_gb_eu
        ur_rate += a0 * a1 * (1 - y_agt) * mparam.p_gg_uu + a0 * a1 * y_agt * mparam.p_gg_eu
        rand = np.random.uniform(0, 1, size=(n_sample, n_agt))
        ishock[rand < ur_rate, t] = 0

    ashock = (ashock * 2 - 1) * mparam.delta_a + 1  # convert 0/1 variable to productivity
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
        assert n_sample == ashock.shape[0], "n_sample is inconsistent with given shocks."
        assert T == ashock.shape[1], "T is inconsistent with given shocks."
    else:
        ashock, ishock = simul_shocks(n_sample, T, mparam)
    cur_k = np.zeros((n_sample, n_agt), dtype='float32')
    if state_init is not None:
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
        k_cross[0] = cur_k
        csmp = np.zeros((T-1, n_sample, n_agt))

    if policy_type == "pde":
        for t in range(1, T):
            next_k = policy(cur_k, ashock[:, t-1:t], ishock[:, :, t-1])
            wealth = next_wealth(cur_k, ashock[:, t-1:t], ishock[:, :, t-1], mparam)
            cur_csmp = wealth - next_k
            cur_k = next_k
            if func:
                if func != 'last':
                    res[t] = func(cur_k, cur_csmp)
            else:
                k_cross[t] = cur_k
                csmp[t-1] = cur_csmp
    elif policy_type == "nn_share":
        for t in range(1, T):
            wealth = next_wealth(cur_k, ashock[:, t-1:t], ishock[:, :, t-1], mparam)
            cur_csmp = np.clip(policy(cur_k, ashock[:, t-1:t], ishock[:, :, t-1]) * wealth, EPSILON, wealth-EPSILON)
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


def next_wealth(k_cross, ashock, ishock, mparam):
    k_mean = np.mean(k_cross, axis=1, keepdims=True)
    tau = np.where(ashock < 1, mparam.tau_b, mparam.tau_g)  # labor tax rate based on ashock
    emp = np.where(ashock < 1, mparam.l_bar*mparam.er_b, mparam.l_bar*mparam.er_g)  # total labor supply based on ashock
    R = 1 - mparam.delta + ashock * mparam.alpha*(k_mean / emp)**(mparam.alpha-1)
    wage = ashock*(1-mparam.alpha)*(k_mean / emp)**(mparam.alpha)
    wealth = R * k_cross + (1-tau)*wage*mparam.l_bar*ishock + mparam.mu*wage*(1-ishock)
    return wealth


def k_policy_bspl(k_cross, ashock, ishock, splines):
    k_next = np.zeros_like(k_cross)
    k_mean = np.repeat(np.mean(k_cross, axis=1, keepdims=True), k_cross.shape[1], axis=1)

    idx = ((ashock < 1) & (ishock == 0))
    k_tmp, km_tmp = k_cross[idx], k_mean[idx]
    k_next[idx] = splines['00'](k_tmp, km_tmp, grid=False)

    idx = ((ashock < 1) & (ishock == 1))
    k_tmp, km_tmp = k_cross[idx], k_mean[idx]
    k_next[idx] = splines['01'](k_tmp, km_tmp, grid=False)

    idx = ((ashock > 1) & (ishock == 0))
    k_tmp, km_tmp = k_cross[idx], k_mean[idx]
    k_next[idx] = splines['10'](k_tmp, km_tmp, grid=False)

    idx = ((ashock > 1) & (ishock == 1))
    k_tmp, km_tmp = k_cross[idx], k_mean[idx]
    k_next[idx] = splines['11'](k_tmp, km_tmp, grid=False)

    return k_next


def construct_bspl(mats, key = 'kprime'):
    # mats is saved in Matlab through
    # "save(filename, 'kprime', 'k', 'km', 'agshock', 'idshock', 'kmts', 'kcross');"
    # new: 'V', 'c', 'kprime', 'k', 'km', 'agshock', 'idshock', 'kmts', 'kcross'
    splines = {
        '00': RectBivariateSpline(mats['k'], mats['km'], mats[key][:, :, 0, 0]),
        '01': RectBivariateSpline(mats['k'], mats['km'], mats[key][:, :, 0, 1]),
        '10': RectBivariateSpline(mats['k'], mats['km'], mats[key][:, :, 1, 0]),
        '11': RectBivariateSpline(mats['k'], mats['km'], mats[key][:, :, 1, 1]),
    }
    return splines

def value_spl(k_cross, km, ashock, ishock, splines, mparam):
    v = np.full(k_cross.shape, np.nan)
    n_total = 0
    # ashock: 1.01, 0.99, ishock: 1,0
    ashock_value = np.array([1-mparam.delta_a, 1+mparam.delta_a])
    for a_idx in range(2):
        for i_idx in range(2):
            idx = (ashock == ashock_value[a_idx])&(ishock == i_idx)
            n_total += np.sum(idx)
            v[idx] = splines[str(a_idx)+str(i_idx)](k_cross[idx], km[idx], grid=False)
    # assert n_total == v.size, "The index of B goes wrong."
    return v
