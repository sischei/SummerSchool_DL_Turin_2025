import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.interpolate import interp1d

EPSILON = 1e-3

# simulate Poisson in discrete time
def simul_shocks(n_sample, T, mparam, state_init=None):
    n_agt = mparam.n_agt
    ashock = mparam.dt**0.5*np.random.normal(0, mparam.sigma, [n_sample, T])
    ishock = np.ones([n_sample, n_agt, T])
    if state_init:
        ishock[..., 0] = state_init["ishock"]
    else:
        ur_rate = mparam.la2/(mparam.la1 + mparam.la2)*np.ones([n_sample, n_agt])
        rand = np.random.uniform(0, 1, size=(n_sample, n_agt))
        ishock[rand < ur_rate, 0] = 0

    for t in range(1, T):
        y_agt = ishock[:, :, t - 1]
        ur_rate = (1 - y_agt) * (1 - mparam.la1 * mparam.dt) # unemployed now, (1-lambda1*dt) to remain unemployed
        ur_rate += y_agt * mparam.la2 * mparam.dt            # employed now, lambda2*dt to become unemployed
        rand = np.random.uniform(0, 1, size=(n_sample, n_agt))
        ishock[rand < ur_rate, t] = 0

    return ashock, ishock

def simul_k(n_sample, T, mparam, c_policy, policy_type, state_init=None, shocks=None):
    # policy_type: "pde" or "nn_share"
    # return k_cross [n_sample, n_agt, T]
    assert policy_type in ["pde", "nn_share"], "Invalid policy type"
    n_agt = mparam.n_agt
    if shocks:
        ashock, ishock = shocks
        assert n_sample == ashock.shape[0], "n_sample is inconsistent with given shocks."
        assert T == ashock.shape[1], "T is inconsistent with given shocks."
        if state_init:
            assert np.array_equal(ishock[..., 0], state_init["ishock"]), \
                "Shock inputs are inconsistent with state_init"
    else:
        ashock, ishock = simul_shocks(n_sample, T, mparam, state_init)
    k_cross = np.zeros([n_sample, n_agt, T])
    B, N = np.zeros([n_sample, T]), np.zeros([n_sample, T])
    csmp = np.zeros([n_sample, n_agt, T-1])
    if state_init:
        assert n_sample == state_init["k_cross"].shape[0], "n_sample is inconsistent with state_init."
        k_cross[:, :, 0] = state_init["k_cross"]
        N[:, 0:1] = state_init["N"]
        B[:, 0] = np.mean(k_cross[:, :, 0], axis=-1)
    else:
        if mparam.with_ashock:
            k_cross[:, :, 0] = mparam.B_sss
            N[:, 0] = mparam.N_sss
            B[:, 0] = mparam.B_sss
        else:
            k_cross[:, :, 0] = mparam.k_dss
            N[:, 0] = mparam.N_dss
            B[:, 0] = mparam.k_dss

    for t in range(1, T):
        K = B[:, t-1] + N[:, t-1]
        wage_unit = (1 - mparam.alpha) * K[:, None]**mparam.alpha
        wage = (ishock[:, :, t-1] * (mparam.z2-mparam.z1) + mparam.z1) * wage_unit  # map 0 to z1 and 1 to z2
        r = mparam.alpha * K[:, None]**(mparam.alpha-1) - mparam.delta - mparam.sigma2*K[:, None]/N[:, t-1:t]
        wealth = (1 + r*mparam.dt) * k_cross[:, :, t-1] + wage * mparam.dt
        if policy_type == "pde":
            # to avoid negative wealth
            csmp[:, :, t-1] = np.minimum(
                c_policy(k_cross[:, :, t-1], N[:, t-1:t], ishock[:, :, t-1]),
                wealth/mparam.dt-EPSILON)
        elif policy_type == "nn_share":
            csmp[:, :, t-1] = c_policy(k_cross[:, :, t-1], N[:, t-1:t], ishock[:, :, t-1]) * (wealth / mparam.dt)
        k_cross[:, :, t] = wealth - csmp[:, :, t-1] * mparam.dt
        B[:, t] = np.mean(k_cross[:, :, t], axis=1)
        dN_drift = mparam.dt * (mparam.alpha * K**(mparam.alpha-1) - mparam.delta - mparam.rhohat - \
            mparam.sigma2*(-B[:, t-1]/N[:, t-1])*(K/N[:, t-1]))*N[:, t-1]
        dN_diff = K * ashock[:, t-1]
        N[:, t] = N[:, t-1] + dN_drift + dN_diff

    # print(B.max(), B.min(), N.max(), N.min(), csmp.min(), csmp.max())
    # if k_cross.min() < 0 or N.min() < 0:
    #     print(k_cross.min(), N.min())
    simul_data = {"k_cross": k_cross, "csmp": csmp, "B": B, "N": N, "ishock": ishock}
    return simul_data


def c_policy_spl_DSS(k_cross, N, ishock, splines):  # pylint: disable=W0613
    c = np.zeros_like(k_cross)
    idx = (ishock == 0)
    c[idx] = splines["y0"](k_cross[idx])
    idx = (ishock == 1)
    c[idx] = splines["y1"](k_cross[idx])
    return c


def construct_spl_DSS(mats, key):
    # mats is saved in Matlab through
    # save 'ss_for_JQ.mat' aa zz V c g_ss -mat (here z is idiosyncratic income level)
    splines = {
        'y0': interp1d(mats['aa'][:, 0], mats[key][:, 0], kind='cubic', fill_value="extrapolate"),
        'y1': interp1d(mats['aa'][:, 1], mats[key][:, 1], kind='cubic', fill_value="extrapolate"),
    }
    return splines


def c_policy_spl_SSS(k_cross, N, ishock, splines):
    # this part is simplified than the notebook, considering that B is always <=Bmax (but possibly <Bmin) in simulation
    # c = np.zeros_like(k_cross)
    c = np.full(k_cross.shape, np.nan)
    B = np.mean(k_cross, axis=-1, keepdims=True)
    dB = 2/3
    # BposD = np.floor((B-mparam.Bmin)/mparam.dB)
    # BposU = np.ceil((B_exp-mparam.Bmin)/mparam.dB)
    # wB = (B_exp - mparam.Bmin - BposD*mparam.dB)/mparam.dB
    BposD = np.maximum(np.floor((B-0.7) / dB), 0)
    wB = (B - 0.7 - BposD*dB) / dB
    BposD = np.repeat(BposD, c.shape[-1], axis=1)
    wB = np.repeat(wB, c.shape[-1], axis=1)
    N = np.repeat(N, c.shape[-1], axis=1)

    n_total = 0
    for i_idx in range(2):
        for BD_idx in range(4):
            idx = (ishock == i_idx)&(BposD == BD_idx)
            n_total += np.sum(idx)
            cD = splines["y"+str(i_idx)+"_B"+str(BD_idx)](k_cross[idx], N[idx], grid=False)
            cU = splines["y"+str(i_idx)+"_B"+str(min(BD_idx+1, 3))](k_cross[idx], N[idx], grid=False)
            c[idx] = (1-wB[idx])*cD +  wB[idx]*cU
    # assert n_total == c.size, "The index of B goes wrong."
    return c


def construct_spl_SSS(mats, key):
    # mats is saved in Matlab
    splines = {
        'y0_B0': RectBivariateSpline(mats['a'][:, 0, 0, 0], mats['N'][0, 0, 0, :], mats[key][:, 0, 0, :]),
        'y1_B0': RectBivariateSpline(mats['a'][:, 0, 0, 0], mats['N'][0, 0, 0, :], mats[key][:, 1, 0, :]),
        'y0_B1': RectBivariateSpline(mats['a'][:, 0, 0, 0], mats['N'][0, 0, 0, :], mats[key][:, 0, 1, :]),
        'y1_B1': RectBivariateSpline(mats['a'][:, 0, 0, 0], mats['N'][0, 0, 0, :], mats[key][:, 1, 1, :]),
        'y0_B2': RectBivariateSpline(mats['a'][:, 0, 0, 0], mats['N'][0, 0, 0, :], mats[key][:, 0, 2, :]),
        'y1_B2': RectBivariateSpline(mats['a'][:, 0, 0, 0], mats['N'][0, 0, 0, :], mats[key][:, 1, 2, :]),
        'y0_B3': RectBivariateSpline(mats['a'][:, 0, 0, 0], mats['N'][0, 0, 0, :], mats[key][:, 0, 3, :]),
        'y1_B3': RectBivariateSpline(mats['a'][:, 0, 0, 0], mats['N'][0, 0, 0, :], mats[key][:, 1, 3, :]),
    }
    return splines


def value_spl_DSS(k_cross, B, N, ishock, splines):  # pylint: disable=W0613
    v = np.zeros_like(k_cross)
    idx = (ishock == 0)
    v[idx] = splines["y0"](k_cross[idx])
    idx = (ishock == 1)
    v[idx] = splines["y1"](k_cross[idx])
    return v


def value_spl_SSS(k_cross, B, N, ishock, splines):
    v = np.full(k_cross.shape, np.nan)
    dB = 2/3
    BposD = np.maximum(np.floor((B-0.7) / dB), 0)
    wB = (B - 0.7 - BposD*dB) / dB

    n_total = 0
    for i_idx in range(2):
        for BD_idx in range(4):
            idx = (ishock == i_idx)&(BposD == BD_idx)
            n_total += np.sum(idx)
            vD = splines["y"+str(i_idx)+"_B"+str(BD_idx)](k_cross[idx], N[idx], grid=False)
            vU = splines["y"+str(i_idx)+"_B"+str(min(BD_idx+1, 3))](k_cross[idx], N[idx], grid=False)
            v[idx] = (1-wB[idx])*vD +  wB[idx]*vU
    # assert n_total == v.size, "The index of B goes wrong."
    return v
