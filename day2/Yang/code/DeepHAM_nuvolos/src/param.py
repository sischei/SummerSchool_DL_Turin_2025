import numpy as np


class KSParam():
    def __init__(self, n_agt, beta, mats_path):
        self.n_agt = n_agt  # number of finite agents
        self.beta = beta  # discount factor
        self.mats_path = mats_path  # matrix from Matlab policy
        self.gamma = 1.0  # utility-function parameter
        self.alpha = 0.36  # share of capital in the production function
        self.delta = 0.025  # depreciation rate
        self.delta_a = 0.01  # (1-delta_a) is the productivity level in a bad state,
        # and (1+delta_a) is the productivity level in a good state
        self.mu = 0.15  # unemployment benefits as a share of wage
        self.l_bar = 1.0 / 0.9  # time endowment normalizes labor supply to 1 in a bad state

        self.epsilon_u = 0  # idiosyncratic shock if the agent is unemployed
        self.epsilon_e = 1  # idiosyncratic shock if the agent is employed

        self.ur_b = 0.1  # unemployment rate in a bad aggregate state
        self.er_b = (1 - self.ur_b)  # employment rate in a bad aggregate state
        self.ur_g = 0.04  # unemployment rate in a good aggregate state
        self.er_g = (1 - self.ur_g)  # employment rate in a good aggregate state

        # labor tax rate in bad and good aggregate states
        self.tau_b = self.mu * self.ur_b / (self.l_bar * self.er_b)
        self.tau_g = self.mu * self.ur_g / (self.l_bar * self.er_g)

        self.k_ss = ((1 / self.beta - (1 - self.delta)) / self.alpha) ** (1 / (self.alpha - 1))
        # steady-state capital in a deterministic model with employment rate of 0.9
        # (i.e., l_bar*L=1, where L is aggregate labor in the paper)

        self.prob_trans = np.array(
            [
                [0.525, 0.35, 0.03125, 0.09375],
                [0.038889, 0.836111, 0.002083, 0.122917],
                [0.09375, 0.03125, 0.291667, 0.583333],
                [0.009115, 0.115885, 0.024306, 0.850694]
            ]
        )

        self.prob_ag = np.zeros([2, 2])
        self.prob_ag[0, 0] = self.prob_trans[0, 0] + self.prob_trans[0, 1]
        self.prob_ag[1, 1] = self.prob_trans[3, 2] + self.prob_trans[3, 3]
        self.prob_ag[0, 1] = 1 - self.prob_ag[0, 0]
        self.prob_ag[1, 0] = 1 - self.prob_ag[1, 1]

        self.p_bb_uu = self.prob_trans[0, 0] / self.prob_ag[0, 0]
        self.p_bb_ue = 1 - self.p_bb_uu
        self.p_bb_ee = self.prob_trans[1, 1] / self.prob_ag[0, 0]
        self.p_bb_eu = 1 - self.p_bb_ee
        self.p_bg_uu = self.prob_trans[0, 2] / self.prob_ag[0, 1]
        self.p_bg_ue = 1 - self.p_bg_uu
        self.p_bg_ee = self.prob_trans[1, 3] / self.prob_ag[0, 1]
        self.p_bg_eu = 1 - self.p_bg_ee
        self.p_gb_uu = self.prob_trans[2, 0] / self.prob_ag[1, 0]
        self.p_gb_ue = 1 - self.p_gb_uu
        self.p_gb_ee = self.prob_trans[3, 1] / self.prob_ag[1, 0]
        self.p_gb_eu = 1 - self.p_gb_ee
        self.p_gg_uu = self.prob_trans[2, 2] / self.prob_ag[1, 1]
        self.p_gg_ue = 1 - self.p_gg_uu
        self.p_gg_ee = self.prob_trans[3, 3] / self.prob_ag[1, 1]
        self.p_gg_eu = 1 - self.p_gg_ee


class DavilaParam():
    def __init__(self, n_agt, beta, mats_path, ashock_type):
        self.n_agt = n_agt  # number of finite agents
        self.beta = beta  # discount factor
        self.mats_path = mats_path  # matrix from Matlab policy
        self.ashock_type = ashock_type  # None, or IAS, or CIS
        self.gamma = 2.0  # utility-function parameter
        self.alpha = 0.36  # share of capital in the production function
        self.delta = 0.08  # annual depreciation rate
        self.delta_a = 0.02  # (1-delta_a) is the productivity level in a bad state,
        # and (1+delta_a) is the productivity level in a good state
        self.k_ss = ((1 / self.beta - (1 - self.delta)) / self.alpha) ** (1 / (self.alpha - 1))
        # steady-state capital in a complete market model
        self.amin = 0.0  # borrowing constraint

        self.epsilon_0 = 1.0  # idiosyncratic state 0
        self.epsilon_1 = 5.29  # idiosyncratic state 1
        self.epsilon_2 = 46.55  # idiosyncratic state 2

        self.ur = 0.49833222  # unemployment rate
        self.er1 = 0.44296197
        self.er2 = 1 - self.ur - self.er1

        if ashock_type == "CIS":
            self.trans_g = np.array([
                [0.98, 0.02, 0.0],
                [0.009, 0.980, 0.011],
                [0.0, 0.083, 0.917]
            ])
            self.trans_b = np.array([
                [0.6512248557478917, 0.34877514425210826, 0.0],
                [0.978, 0.011, 0.011],
                [0.0, 0.083, 0.917]
            ])
            self.ur_g = 0.28435478  # unemployment rate in good aggregate state
            self.er1_g = 0.63189951
            self.er2_g = 1 - self.ur_g - self.er1_g
            self.ur_b = 0.71230967  # unemployment rate in good aggregate state
            self.er1_b = 0.25402444
            self.er2_b = 1 - self.ur_g - self.er1_g
            self.emp_g = self.epsilon_0*self.ur_g + self.epsilon_1*self.er1_g + self.epsilon_2*self.er2_g
            self.emp_b = self.epsilon_0*self.ur_b + self.epsilon_1*self.er1_b + self.epsilon_2*self.er2_b
        elif ashock_type == "CIS_rare":
            self.trans_g = np.array([
                [0.98, 0.02, 0.0],
                [0.01, 0.98, 0.01],
                [0.0, 0.2, 0.8]
            ])
            self.trans_b = np.array([
                [0.6512248557478917, 0.34877514425210826, 0.0],
                [0.98, 0.01, 0.01],
                [0.0, 0.2, 0.8]
            ])
            self.ur_g = 0.32258065  # unemployment rate in good aggregate state
            self.er1_g = 0.64516129
            self.er2_g = 1 - self.ur_g - self.er1_g
            self.ur_b = 0.72795341  # unemployment rate in good aggregate state
            self.er1_b = 0.25909199
            self.er2_b = 1 - self.ur_g - self.er1_g
            self.emp_g = self.epsilon_0*self.ur_g + self.epsilon_1*self.er1_g + self.epsilon_2*self.er2_g
            self.emp_b = self.epsilon_0*self.ur_b + self.epsilon_1*self.er1_b + self.epsilon_2*self.er2_b
        else:
            self.prob_trans = np.array([
                [0.992, 0.008, 0.0],
                [0.009, 0.980, 0.011],
                [0.0, 0.083, 0.917]
            ])
            self.emp_g = self.epsilon_0*self.ur + self.epsilon_1*self.er1 + self.epsilon_2*self.er2
            self.emp_b = self.emp_g


class JFVParam():
    def __init__(self, n_agt, dt, mats_path, with_ashock):
        self.n_agt = n_agt  # number of finite agents
        self.dt = dt
        self.mats_path = mats_path
        self.with_ashock = with_ashock
        self.rho = 0.05  # discount rate
        self.rhohat = 0.04971  # discount rate for experts
        self.gamma = 2.0  # utility-function parameter
        self.alpha = 0.35  # share of capital in the production function
        self.delta = 0.1  # depreciation rate
        self.la1 = 0.986  #transition probability from low to high
        self.la2 = 0.052  #transition probability from high to low
        self.z1 = 0.72 # low type labor productivity
        self.z2 = 1 + self.la2/self.la1 * (1-self.z1)# high type labor productivity
        if with_ashock: # SSS
            self.sigma = 0.0140 # sigma for aggregate capital quality shock
            self.sigma2 = self.sigma**2 # sigma^2
        else:   # DSS
            self.sigma = 0
            self.sigma2 = 0
        self.beta = np.exp(-self.rho * self.dt)
        self.k_dss = 1.8718155468494229
        self.N_dss = 1.8214550560258203
        # self.B_sss = 1.8531388366299562
        # self.N_sss = 1.8267371165081967
        self.B_sss = 1.9903560465449313
        self.N_sss = 1.6837810785365737

        self.amin = 0.0  # borrowing constraint
        self.amax = 20.0 # max value of individual savings
        self.Bmin = 0.7 # relevant range for aggregate savings
        self.Bmax = 2.7
        self.Nmin = 1.2 # relevant range for aggregate equity
        self.Nmax = 3.2

        self.nval_a = 501 # number of points in amin-to-amax range (individual savings)
        self.nval_z = 2 # number of options for z (the idiosincratic shock)
        self.nval_B = 4 # number of points in Bmin-to-Bmax range (aggregate savings), on the coarse grid for HJB
        self.nval_N = 51 # number of points in Nmin-to-Nmax range (aggregate equity), on the coarse grid for HJB

        self.nval_BB = 101 # finer grid, used for training the NN, for determining visited range and for the convergence
        self.nval_NN = 101 # finer grid, used for training the NN, for determining visited range and for the convergence

        self.da = (self.amax-self.amin)/(self.nval_a-1) # size of a jump
        self.dB = (self.Bmax-self.Bmin)/(self.nval_B-1) # size of B jump on the coarse grid
        self.dN = (self.Nmax-self.Nmin)/(self.nval_N-1) # size of N jump on the coarse grid
        self.dBB = (self.Bmax-self.Bmin)/(self.nval_BB-1)# size of B jump on the fine grid
        self.dNN = (self.Nmax-self.Nmin)/(self.nval_NN-1)# size of B jump on the fine grid

        # prices in the DSS
        self.r_dss = self.rhohat # DSS interest rate
        self.K_dss = ((self.rhohat + self.delta)/self.alpha)**(1.0/(self.alpha-1.0)) # DSS capital
        self.w_dss = (1.0-self.alpha)*(self.K_dss**self.alpha) # DSS wage
