import importlib
import pandas as pd
import numpy as np
import tensorflow as tf
import Parameters
import matplotlib.pyplot as plt
import State
import PolicyState
import Definitions
from Graphs import run_episode
from Graphs import do_random_step
import sys
import os
import torch
import gpytorch
from scipy.stats import qmc
from torch.linalg import vector_norm, norm
import pickle
# from scipy.optimize import minimize, basinhopping
# from scipy.optimize import fsolve

###############
# TODO: finalize generate random numbers by creating simulated data not randomly drawn data. Check performance.
torch.manual_seed(1512)  # For reproducibility
rng = np.random.default_rng(seed=12)

Hooks = importlib.import_module(Parameters.MODEL_NAME + ".Hooks")

import Globals
Globals.POST_PROCESSING=True

tf.get_logger().setLevel('CRITICAL')

pd.set_option('display.max_columns', None)
starting_policy = Parameters.policy(Parameters.starting_state)
Equations = importlib.import_module(Parameters.MODEL_NAME + ".Equations")

# generates training data for the GP
def generate_data(num_points=50, random=True, starting_state=Parameters.starting_state, N_simulated_episode_length=1000, N_burnin=100):

    Total_episode_length = N_simulated_episode_length + N_burnin

    if random:
        # Use the sampled indices to select rows from the tensor
        sample_state = tf.random.shuffle(starting_state)[:num_points,:]
    else:
        sample_state = starting_state
    
    ## simulate from a multiple starting states drawn from the initial distribution
    # print("Running a wide simulation path")
    N_simulated_batch_size = num_points # we simulate 50 paths
    simulation_starting_state = sample_state
    if "post_init" in dir(Hooks):
        print("Running post-init hook...")
        Hooks.post_init()
        print("Starting state after post-init:")
        print(Parameters.starting_state)
            
    state_episode = tf.tile(tf.expand_dims(simulation_starting_state, axis = 0), [Total_episode_length, 1, 1])



    # turn of alpha and beta variation
    # print("turning off alpha and beta variation")
    Globals.DISABLE_SCHOCKS = True
    Globals.PROD_SHOCK = True
    Globals.PREF_SHOCK = True

    print("Running episode to get range of variables...")

    #run episode
    #state_episode = tf.reshape(run_episode(state_episode), [N_simulated_episode_length * N_simulated_batch_size,len(Parameters.states)])

    state_episode = run_episode(state_episode)
    # delete the first N_burnin states 
    state_episode = state_episode[N_burnin:,:,:]
    # reshape
    state_episode = tf.reshape(state_episode, [N_simulated_episode_length * N_simulated_batch_size,len(Parameters.states)])

    policy_episode = Parameters.policy(state_episode)

    # print("Dimensions of state_episode: ", state_episode.shape)
    # print("Dimensions of policy_episode: ", policy_episode.shape)

    # calculate consumption ratio
    CK = Definitions.CK_y(state_episode, policy_episode)

    mean_CK = tf.reduce_mean(tf.reshape(CK, [N_simulated_episode_length, N_simulated_batch_size, 1]), axis=0)
    std_CK = tf.math.reduce_std(tf.reshape(CK, [N_simulated_episode_length, N_simulated_batch_size, 1]), axis=0)

    # reshape state_episode to get every path separately
    state_episode = tf.reshape(state_episode, [N_simulated_episode_length, N_simulated_batch_size, len(Parameters.states)])

    # get the mean of the state variables 
    mean_states = tf.reduce_mean(state_episode, axis = 0)
    std_states = tf.math.reduce_std(state_episode, axis = 0)
    # extract alpha and beta
    param_states = tf.stack([State.alpha_x(mean_states), State.beta_x(mean_states)],axis=1) 
    mean_states = tf.concat([mean_states[:,0:3], param_states], axis=1)
    std_states = tf.concat([std_states[:,0:3], param_states], axis=1)

    # create train data
    mean_x_train = torch.from_numpy(mean_states.numpy())
    std_x_train = torch.from_numpy(std_states.numpy())
    # mean_y_train = torch.from_numpy(np.squeeze(mean_CK.numpy()))
    mean_y_train = torch.from_numpy(mean_CK.numpy().flatten())
    std_y_train = torch.from_numpy(std_CK.numpy().flatten())

    return mean_x_train, std_x_train, mean_y_train, std_y_train

# takes alpha and beta and creates state for deqn
def gen_single_state(update_vals):
    length_state = tf.shape(update_vals)[0]
    single_state = tf.math.reduce_mean(Parameters.starting_state, axis = 0, keepdims=True)
    single_state = tf.tile(single_state, [length_state, 1])
    single_state = State.update(single_state, "alpha_x", update_vals[:,0])
    single_state = State.update(single_state, "beta_x", update_vals[:,1])
    # single_state = tf.expand_dims(single_state, axis = 0)

    return single_state

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# setup GP model
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def generate_random_points(num_points=50, K_max=2.593681, K_min=0.537365, a_max=1.425446, a_min=0.720068, d_max=1.130484, d_min=0.877030,alpha_lower=0.22, alpha_upper=0.38, beta_lower=0.75, beta_upper=0.95):
    # Generate a set of random points
    K = torch.rand(num_points) * (K_max - K_min) + K_min
    a = torch.rand(num_points) * (a_max - a_min) + a_min
    d = torch.rand(num_points) * (d_max - d_min) + d_min
    alpha = torch.rand(num_points) * (alpha_upper - alpha_lower) + alpha_lower
    beta = torch.rand(num_points) * (beta_upper - beta_lower) + beta_lower
    X = torch.stack([K,a,d,alpha, beta], -1)
    return X

def sim_random_points(num_points=50, sim_length=1000, N_burnin=100):
    X_temp = generate_random_points(num_points=num_points)
    X_sim = tf.convert_to_tensor(X_temp.numpy(), dtype=tf.float32)
    X_pool_mean, X_pool_std, _, _2 = generate_data(num_points=num_points, random=False, starting_state=X_sim, N_simulated_episode_length=sim_length, N_burnin=N_burnin)
    X_pool_mean = X_pool_mean.numpy()
    X_pool_mean = torch.from_numpy(X_pool_mean)
    X_pool_std = X_pool_std.numpy()
    X_pool_std = torch.from_numpy(X_pool_std)
    return X_pool_mean, X_pool_std
    
# Generate a set of random points
def generate_random_params(n_candidates,):
    alpha = rng.random(n_candidates) * (0.22 - 0.38) + 0.22
    beta = rng.random(n_candidates) * (0.75 - 0.95) + 0.75
    params = np.stack([alpha, beta], axis=1)
    return params

# generate state from random params
# takes alpha and beta and creates state for deqn
def gen_state(update_vals, starting_state=Parameters.starting_state):
    # set up shocks
    Globals.DISABLE_SCHOCKS = False
    Globals.PROD_SHOCK = False
    Globals.PREF_SHOCK = False
    starting_state = do_random_step(starting_state)
    length_state = tf.shape(update_vals)[0]
    single_state = tf.math.reduce_mean(Parameters.starting_state, axis = 0, keepdims=True)
    single_state = tf.tile(single_state, [length_state, 1])
    single_state = State.update(single_state, "alpha_x", update_vals[:,0])
    single_state = State.update(single_state, "beta_x", update_vals[:,1])
    # single_state = tf.expand_dims(single_state, axis = 0)


    return single_state

# Model class
class MyGP(gpytorch.models.ExactGP):
	def __init__(self, train_x, train_y, likelihood):
		super().__init__(train_x, train_y, likelihood)
		self.mean_module = gpytorch.means.ConstantMean()
		self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

	def forward(self, x):
		mean = self.mean_module(x)
		covar = self.covar_module(x)
		return gpytorch.distributions.MultivariateNormal(mean, covar)

# Utility function for Bayesian active learning
def bal_utility(e_gp, var_gp, rho=0.5, beta=0.5):
    """Calculate utility for Bayesian active learning."""
    utility = rho * e_gp + (beta / 2.0) * torch.log(var_gp)

    return utility

# Active Learning Step 
def active_learning_step(model, likelihood, X, F, moment, num_new_points=1, n_candidates=1000, simulation_length=1000):
    model.eval()
    likelihood.eval()
    with torch.no_grad():
        # Create a large pool of random points to choose from and take points with highest variance
        pool_X = generate_random_points(num_points=n_candidates)
        preds = model(pool_X)
        variances = preds.variance
        U_bal = bal_utility(preds.mean, variances, rho=1, beta=1000)
        new_mean_Y, indices = torch.topk(U_bal, num_new_points, dim=0)
        new_X = pool_X[indices]
        print('selected point: ', new_X)
        # simulate points of interest to get new y value
        #new_state = gen_single_state(new_X)
        new_state = tf.convert_to_tensor(new_X.numpy(), dtype=tf.float32)
        new_X_mean, new_X_std, new_y_mean, new_y_std = generate_data(num_points=tf.shape(new_state)[0], starting_state=new_state,N_simulated_episode_length=simulation_length, random=False)
    # update dataset with new points
    if moment == "mean":
        X = torch.cat([X, new_X_mean])
        F = torch.cat([F, new_y_mean])
    elif moment == "std":
        X = torch.cat([X, new_X_std])
        F = torch.cat([F, new_y_std])
    else:
        raise ValueError("Invalid moment parameter, needs to be either 'mean' or 'std'")

    return X, F

# Active Learning Step 
def active_learning_step_sim(model, likelihood, X, F, moment, starting_state = Parameters.starting_state,num_new_points=1, n_candidates=1000, simulation_length=1000):
    model.eval()
    likelihood.eval()
    with torch.no_grad():
        # Create a large pool of random points to choose from and take points with highest variance
        pool_X_mean, pool_X_std = sim_random_points(num_points=n_candidates, sim_length=simulation_length, N_burnin=100)
        if moment == "mean":
            pool_X = pool_X_mean
        elif moment == "std":
            pool_X = pool_X_std
        else:
            raise ValueError("Invalid moment parameter, needs to be either 'mean' or 'std'")
        
        preds = model(pool_X)
        variances = preds.variance
        U_bal = bal_utility(preds.mean, variances, rho=1, beta=1000)
        new_mean_Y, indices = torch.topk(U_bal, num_new_points, dim=0)
        new_X = pool_X[indices]
        
        # simulate points of interest to get new y value
        #new_state = gen_single_state(new_X)
        new_state = tf.convert_to_tensor(new_X.numpy(), dtype=tf.float32)
        new_X_mean, new_X_std, new_y_mean, new_y_std = generate_data(num_points=tf.shape(new_state)[0], starting_state=new_state,N_simulated_episode_length=simulation_length, random=False)
    # update dataset with new points
    if moment == "mean":
        print('selected point: ', new_X_mean)
        X = torch.cat([X, new_X_mean])
        F = torch.cat([F, new_y_mean])
    elif moment == "std":
        print('selected point: ', new_X_std)
        X = torch.cat([X, new_X_std])
        F = torch.cat([F, new_y_std])
    else:
        raise ValueError("Invalid moment parameter, needs to be either 'mean' or 'std'")

    return X, F

# Train the GP model, includes stopping criteria
def train(model, likelihood, optimizer, mll, X, F, selected_criterion=1, verbose=True):
    # Stopping criteria configurations
    STOP_CRITERIA = {
        "change_in_loss": {"enabled": selected_criterion == 1, "threshold": 1e-4},
        "gradient_norm": {"enabled": selected_criterion == 2, "threshold": 1e-3},
        "change_in_parameters": {"enabled": selected_criterion == 3, "threshold": 1e-4}
    }

    # Print enabled stopping criterion
    criterion_names = {
        1: "Change in Loss",
        2: "Gradient Norm",
        3: "Change in Parameters"
    }
    #print(f"Enabled Stopping Criterion: {criterion_names[selected_criterion]}")
    model.train()
    likelihood.train()
    prev_loss = None
    prev_params = [param.clone() for param in model.parameters()]

    # Track the number of optimization steps
    steps = 0

    for i in range(1000):
        optimizer.zero_grad()
        output = model(X)
        loss = -mll(output, F)
        loss.backward()
        
        if STOP_CRITERIA["gradient_norm"]["enabled"]:
            grad_norm = vector_norm(torch.stack([p.grad.norm() for p in model.parameters() if p.grad is not None]))
            if grad_norm < STOP_CRITERIA["gradient_norm"]["threshold"]:
                if verbose:
                    print(f"Stopping: Gradient norm < {STOP_CRITERIA['gradient_norm']['threshold']} at step {steps}")
                break
        
        optimizer.step()
        
        steps += 1
        
        if STOP_CRITERIA["change_in_loss"]["enabled"] and prev_loss is not None:
            if abs(prev_loss - loss.item()) < STOP_CRITERIA["change_in_loss"]["threshold"]:
                if verbose:
                    print(f"Stopping: Change in loss < {STOP_CRITERIA['change_in_loss']['threshold']} at step {steps}")
                break
        
        if STOP_CRITERIA["change_in_parameters"]["enabled"]:
            max_param_change = max(torch.max(torch.abs(prev_param - param)).item() for prev_param, param in zip(prev_params, model.parameters()))
            if max_param_change < STOP_CRITERIA["change_in_parameters"]["threshold"]:
                if verbose:
                    print(f"Stopping: Change in parameters < {STOP_CRITERIA['change_in_parameters']['threshold']} at step {steps}")
                break
        
        prev_loss = loss.item()
        prev_params = [param.clone() for param in model.parameters()]
    if verbose:
        print(f"Total optimization steps: {steps}")
        


# loo error where the model is initialized from scratch for each iteration
def compute_loo_error(X, y, criterion = 2):
    loo_error = torch.empty(0)
    for i in range(X.shape[0]):
        # data wrangling
        X_new = torch.cat((X[:i], X[i+1:]), dim=0)
        X_test = torch.unsqueeze(X[i], 0)
        y_new = torch.cat((y[:i], y[i+1:]), dim=0)
        y_test = y[i]

        #model_reduced = MyGP(X_new, y_new, likelihood)
        # Update the model with new data
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        mod = MyGP(X_new, y_new, likelihood)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, mod)
        optimizer = torch.optim.Adam(mod.parameters(), lr=0.05)

        train(mod, likelihood, optimizer, mll, X_new, y_new, selected_criterion=criterion, verbose=False)
        mod.eval()
        likelihood.eval()
        pred = mod(X_test)
        loss = (pred.mean - y_test) ** 2
        loo_error = torch.cat((loo_error, loss), dim=0)
    return torch.mean(loo_error), loo_error


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# fit GP 
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# # Initial setup
# print('Creating initial data')
# num_points_initial = 100
# Simulation_episode_length = 10000
# # do some burnin steps

# K_min = 0.537365
# K_max = 2.593681
# a_min = 0.720068
# a_max = 1.425446
# d_min = 0.877030
# d_max = 1.130484
# alpha_min = 0.22
# alpha_max = 0.38
# beta_min = 0.75
# beta_max = 0.95

# sampler = qmc.LatinHypercube(d=5, seed=rng)
# sample = sampler.random(n=num_points_initial)

# l_bounds = [K_min, a_min, d_min, alpha_min, beta_min]
# u_bounds = [K_max, a_max, d_max, alpha_max, beta_max]
# sample_scaled = tf.convert_to_tensor(qmc.scale(sample, l_bounds, u_bounds), dtype=tf.float32)

# print('sample_scaled: ', sample_scaled)

# # generate training data
# X_mean, X_std, y_mean, y_std = generate_data(num_points=num_points_initial, random=False, starting_state=sample_scaled, N_simulated_episode_length=Simulation_episode_length, N_burnin=100)
# #print('next message should be : test')

# # fitting the GP
# likelihood = gpytorch.likelihoods.GaussianLikelihood()
# model = MyGP(X_mean, y_mean, likelihood)
# mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
# train(model, likelihood, optimizer, mll, X_mean, y_mean, selected_criterion=2)

# # fit the GP for the covariance
# likelihood_std = gpytorch.likelihoods.GaussianLikelihood()
# model_std = MyGP(X_std, y_std, likelihood_std)
# mll_std = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood_std, model_std)
# optimizer_std = torch.optim.Adam(model_std.parameters(), lr=0.1)
# train(model_std, likelihood_std, optimizer_std, mll_std, X_std, y_std, selected_criterion=2)



# # add 50 points by active learning
# for i in range(50):
#     # update the training set with new points
#     X_mean, y_mean = active_learning_step_sim(model, likelihood, X_mean, y_mean, moment="mean",  num_new_points=1, n_candidates=100, simulation_length=Simulation_episode_length)
#     # retrain the model
#     model.set_train_data(inputs=X_mean, targets=y_mean, strict=False)
#     train(model, likelihood, optimizer, mll, X_mean, y_mean, selected_criterion=2, verbose=True)


# # add 50 points by active learning
# for i in range(50):
#     # update the training set with new points
#     X_std, y_std = active_learning_step_sim(model_std, likelihood_std, X_std, y_std, moment="std",  num_new_points=1, n_candidates=100, simulation_length=Simulation_episode_length)
#     # retrain the model
#     model_std.set_train_data(inputs=X_std, targets=y_std, strict=False)
#     train(model_std, likelihood_std, optimizer_std, mll_std, X_std, y_std, selected_criterion=2, verbose=True)
 
# print('final training X data: ', X_std)
# print('final training y data: ', y_std) 


# # print('test')

# # # -------------
# # # second method to get training data
# # # -------------
# # X_init = torch.from_numpy(sample_scaled.numpy())

# # # fitting the GP
# # likelihood = gpytorch.likelihoods.GaussianLikelihood()
# # model = MyGP(X_init, y_mean, likelihood)
# # mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
# # optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
# # train(model, likelihood, optimizer, mll, X_init, y_mean, selected_criterion=2)

# # # fit the GP for the covariance
# # likelihood_std = gpytorch.likelihoods.GaussianLikelihood()
# # model_std = MyGP(X_init, y_std, likelihood_std)
# # mll_std = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood_std, model_std)
# # optimizer_std = torch.optim.Adam(model_std.parameters(), lr=0.1)
# # train(model_std, likelihood_std, optimizer_std, mll_std, X_init, y_std, selected_criterion=2)



# # print('Calculating loo error...')
# # loo = compute_loo_error(X_std, y_std, criterion=2)
# # print('loo error: ', loo)

# # save model
# #save the model to a file
# output_file = Parameters.LOG_DIR +"/extended_gp_model_" + str(Simulation_episode_length) + ".pcl"
# print(output_file )
# with open(output_file, 'wb') as fd:
#     pickle.dump(model, fd, protocol=pickle.HIGHEST_PROTOCOL)
#     print("GP model written to disk")
#     print(" -------------------------------------------")
# fd.close()

# # save the covariance model
# output_file = Parameters.LOG_DIR +"/extended_gp_model_" + str(Simulation_episode_length) + "_std.pcl"
# print(output_file )
# with open(output_file, 'wb') as fd:
#     pickle.dump(model_std, fd, protocol=pickle.HIGHEST_PROTOCOL)
#     print("GP model written to disk")
#     print(" -------------------------------------------")
# fd.close()
# # -----------------------------------------------------------------------

Simulation_episode_length = 1000

# filepath to the model
path_mean_model = Parameters.LOG_DIR +"/extended_gp_model_" + str(Simulation_episode_length) + ".pcl"
# Load the model and do predictions
with open(path_mean_model, 'rb') as fd:
    model = pickle.load(fd)
    print("data loaded from disk")

# filepath to the model
path_cov_model = Parameters.LOG_DIR +"/extended_gp_model_" + str(Simulation_episode_length) + "_std.pcl"
# Load the model and do predictions
with open(path_cov_model, 'rb') as fd:
    model_cov = pickle.load(fd)
    print("data loaded from disk")

print('--------------------------------')
print('Compare NN and GP model')

model.eval()
model_cov.eval()


# Plot the comparison of NN and GP
plot_X_mean, plot_X_std, plot_y_mean, plot_y_std = generate_data(num_points=1000, starting_state=Parameters.starting_state, N_simulated_episode_length=100000) # Start with 50 points

# save simulated data
sim_dat = [plot_X_mean, plot_X_std, plot_y_mean, plot_y_std]
torch.save(sim_dat, Parameters.LOG_DIR + '/sim_test_data.pt')
plot_X_mean, plot_X_std, plot_y_mean, plot_y_std = torch.load(Parameters.LOG_DIR + '/sim_test_data.pt')

model.eval()
preds = model(plot_X_mean)
plot_y_GP = preds.mean.detach()
# Create DataFrame
df = pd.DataFrame({
    'K': plot_X_mean.numpy()[:, 0],
    'a': plot_X_mean.numpy()[:, 1],
    'd': plot_X_mean.numpy()[:, 2],
    'alpha': plot_X_mean.numpy()[:, 3],
    'beta': plot_X_mean.numpy()[:, 4],
    'CK NN': plot_y_mean.numpy(),
    'CK GP': plot_y_GP.numpy(),
    'error in %': (np.abs(plot_y_GP.numpy() - plot_y_mean.numpy()) / plot_y_mean.numpy() * 100)
})

# create summary statistics
df_summary = df.describe(include='all')

filename = Parameters.LOG_DIR + '/GP_errors_mean_' + str(Simulation_episode_length) + '.csv'
df_summary['error in %'].to_csv(filename)

print('Summary statistics of the comparison of NN and GP model mean')
print(df_summary)
print('--------------------------------')

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
color_map = plt.get_cmap('plasma')

p = ax.scatter(df['beta'],df['alpha'],df['CK NN'],c='red')
# Second scatter plot (blue dots) using CK_limit
p2 = ax.scatter(df['beta'], df['alpha'], df['CK GP'], c='blue', label='CK Limit')

ax.set_xlabel('$\\beta$')
ax.set_ylabel('$\\alpha$')
ax.set_zlabel('ck ratio std')
#fig.colorbar(p, label='ck ratio std')

 
plt.savefig(Parameters.LOG_DIR + '/paramspace_' +  'extended_ckratio_comparison' + '.png')
plt.show()
# #plt.close() 

# -------------
# plot std comparison
# -------------

model_cov.eval()
preds = model_cov(plot_X_std)
plot_y_GP = preds.mean.detach()
# Create DataFrame
df = pd.DataFrame({
    'K': plot_X_std.numpy()[:, 0],
    'a': plot_X_mean.numpy()[:, 1],
    'd': plot_X_mean.numpy()[:, 2],
    'alpha': plot_X_mean.numpy()[:, 3],
    'beta': plot_X_mean.numpy()[:, 4],
    'CK NN': plot_y_std.numpy(),
    'CK GP': plot_y_GP.numpy(),
    'error in %': (np.abs(plot_y_GP.numpy() - plot_y_std.numpy()) / plot_y_std.numpy() * 100)
})

# create summary statistics
df_summary = df.describe(include='all')

filename = Parameters.LOG_DIR + '/GP_errors_std_' + str(Simulation_episode_length) + '.csv'
df_summary['error in %'].to_csv(filename)

print('Summary statistics of the comparison of NN and GP model std')
print(df_summary)
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
color_map = plt.get_cmap('plasma')

p = ax.scatter(df['beta'],df['alpha'],df['CK NN'],c='red')
# Second scatter plot (blue dots) using CK_limit
p2 = ax.scatter(df['beta'], df['alpha'], df['CK GP'], c='blue', label='CK Limit')

ax.set_xlabel('$\\beta$')
ax.set_ylabel('$\\alpha$')
ax.set_zlabel('ck ratio std')
#fig.colorbar(p, label='ck ratio std')

 
plt.savefig(Parameters.LOG_DIR + '/paramspace_' +  'extended_ckratio_std_comparison' + '.png')
#plt.savefig(latex_dir + '/paramspace_' +  'ckratio_std_comparison' + '.png')
plt.show()
#plt.close() 
 

# -------------
# create file from analyses
# -------------
# create a file with the summary statistics
files = [10,100,1000,10000]
df = pd.DataFrame()

for file in files:
    path_file = Parameters.LOG_DIR + '/GP_errors_mean_' + str(file) + '.csv'
    df_file = pd.read_csv(path_file, index_col=0)
    colname = str(file) + '_sim_periods_mean'
    df_file.rename(columns={'error in %': colname}, inplace=True)
    if df.empty:
        df = df_file
    else:
        df = pd.merge(df, df_file, left_index=True, right_index=True)

    
for file in files:
    path_file = Parameters.LOG_DIR + '/GP_errors_std_' + str(file) + '.csv'
    df_file = pd.read_csv(path_file, index_col=0)
    colname = str(file) + '_sim_periods_std'
    df_file.rename(columns={'error in %': colname}, inplace=True)
    if df.empty:
        df = df_file
    else:
        df = pd.merge(df, df_file, left_index=True, right_index=True)

df.to_csv(Parameters.LOG_DIR + '/GP_errors_T.csv')