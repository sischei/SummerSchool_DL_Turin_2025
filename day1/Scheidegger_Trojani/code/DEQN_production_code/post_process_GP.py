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
from scipy.stats.qmc import LatinHypercube
from torch.linalg import vector_norm, norm
from scipy.optimize import minimize, basinhopping
import pickle
from scipy.optimize import fsolve

""" 
This is a file that uses Gaussian processes to match a moment in the data (mean and std of consumption capital) using alpha and beta
Notes: - For the standard deviation we need to simulate the model for longer episodes than is necessary for the mean. 
- The data generating process produces mean and std at the same time. While this makes it faster to generate data, one needs to be cautious when using the data for training and BAL
- To increase the speed one could do the active learning step simultaneously for both the mean and the std. NOT IMPLEMENTED
- The solution (target) might not be unique. Therefore, the optimization routine might not find the same solution each time.
   - Also, one has to check (by resimulating the model) whether the found parameter combination does in fact match the moments.
   - For the std I tried to match moments in the corner, as the function is flat for the most part.
- On the bottom of the file there are some plots that compare the GP predictions with the true values, might be helpful to debug.
"""


torch.manual_seed(1512)  # For reproducibility
rng = np.random.default_rng(seed=12)

Hooks = importlib.import_module(Parameters.MODEL_NAME + ".Hooks")

import Globals
Globals.POST_PROCESSING=True

tf.get_logger().setLevel('CRITICAL')

pd.set_option('display.max_columns', None)
Equations = importlib.import_module(Parameters.MODEL_NAME + ".Equations")

start_of_simulations = do_random_step(Parameters.starting_state)
starting_policy = Parameters.policy(start_of_simulations)

# takes alpha and beta and creates state for deqn
def gen_single_state(update_vals):
    length_state = tf.shape(update_vals)[0]
    single_state = tf.math.reduce_mean(start_of_simulations, axis = 0, keepdims=True)
    single_state = tf.tile(single_state, [length_state, 1])
    single_state = State.update(single_state, "alpha_x", update_vals[:,0])
    single_state = State.update(single_state, "beta_x", update_vals[:,1])
    # single_state = tf.expand_dims(single_state, axis = 0)

    return single_state

# generates training data for the GP
def generate_data(num_points=50, random=True, starting_state=start_of_simulations, N_simulated_episode_length=1000, N_burnin=100):

    Total_episode_length = N_simulated_episode_length + N_burnin
    # Size_state = tf.shape(starting_state)[0]
    # #sampled_indices = np.random.choice(Size_state, size=num_points, replace=False)
    # sampled_indices = rng.choice(Size_state, size=num_points, replace=False)
    if random:
        # Use the sampled indices to select rows from the tensor
        sample_state = tf.random.shuffle(starting_state)[:num_points,:]
    else:
        sample_state = starting_state
    #print('sampled state: ', sample_state)
    
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
    Globals.PSEUDO_SHOCK = False

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

    state_episode = tf.reshape(state_episode, [N_simulated_episode_length, N_simulated_batch_size, len(Parameters.states)])

    # get the mean of the state variables 
    mean_states = tf.reduce_mean(state_episode, axis = 0)
    # extract alpha and beta
    mean_states = tf.stack([State.alpha_x(mean_states), State.beta_x(mean_states)],axis=1)
    # create train data
    mean_x_train = torch.from_numpy(mean_states.numpy())
    # mean_y_train = torch.from_numpy(np.squeeze(mean_CK.numpy()))
    mean_y_train = torch.from_numpy(mean_CK.numpy().flatten())
    std_y_train = torch.from_numpy(std_CK.numpy().flatten())

    return mean_x_train, mean_y_train, std_y_train

# train a gaussian process on the data where C_K_train is the output and state_episode is the input
print("Training GP")

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# setup GP model
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def generate_random_points(num_points=50, a_lower=0.22, a_upper=0.38, b_lower=0.75, b_upper=0.95):
    # Generate a set of random points
    # lower and upper bounds if alpha is a pseudo state
    a_low = a_lower 
    a_up = a_upper 
    # lower and upper bounds if beta is a pseudo state
    b_low = b_lower
    b_up = b_upper
    a = torch.rand(num_points) * (a_up - a_low) + a_low
    b = torch.rand(num_points) * (b_up - b_low) + b_low
    X = torch.stack([a, b], -1)
    return X

# Model class
class MyGP(gpytorch.models.ExactGP):
	def __init__(self, train_x, train_y, likelihood):
		super().__init__(train_x, train_y, likelihood)
		self.mean_module = gpytorch.means.ConstantMean()
		self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(3/2))

	def forward(self, x):
		mean = self.mean_module(x)
		covar = self.covar_module(x)
		return gpytorch.distributions.MultivariateNormal(mean, covar)

# Utility function for Bayesian active learning
def bal_utility(e_gp, var_gp, rho=0.5, beta=0.5):
    """Calculate utility for Bayesian active learning."""
    utility = rho * e_gp + (beta / 2.0) * torch.log(var_gp)

    return utility

# Active Learning Step with Maximum Variance F refers to the mean F2 refers to the std
def active_learning_step(model, likelihood, X, F, F2,  num_new_points=1, n_candidates=1000, simulation_length=1000):
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
        new_state = gen_single_state(new_X)
        _, new_F, new_F2 = generate_data(num_points=tf.shape(new_state)[0], starting_state=new_state,N_simulated_episode_length=simulation_length, random=False)
    # update dataset with new points
    X = torch.cat([X, new_X])
    F = torch.cat([F, new_F])
    F2 = torch.cat([F2, new_F2])
    
        
    return X, F, F2

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

# Initial setup
print('Creating initial data')
# do some burnin steps
starting_state = start_of_simulations
for t in range(200): 
    starting_state = do_random_step(starting_state)

mean_x, mean_y, std_y = generate_data(num_points=50, starting_state=starting_state) # Start with 50 points
# update x data
X_cov = mean_x

#print('initial training data: ', mean_x)
# fitting the GP
likelihood = gpytorch.likelihoods.GaussianLikelihood()
likelihood_cov = gpytorch.likelihoods.GaussianLikelihood()
model = MyGP(mean_x, mean_y, likelihood)
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
train(model, likelihood, optimizer, mll, mean_x, mean_y, selected_criterion=2)

# fit the GP for the covariance
model_cov = MyGP(X_cov, std_y, likelihood_cov)
mll_cov = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood_cov, model_cov)
optimizer_cov = torch.optim.Adam(model_cov.parameters(), lr=0.05)
train(model_cov, likelihood_cov, optimizer_cov, mll_cov, X_cov, std_y, selected_criterion=2)

# loo = compute_loo_error(mean_x, mean_y)
# print('First loo error: ', loo)
 
for i in range(50):
    # update the training set with new points
    mean_x, mean_y, _ = active_learning_step(model, likelihood, mean_x, mean_y, std_y,  num_new_points=1, n_candidates=100, simulation_length=1000)
    # retrain the model
    model.set_train_data(inputs=mean_x, targets=mean_y, strict=False)
    train(model, likelihood, optimizer, mll, mean_x, mean_y, selected_criterion=2, verbose=True)
 
    
for i in range(50):
    # update the training set with new points
    X_cov, _, std_y = active_learning_step(model_cov, likelihood_cov, X_cov, mean_y, std_y, num_new_points=1, n_candidates=100, simulation_length=10000)
    # retrain the model
    model_cov.set_train_data(inputs=X_cov, targets=std_y, strict=False)
    train(model_cov, likelihood_cov, optimizer_cov, mll_cov, X_cov, std_y, selected_criterion=2, verbose=True)

print('final training X data: ', X_cov)
print('final training y data: ', std_y) 

# calculate loo error after first BAL
# loo, loo_vec = compute_loo_error(X_cov, std_y)
# print('loo error after first BAL: ', loo)
# print('loo values: ', loo_vec)

# place holder to skip the loop
loo = 0.001
# Now we want to add 5 points by active learning each time
iter = 0
while loo > 0.01 and iter < 21:

    for i in range(5):
        # update the training set with new points
        mean_x, mean_y = active_learning_step(model_cov, likelihood, mean_x, mean_y, num_new_points=1)
        # retrain the model
        model.set_train_data(inputs=mean_x, targets=mean_y, strict=False)
        train(model, likelihood, optimizer, mll, mean_x, mean_y, selected_criterion=2, verbose=False)
    
 
    loo = compute_loo_error(mean_x, mean_y)
    print('loo after iteration',iter, 'is ' , loo)

    iter += 1

# save model
#save the model to a file
output_file = Parameters.LOG_DIR +"/final_gp_model" + ".pcl"
print(output_file )
with open(output_file, 'wb') as fd:
    pickle.dump(model, fd, protocol=pickle.HIGHEST_PROTOCOL)
    print("GP model written to disk")
    print(" -------------------------------------------")
fd.close()

# save the covariance model
output_file = Parameters.LOG_DIR +"/final_gp_model_cov" + ".pcl"
print(output_file )
with open(output_file, 'wb') as fd:
    pickle.dump(model_cov, fd, protocol=pickle.HIGHEST_PROTOCOL)
    print("GP model written to disk")
    print(" -------------------------------------------")
fd.close()

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# minimize distance to target using scipy and the GP model
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# filepath to the model
path_mean_model = Parameters.LOG_DIR +"/final_gp_model" + ".pcl"
# Load the model and do predictions
with open(path_mean_model, 'rb') as fd:
    model = pickle.load(fd)
    print("data loaded from disk")

# filepath to the model
path_cov_model = Parameters.LOG_DIR +"/final_gp_model_cov" + ".pcl"
# Load the model and do predictions
with open(path_cov_model, 'rb') as fd:
    model_cov = pickle.load(fd)
    print("data loaded from disk")

print('--------------------------------')
print('Do minimization routine')

model.eval()
model_cov.eval()

# ----------------
# Simulation of test points to evaluate GP
# ----------------
# Uncomment to re-simulate test points again
print('Run long simulation path on 10 random points')
# run long simulation path on a random point (adapt parameter bounds)
true_vals = generate_random_points(num_points=10, a_lower=0.26, a_upper=0.34, b_lower=0.79, b_upper=0.91)
starting_state = gen_single_state(true_vals)
print('true_val: ', true_vals)
_, mean_y_test, cov_y_test = generate_data(num_points=10, starting_state=starting_state, N_simulated_episode_length=100000, N_burnin=200, random=False)

# save test values to file
output_file = os.environ["USE_CONFIG_FROM_RUN_DIR"] +"/test_values.csv"

# Create DataFrame
df = pd.DataFrame({
    'alpha': true_vals.numpy()[:, 0],
    'beta': true_vals.numpy()[:, 1],
    'CKratio': mean_y_test.numpy(),
})

# Save to file
df.to_csv(output_file, index=False)

print('Run long simulation path on 10 random points')
# run long simulation path on a random point (adapt parameter bounds)
true_vals = generate_random_points(num_points=3, a_lower=0.225, a_upper=0.25, b_lower=0.75, b_upper=0.8)
starting_state = gen_single_state(true_vals)
print('true_val: ', true_vals)
_, mean_y_test, cov_y_test = generate_data(num_points=3, starting_state=starting_state, N_simulated_episode_length=100000, N_burnin=200, random=False)

# save test values to file
output_file = Parameters.LOG_DIR +"/test_values_std.csv"

# Create DataFrame
df = pd.DataFrame({
    'alpha': true_vals.numpy()[:, 0],
    'beta': true_vals.numpy()[:, 1],
    'CKstd': cov_y_test.numpy(),
})

# Save to file
df.to_csv(output_file, index=False)

mean_file = Parameters.LOG_DIR + '/test_values.csv'
std_file = Parameters.LOG_DIR + '/test_values_std.csv'
# Load and format data
df = pd.read_csv(mean_file)
df_std = pd.read_csv(std_file)


mean_y_test = df['CKratio'].values
mean_x_test = torch.tensor(df[['alpha','beta']].values, dtype=torch.float32)

cov_y_test = df_std['CKstd'].values
X_cov_test = torch.tensor(df_std[['alpha','beta']].values, dtype=torch.float32)

# define objective function
def objective(x, target, model=model):
    model.eval()
    x = torch.tensor(x, dtype=torch.float32).reshape(1,2)

    # make a prediction
    preds = model(x).mean.detach().numpy()

    # Calculate the distance between the predictions and the target value 
    #distance = np.linalg.norm(preds - target)  # Euclidean distance
    distance = np.abs(preds - target)
    
    return distance


# initialize lists
parameters_chosen = []
func_vals = []

for combination in range(mean_x_test.shape[0]):
    
    # true vals
    true_vals = mean_x_test[combination:combination+1,:]
    # define target
    target_value = mean_y_test[combination]

    # prediction
    objective(true_vals, target_value)


    initial_guess = [0.3442,0.8512]
    #result = minimize(objective, initial_guess, args=(target_value,), method='Nelder-Mead', bounds=[(0.22, 0.38), (0.75, 0.95)])
    result = basinhopping(objective, initial_guess, minimizer_kwargs={"method": "Nelder-Mead", "args": (target_value,), "bounds": [(0.22, 0.38), (0.75, 0.95)], "tol": 1e-6})

    # make a prediction
    prediction_target = model(torch.tensor(result.x, dtype=torch.float32).reshape(1,2)).mean.detach().numpy()
    print('Predicition for value: ', combination)

    preds_true = model(true_vals).mean.detach().numpy()
    print('True Parameters: ', true_vals)
    print('Predicted parameters (GP): ', result.x)
    print('distance: ', result.fun)
    print('Relative error in %: ', np.abs(result.x - true_vals.numpy()) / true_vals.numpy() * 100)
    print('target value (NN output): ', target_value)
    print('Predicted value (GP) using true params: ', preds_true)
    print('Relative error in target: ', np.abs(preds_true - target_value) / target_value * 100)
    print('Predicted value (GP) using optimized params: ', prediction_target)
    print(' -------------------------------------------')
    parameters_chosen.append(result.x)
    func_vals.append(prediction_target)
    #roots_all.append(roots_x.x)

parameters_chosen = np.array(parameters_chosen)
print('parameters chosen: ', parameters_chosen.shape)

func_vals = np.array(func_vals)

state_sim = gen_single_state(parameters_chosen)
_, mean_y_sim, std_y_sim = generate_data(num_points=parameters_chosen.shape[0], starting_state=state_sim, random=False)

print('Function values: ', func_vals.shape)
mean_y_sim = mean_y_sim.numpy().reshape(10,1)
print('Mean_y_sim: ', mean_y_sim.shape)
error_mean = (np.abs(func_vals- mean_y_sim)/ mean_y_sim * 100).reshape(10,1)
print('Relative error in percent: ',error_mean)

# save results to file
result_df = pd.DataFrame({'alpha': parameters_chosen[:,0].tolist(), 'beta': parameters_chosen[:,1].tolist(), 'CKratioGP': func_vals[:,0], 'CKratioNN': mean_y_sim[:,0], 'error': error_mean[:,0]})
result_df.round(3).to_csv(Parameters.LOG_DIR + '/results_mean.csv', index=False)

# ++++++++
# search cov
print('-------------------------------------------------------')
print('Redo exercise with standard deviation')
# initialize lists
parameters_chosen = []
func_vals = []

for combination in range(X_cov_test.shape[0]):
    
    # true vals
    true_vals = X_cov_test[combination:combination+1,:]
    # define target
    target_value = cov_y_test[combination]

    # prediction
    objective(true_vals, target_value)

    # we want a point in the steep part of the function
    initial_guess = [0.24,0.77]
    result = minimize(objective, initial_guess, args=(target_value,model_cov,), method='Nelder-Mead', bounds=[(0.22, 0.28), (0.75, 0.85)])
    #result = basinhopping(objective, initial_guess, minimizer_kwargs={"method": "Nelder-Mead", "args": (target_value,), "bounds": [(0.22, 0.25), (0.75, 0.9)], "tol": 1e-6})

    # make a prediction
    prediction_target = model_cov(torch.tensor(result.x, dtype=torch.float32).reshape(1,2)).mean.detach().numpy()
    print('Predicition for value: ', combination)

    preds_true = model_cov(true_vals).mean.detach().numpy()
    print('True Parameters: ', true_vals)
    print('Predicted parameters (GP): ', result.x)
    print('distance: ', result.fun)
    print('Relative error in %: ', np.abs(result.x - true_vals.numpy()) / true_vals.numpy() * 100)
    print('target value (NN output): ', target_value)
    print('Predicted value (GP) using true params: ', preds_true)
    print('Relative error in target: ', np.abs(preds_true - target_value) / target_value * 100)
    print('Predicted value (GP) using optimized params: ', prediction_target)
    print(' -------------------------------------------')
    parameters_chosen.append(result.x)
    func_vals.append(prediction_target)
    #roots_all.append(roots_x.x)

parameters_chosen = np.array(parameters_chosen)
print('parameters chosen: ', parameters_chosen)

func_vals = np.array(func_vals)

state_sim = gen_single_state(parameters_chosen)
_, mean_y_sim, std_y_sim = generate_data(num_points=parameters_chosen.shape[0], starting_state=state_sim, N_simulated_episode_length=50000,random=False)

std_y_sim = std_y_sim.numpy().reshape(3,1)
print('std_y_sim: ', std_y_sim)
error_std = (np.abs(func_vals- std_y_sim)/ std_y_sim * 100).reshape(3,1)
print('Relative error in percent: ',error_std)

# save results to file
result_df = pd.DataFrame({'alpha': parameters_chosen[:,0], 'beta': parameters_chosen[:,1], 'CKstdGP': func_vals[:,0], 'CKstdNN': std_y_sim[:,0], 'error': error_std[:,0]})
result_df.round(3).to_csv(Parameters.LOG_DIR + '/results_std.csv', index=False)
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# plots
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# directory for latex figures
latex_dir = "../notebooks/ABC_model/latex/tables/gp"
# plot
# mean_x, mean_y, std_y = generate_data(num_points=1000, starting_state=starting_state) # Start with 50 points

# model.eval()
# preds = model(mean_x)
# # Create DataFrame
# df = pd.DataFrame({
#     'alpha': mean_x.numpy()[:, 0],
#     'beta': mean_x.numpy()[:, 1],
#     'CKratio': std_y.numpy(),
# })


# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# color_map = plt.get_cmap('plasma')

# p = ax.scatter(df['beta'],df['alpha'],df['CKratio'],c=df['CKratio'],cmap=color_map)
# ax.set_xlabel('$\\beta$')
# ax.set_ylabel('$\\alpha$')
# ax.set_zlabel('ck ratio std')
# fig.colorbar(p, label='ck ratio std')

# plt.savefig(Parameters.LOG_DIR + '/paramspace_' +  'ckratio_std' + '.png')
# plt.savefig(latex_dir + '/paramspace_' +  'ckratio_std' + '.png')
# plt.close() 

# Plot the comparison of NN and GP
plot_x, plot_y, std_y = generate_data(num_points=1000, starting_state=Parameters.starting_state) # Start with 50 points

model.eval()
preds = model(plot_x)
plot_y_GP = preds.mean.detach()
# Create DataFrame
df = pd.DataFrame({
    'alpha': plot_x.numpy()[:, 0],
    'beta': plot_x.numpy()[:, 1],
    'CK NN': plot_y.numpy(),
    'CK GP': plot_y_GP.numpy(),
    'error in %': (np.abs(plot_y_GP.numpy() - plot_y.numpy()) / plot_y.numpy() * 100)
})

# create summary statistics
df_summary = df.describe(include='all')

df_summary.round(3).to_csv(latex_dir + '/summary_mean_GP.csv', index=True)

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

 
plt.savefig(Parameters.LOG_DIR + '/paramspace_' +  'ckratio_comparison' + '.png')
plt.show()
# #plt.close() 

# -------------
# plot std comparison
# -------------
plot_x, mean_y, plot_y = generate_data(num_points=1000, starting_state=Parameters.starting_state) # Start with 50 points

model_cov.eval()
preds = model_cov(plot_x)
plot_y_GP = preds.mean.detach()
# Create DataFrame
df = pd.DataFrame({
    'alpha': plot_x.numpy()[:, 0],
    'beta': plot_x.numpy()[:, 1],
    'CK NN': plot_y.numpy(),
    'CK GP': plot_y_GP.numpy(),
    'error in %': (np.abs(plot_y_GP.numpy() - plot_y.numpy()) / plot_y.numpy() * 100)
})

# create summary statistics
df_summary = df.describe(include='all')
df_summary.round(3).to_csv(latex_dir + '/summary_std_GP.csv', index=True)

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

 
plt.savefig(Parameters.LOG_DIR + '/paramspace_' +  'ckratio_std_comparison' + '.png')
plt.savefig(latex_dir + '/paramspace_' +  'ckratio_std_comparison' + '.png')
plt.show()
plt.close() 