import importlib
import pandas as pd
import tensorflow as tf
import Parameters
import matplotlib.pyplot as plt
import State
import PolicyState
import Definitions
from Graphs import run_episode
from Graphs import do_random_step
import sys

"""
Example file for postprocessing 
"""

Hooks = importlib.import_module(Parameters.MODEL_NAME + ".Hooks")

import Globals
Globals.POST_PROCESSING=True

tf.config.run_functions_eagerly(True)
tf.get_logger().setLevel('CRITICAL')

pd.set_option('display.max_columns', None)

# -------------------------------------------
# adapt starting state as we had 2 periods too many, we drop the first and last rows
# -------------------------------------------
# ensures that we don't do random steps in basic models (backward compatibility)
if Parameters.MODEL_NAME == "ABC":
    start_of_simulations = do_random_step(Parameters.starting_state)
else:
    start_of_simulations = Parameters.starting_state
print("Starting state: ", start_of_simulations)

starting_policy = Parameters.policy(start_of_simulations)

Equations = importlib.import_module(Parameters.MODEL_NAME + ".Equations")

for i, s in enumerate(Parameters.states):
    for ps in Parameters.policy_states:
        plt.plot(getattr(State,s)(start_of_simulations).numpy(), getattr(PolicyState,ps)(starting_policy).numpy(), 'bs')
        # add policy lines at min / max / median
        state_where_policy_is_min = tf.math.argmin(getattr(PolicyState,ps)(starting_policy))
        state_where_policy_is_max = tf.math.argmax(getattr(PolicyState,ps)(starting_policy))

        test_states = tf.sort(getattr(State,s)(start_of_simulations) * 4 - 3*tf.math.reduce_mean(getattr(State,s)(start_of_simulations)))

        states_lower = tf.tile(tf.expand_dims(start_of_simulations[state_where_policy_is_min,:],axis=0),[starting_policy.shape[0],1])
        states_lower = tf.tensor_scatter_nd_update(states_lower,[[j,i] for j in range(start_of_simulations.shape[0])], test_states)

        plt.plot(test_states.numpy(), getattr(PolicyState,ps)(Parameters.policy(states_lower)).numpy(), 'r--')        
        
        states_upper = tf.tile(tf.expand_dims(start_of_simulations[state_where_policy_is_max,:],axis=0),[starting_policy.shape[0],1])     
        states_upper = tf.tensor_scatter_nd_update(states_upper,[[j,i] for j in range(start_of_simulations.shape[0])], test_states)

        plt.plot(test_states.numpy(), getattr(PolicyState,ps)(Parameters.policy(states_upper)).numpy(), 'r--')        

        plt.savefig(Parameters.LOG_DIR + '/' + s + '_' + ps + '.png')
        plt.close()
        
    for de in Parameters.definitions:
        defined_value = getattr(Definitions,de)(start_of_simulations, starting_policy)
        plt.plot(getattr(State,s)(start_of_simulations).numpy(), defined_value.numpy(), 'bs')
        # add policy lines at min / max / median
        state_where_def_is_min = tf.math.argmin(defined_value)
        state_where_def_is_max = tf.math.argmax(defined_value)

        test_states = tf.sort(getattr(State,s)(start_of_simulations) * 4 - 3*tf.math.reduce_mean(getattr(State,s)(start_of_simulations)))

        states_lower = tf.tile(tf.expand_dims(start_of_simulations[state_where_def_is_min,:],axis=0),[starting_policy.shape[0],1])
        states_lower = tf.tensor_scatter_nd_update(states_lower,[[j,i] for j in range(start_of_simulations.shape[0])], test_states)

        plt.plot(test_states.numpy(), getattr(Definitions,de)(states_lower, Parameters.policy(states_lower)).numpy(), 'r--')        
        
        states_upper = tf.tile(tf.expand_dims(start_of_simulations[state_where_def_is_max,:],axis=0),[starting_policy.shape[0],1])     
        states_upper = tf.tensor_scatter_nd_update(states_upper,[[j,i] for j in range(start_of_simulations.shape[0])], test_states)

        plt.plot(test_states.numpy(), getattr(Definitions,de)(states_upper, Parameters.policy(states_upper)).numpy(), 'r--')        

        plt.savefig(Parameters.LOG_DIR + '/' + s + '_' + de + '.png')
        plt.close()
        
Parameters.initialize_each_episode = False        
if not Parameters.initialize_each_episode:
    ## simulate from a single starting state which is the mean across the N_sim_batch individual trajectory starting states
    simulation_starting_state = tf.math.reduce_mean(start_of_simulations, axis = 0, keepdims=True)
    ## simulate a long range and calculate variable bounds + means from it for plotting
    print("Running a long simulation path")
    N_simulated_episode_length = 10000 #Parameters.N_simulated_episode_length or 10000
    N_simulated_batch_size = 1# Parameters.N_simulated_batch_size or 1
else:
    ## simulate from a multiple starting states drawn from the initial distribution
    print("Running a wide simulation path")
    N_simulated_episode_length = Parameters.N_simulated_episode_length or 1
    N_simulated_batch_size = Parameters.N_simulated_batch_size or 10000
    simulation_starting_state = Parameters.initialize_states(N_simulated_batch_size)
    if "post_init" in dir(Hooks):
        print("Running post-init hook...")
        Hooks.post_init()
        print("Starting state after post-init:")
        print(start_of_simulations)
        
state_episode = tf.tile(tf.expand_dims(simulation_starting_state, axis = 0), [N_simulated_episode_length, 1, 1])

print("Running episode to get range of variables...")
# adapt Globals to activate all shocks
Globals.PROD_SHOCK = True
Globals.PREF_SHOCK = True
Globals.DISABLE_SCHOCKS = True
Globals.PSEUDO_SHOCK = True

state_episode = tf.reshape(run_episode(state_episode), [N_simulated_episode_length * N_simulated_batch_size,len(Parameters.states)])

mean_states = tf.math.reduce_mean(state_episode, axis = 0, keepdims=True)
min_states = tf.math.reduce_min(state_episode, axis = 0)
max_states = tf.math.reduce_max(state_episode, axis = 0)


for i, s in enumerate(Parameters.states):
    for ps in Parameters.policy_states:
        plot_states = tf.tile(mean_states,[100,1])
        state_linspace = tf.linspace(min_states[i],max_states[i],plot_states.shape[0])
        
        plot_states = tf.tensor_scatter_nd_update(plot_states,[[j,i] for j in range(plot_states.shape[0])], state_linspace)
        plt.plot(state_linspace.numpy(), getattr(PolicyState,ps)(Parameters.policy(plot_states)).numpy(), 'b-')        
        
        plt.savefig(Parameters.LOG_DIR + '/linspace_' + s + '_' + ps + '.png')
        plt.close()
    


print("Finished plots. Calculating Euler discrepancies...")

## calculate euler deviations
policy_episode = Parameters.policy(state_episode)
euler_discrepancies = pd.DataFrame(Equations.equations(state_episode, policy_episode))
print("Euler discrepancy (absolute value) metrics")
print(euler_discrepancies.abs().describe(include='all'))

# save all relevant quantities along the trajectory 
euler_discrepancies.to_csv(Parameters.LOG_DIR + "/simulated_euler_discrepancies.csv", index=False)

state_episode_df = pd.DataFrame({s:getattr(State,s)(state_episode) for s in Parameters.states})
state_episode_df.to_csv(Parameters.LOG_DIR + "/simulated_states.csv", index=False)

policy_episode_df = pd.DataFrame({ps:getattr(PolicyState,ps)(policy_episode) for ps in Parameters.policy_states})
policy_episode_df.to_csv(Parameters.LOG_DIR + "/simulated_policies.csv", index=False)

definition_episode_df = pd.DataFrame({d:getattr(Definitions,d)(state_episode, policy_episode) for d in Parameters.definitions})
definition_episode_df.to_csv(Parameters.LOG_DIR + "/simulated_definitions.csv", index=False)

print("State metrics")
print(state_episode_df.describe(include='all'))

print("Policy metrics")
print(policy_episode_df.describe(include='all'))

print("Definition metrics")
print(definition_episode_df.describe(include='all'))

del sys.modules['Parameters']

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Calculate the steady state
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# randomly draw 1000 states from state_episode
# simulation_starting_state = tf.random.shuffle(state_episode)[:1000,:]
simulation_starting_state = tf.random.shuffle(start_of_simulations)[:1000,:]


# reset starting state
#state_episode = tf.tile(tf.expand_dims(simulation_starting_state, axis = 0), [N_simulated_episode_length, 1, 1])

#print("state episode: ",state_episode)

print("Running episode without shocks to get steady state..")


print("simulation_starting_state: ",simulation_starting_state.shape)
#state_episode = tf.reshape(run_episode(state_episode), [N_simulated_episode_length * N_simulated_batch_size,len(Parameters.states)])
no_steps = 200 # need to run for a while to get to steady state

# disable shocks
Globals.PROD_SHOCK = False
Globals.PREF_SHOCK = False
Globals.DISABLE_SCHOCKS = True
Globals.PSEUDO_SHOCK = False

for i in range(no_steps):
    # evaluate the state forward
    simulation_starting_state = do_random_step(simulation_starting_state)

    # print every 20 steps to check convergence

    if i % 20 == 0:
        print("warm-up; shock, step = ", i, simulation_starting_state)    
    

state_episode = simulation_starting_state
state_episode_df = pd.DataFrame({s:getattr(State,s)(state_episode) for s in Parameters.states})
state_episode_df.to_csv(Parameters.LOG_DIR + "/steady_state.csv", index=False)
# create steady state episode

## add parameter policy plot
policy_episode = Parameters.policy(state_episode)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
color_map = plt.get_cmap('plasma')

p = ax.scatter(getattr(State,'beta_x')(state_episode),getattr(State,'alpha_x')(state_episode),getattr(PolicyState,'Lamda_y')(policy_episode),c=getattr(PolicyState,'Lamda_y')(policy_episode),cmap=color_map)
ax.set_xlabel('$\\beta$')
ax.set_ylabel('$\\alpha$')
ax.set_zlabel('$\\lambda$')
fig.colorbar(p, label='$\\lambda$')
 
plt.savefig(Parameters.LOG_DIR + '/paramspace_' +  'Lamda' + '.png')
plt.close() 

## add parameter plot cconsumption capital ratio
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
color_map = plt.get_cmap('plasma')

p = ax.scatter(getattr(State,'beta_x')(state_episode),getattr(State,'alpha_x')(state_episode),getattr(Definitions,'CK_y')(state_episode, policy_episode),c=getattr(Definitions,'CK_y')(state_episode, policy_episode),cmap=color_map)
ax.set_xlabel('$\\beta$')
ax.set_ylabel('$\\alpha$')
ax.set_zlabel('C/K')
fig.colorbar(p, label='C/K')
 
plt.savefig(Parameters.LOG_DIR + '/paramspace_' +  'CK_y' + '.png')
plt.close() 

# plot K_x as a function of alpha_x and beta_x
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
color_map = plt.get_cmap('plasma')

ax.scatter(getattr(State,'beta_x')(state_episode),getattr(State,'alpha_x')(state_episode),getattr(State,'K_x')(state_episode),c=getattr(State,'K_x')(state_episode),cmap=color_map)
ax.set_xlabel('$\\beta$')
ax.set_ylabel('$\\alpha$')
ax.set_zlabel('$K_x$')

plt.savefig(Parameters.LOG_DIR + '/paramspace_' +  'K_x' + '.png')
plt.close()

