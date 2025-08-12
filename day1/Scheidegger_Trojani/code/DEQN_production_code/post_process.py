import importlib
import pandas as pd
import tensorflow as tf
import Parameters
import matplotlib.pyplot as plt
import State
import PolicyState
import Definitions
from Graphs import run_episode
import sys

"""
Example file for postprocessing 
"""

Hooks = importlib.import_module(Parameters.MODEL_NAME + ".Hooks")

import Globals
Globals.POST_PROCESSING=True

tf.get_logger().setLevel('CRITICAL')

pd.set_option('display.max_columns', None)
starting_policy = Parameters.policy(Parameters.starting_state)

Equations = importlib.import_module(Parameters.MODEL_NAME + ".Equations")

#for i, s in enumerate(Parameters.states):
    #for ps in Parameters.policy_states:
        #plt.plot(getattr(State,s)(Parameters.starting_state).numpy(), getattr(PolicyState,ps)(starting_policy).numpy(), 'bs')
        ## add policy lines at min / max / median
        #state_where_policy_is_min = tf.math.argmin(getattr(PolicyState,ps)(starting_policy))
        #state_where_policy_is_max = tf.math.argmax(getattr(PolicyState,ps)(starting_policy))

        #test_states = tf.sort(getattr(State,s)(Parameters.starting_state) * 4 - 3*tf.math.reduce_mean(getattr(State,s)(Parameters.starting_state)))

        #states_lower = tf.tile(tf.expand_dims(Parameters.starting_state[state_where_policy_is_min,:],axis=0),[starting_policy.shape[0],1])
        #states_lower = tf.tensor_scatter_nd_update(states_lower,[[j,i] for j in range(Parameters.starting_state.shape[0])], test_states)

        #plt.plot(test_states.numpy(), getattr(PolicyState,ps)(Parameters.policy(states_lower)).numpy(), 'r--')        
        
        #states_upper = tf.tile(tf.expand_dims(Parameters.starting_state[state_where_policy_is_max,:],axis=0),[starting_policy.shape[0],1])     
        #states_upper = tf.tensor_scatter_nd_update(states_upper,[[j,i] for j in range(Parameters.starting_state.shape[0])], test_states)

        #plt.plot(test_states.numpy(), getattr(PolicyState,ps)(Parameters.policy(states_upper)).numpy(), 'r--')        

        #plt.savefig(Parameters.LOG_DIR + '/' + s + '_' + ps + '.png')
        #plt.close()
        
    #for de in Parameters.definitions:
        #defined_value = getattr(Definitions,de)(Parameters.starting_state, starting_policy)
        #plt.plot(getattr(State,s)(Parameters.starting_state).numpy(), defined_value.numpy(), 'bs')
        ## add policy lines at min / max / median
        #state_where_def_is_min = tf.math.argmin(defined_value)
        #state_where_def_is_max = tf.math.argmax(defined_value)

        #test_states = tf.sort(getattr(State,s)(Parameters.starting_state) * 4 - 3*tf.math.reduce_mean(getattr(State,s)(Parameters.starting_state)))

        #states_lower = tf.tile(tf.expand_dims(Parameters.starting_state[state_where_def_is_min,:],axis=0),[starting_policy.shape[0],1])
        #states_lower = tf.tensor_scatter_nd_update(states_lower,[[j,i] for j in range(Parameters.starting_state.shape[0])], test_states)

        #plt.plot(test_states.numpy(), getattr(Definitions,de)(states_lower, Parameters.policy(states_lower)).numpy(), 'r--')        
        
        #states_upper = tf.tile(tf.expand_dims(Parameters.starting_state[state_where_def_is_max,:],axis=0),[starting_policy.shape[0],1])     
        #states_upper = tf.tensor_scatter_nd_update(states_upper,[[j,i] for j in range(Parameters.starting_state.shape[0])], test_states)

        #plt.plot(test_states.numpy(), getattr(Definitions,de)(states_upper, Parameters.policy(states_upper)).numpy(), 'r--')        

        #plt.savefig(Parameters.LOG_DIR + '/' + s + '_' + de + '.png')
        #plt.close()
        
Parameters.initialize_each_episode = False        
if not Parameters.initialize_each_episode:
    ## simulate from a single starting state which is the mean across the N_sim_batch individual trajectory starting states
    simulation_starting_state = tf.math.reduce_mean(Parameters.starting_state, axis = 0, keepdims=True)
    ## simulate a long range and calculate variable bounds + means from it for plotting
    print("Running a long simulation path")
    N_simulated_episode_length = 10000 #Parameters.N_simulated_episode_length or 10000
    N_simulated_batch_size = 1# Parameters.N_simulated_batch_size or 1
#else:
    ### simulate from a multiple starting states drawn from the initial distribution
    #print("Running a wide simulation path")
    #N_simulated_episode_length = Parameters.N_simulated_episode_length or 1
    #N_simulated_batch_size = Parameters.N_simulated_batch_size or 10000
    #simulation_starting_state = Parameters.initialize_states(N_simulated_batch_size)
    #if "post_init" in dir(Hooks):
        #print("Running post-init hook...")
        #Hooks.post_init()
        #print("Starting state after post-init:")
        #print(Parameters.starting_state)
        
state_episode = tf.tile(tf.expand_dims(simulation_starting_state, axis = 0), [N_simulated_episode_length, 1, 1])

print("Running episode to get range of variables...")
# we are not going to re-run this graph, so let's not trace it
tf.config.experimental_run_functions_eagerly(True)

state_episode = tf.reshape(run_episode(state_episode), [N_simulated_episode_length * N_simulated_batch_size,len(Parameters.states)])

mean_states = tf.math.reduce_mean(state_episode, axis = 0, keepdims=True)
min_states = tf.math.reduce_min(state_episode, axis = 0)
max_states = tf.math.reduce_max(state_episode, axis = 0)

"""
for i, s in enumerate(Parameters.states):
    for ps in Parameters.policy_states:
        plot_states = tf.tile(mean_states,[100,1])
        state_linspace = tf.linspace(min_states[i],max_states[i],plot_states.shape[0])
        
        plot_states = tf.tensor_scatter_nd_update(plot_states,[[j,i] for j in range(plot_states.shape[0])], state_linspace)
        plt.plot(state_linspace.numpy(), getattr(PolicyState,ps)(Parameters.policy(plot_states)).numpy(), 'b-')        
        
        plt.savefig(Parameters.LOG_DIR + '/linspace_' + s + '_' + ps + '.png')
        plt.close()
    
"""
    
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
