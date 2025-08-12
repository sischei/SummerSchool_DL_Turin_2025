import itertools
import tensorflow as tf
import State
import PolicyState
import Definitions
import Globals

# shocks
shocks_tfp = [.95, 1.05]
shocks_depr = [.5, .9]

probs_tfp = [.5, .5]
probs_depr = [.5, .5]

shock_values = tf.constant(list(itertools.product(shocks_tfp, shocks_depr)))
shock_probs = tf.constant([ p_tfp * p_depr  for p_tfp, p_depr in list(itertools.product(probs_tfp, probs_depr))])

def total_step_random(prev_state, policy_state):
    ar = AR_step(prev_state)
    if not Globals.DISABLE_SCHOCKS:
        shock = shock_step_random(prev_state)
    else:
        shock = tf.zeros_like(prev_state)
        
    policy = policy_step(prev_state, policy_state)
    
    total = ar + shock + policy     

    return augment_state(total)

# same as above, but non randomized shock, but rather the same shock for each realization
def total_step_spec_shock(prev_state, policy_state, shock_index):
    ar = AR_step(prev_state)
    shock = shock_step_spec_shock(prev_state, shock_index)
    policy = policy_step(prev_state, policy_state)
    
    total = ar + shock + policy        
    return augment_state(total)

def to_col(x):
    return [tf.expand_dims(c,axis=1) for c in x]

def augment_state(state):
    #import pdb
    #pdb.set_trace()
    # TODO: Move into Definitions + Add propapagation
    return tf.concat(
            to_col([State.TFP(state), State.depr(state)] +
            [getattr(State,"K" + str(i))(state) for i in range(2,7)] + 
            [Definitions.K_total(state, None), Definitions.r(state, None), Definitions.w(state, None),  Definitions.Y(state, None)] + 
            [Definitions.r(state, None) * getattr(State,"K" + str(i))(state) for i in range(2,7)]),
            axis=1)
    
def AR_step(prev_state):
    # only rf and yt have autoregressive components
    return tf.zeros_like(prev_state)

def shock_step_random(prev_state):
    sample_index = tf.random.categorical(tf.math.log([shock_probs]), prev_state.shape[0])[0,:]
    return tf.concat([tf.gather(shock_values,sample_index), tf.zeros([prev_state.shape[0],prev_state.shape[1] - shock_values.shape[1]])], axis=1)

def shock_step_spec_shock(prev_state, shock_index):
    # Use a specific shock - for calculating expectations
     return tf.concat([tf.tile(tf.slice(shock_values,[shock_index,0],[1,shock_values.shape[1]]),[prev_state.shape[0],1]), tf.zeros([prev_state.shape[0],prev_state.shape[1] - shock_values.shape[1]])], axis=1)

def policy_step(prev_state, policy_state):
    # tomorrow's capital holding = today's savings (as defined by policy)
    return tf.concat([tf.zeros([prev_state.shape[0],2])] + to_col([getattr(PolicyState,"a" + str(i))(policy_state) for i in range(1,6)]) + [tf.zeros([prev_state.shape[0],prev_state.shape[1]-7])], axis=1)
    