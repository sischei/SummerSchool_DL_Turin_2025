import tensorflow as tf
import math
import itertools
import Definitions
import State
import Parameters
from Parameters import sigma_ax, rho, alpha, beta
import PolicyState

# shocks
shocks_ax = [x * math.sqrt(2.0) * sigma_ax for x in [-1.224744871, 0.0, +1.224744871]]
probs_ax = [x / math.sqrt(math.pi) for x in [0.2954089751, 1.181635900, 0.2954089751]]

shock_values = tf.constant(list(itertools.product(shocks_ax)))
shock_probs = tf.constant([ p_ax for p_ax in list(itertools.product(probs_ax))])

if Parameters.expectation_type == 'monomial':
   shock_values, shock_probs = State.monomial_rule([sigma_ax])

def total_step_random(prev_state, policy_state):
    ar = AR_step(prev_state)
    shock = shock_step_random(prev_state)
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

def augment_state(state):
    return State.update(state, "a_x", tf.math.exp(State.a_x(state)))

def AR_step(prev_state):
    # only rf, yt and kappa have autoregressive components
    ar_step = tf.zeros_like(prev_state)
    ar_step = State.update(ar_step, "a_x", Parameters.rho * tf.math.log(State.a_x(prev_state)) )
    return ar_step


def shock_step_random(prev_state):
    shock_step = tf.zeros_like(prev_state)
    random_normals = Parameters.rng.normal([prev_state.shape[0],1])
    shock_step = State.update(shock_step, "a_x", random_normals[:,0] * Parameters.sigma_ax)
    return shock_step

def shock_step_spec_shock(prev_state, shock_index):
    # Use a specific shock - for calculating expectations
    shock_step = tf.zeros_like(prev_state)
    shock_step = State.update(shock_step,"a_x", tf.repeat(shock_values[shock_index,0], prev_state.shape[0]))
    return shock_step

def policy_step(prev_state, policy_state):
    # coming from the lagged policy / definition
    policy_step = tf.zeros_like(prev_state)
    policy_step = State.update(policy_step, "K_x",PolicyState.K_y(policy_state))
    
 
    return policy_step
    
