# calculate equilibrium conditions, based on state values & estimated policy function

import importlib
import tensorflow as tf
import PolicyState 
import Definitions
from Parameters import definition_bounds_hard, horovod_worker, MODEL_NAME, optimizer, policy_bounds_hard, loss_choice

Equations = importlib.import_module(MODEL_NAME + ".Equations")

def penalty_bounds_policy(state, policy_state):
    res = tf.constant(0.0)
    for bound_vars in policy_bounds_hard['lower'].keys():
        # bounded is always >= raw - we measure how strong this bound is
        raw_vs_bounded = getattr(PolicyState, bound_vars )(policy_state) - getattr(PolicyState, bound_vars + "_RAW")(policy_state)
        penalty = tf.math.reduce_sum(policy_bounds_hard['penalty_lower'][bound_vars] * (raw_vs_bounded ** 2))
        tf.summary.scalar('penalty_lower_policy_'+bound_vars, penalty)
        res += penalty
    
    for bound_vars in policy_bounds_hard['upper'].keys():
        # bounded is always <= raw - we measure how strong this bound is
        raw_vs_bounded = getattr(PolicyState, bound_vars + "_RAW")(policy_state) - getattr(PolicyState, bound_vars )(policy_state) 
        penalty = tf.math.reduce_sum(policy_bounds_hard['penalty_upper'][bound_vars] * (raw_vs_bounded ** 2))
        if not horovod_worker:
            tf.summary.scalar('penalty_upper_policy_' + bound_vars, penalty)
        res += penalty
    
    for bound_vars in definition_bounds_hard['lower'].keys():
        # bounded is always >= raw - we measure how strong this bound is
        raw_vs_bounded = getattr(Definitions, bound_vars )(state, policy_state) - getattr(Definitions, bound_vars + "_RAW")(state, policy_state)
        penalty = tf.math.reduce_sum(definition_bounds_hard['penalty_lower'][bound_vars] * (raw_vs_bounded ** 2))
        if not horovod_worker:
            tf.summary.scalar('penalty_lower_def_'+bound_vars, penalty)
        res += penalty
    
    for bound_vars in definition_bounds_hard['upper'].keys():
        # bounded is always <= raw - we measure how strong this bound is
        raw_vs_bounded = getattr(Definitions, bound_vars + "_RAW")(state, policy_state) - getattr(Definitions, bound_vars )(state, policy_state)
        penalty = tf.math.reduce_sum(definition_bounds_hard['penalty_upper'][bound_vars] * (raw_vs_bounded ** 2))
        if not horovod_worker:
            tf.summary.scalar('penalty_upper_def_' + bound_vars, penalty)
        res += penalty
    
    return res

# Huber loss
Huber_loss_delta = 1.0
def Huber_loss(delta_y, delta):
    abs = tf.math.abs(delta_y)
    return tf.where(
        tf.less_equal(abs,delta),
        0.5*delta_y**2,
        delta*(abs - 0.5*delta)) 

def loss(state, policy_state):
    loss_val = tf.constant(0.0)         # total loss
    net_loss_val = tf.constant(0.0)     # net loss (without penalty)
    tf.summary.experimental.set_step(optimizer.iterations)
    losses = Equations.equations(state, policy_state)
    for eq_f in losses.keys():
        if loss_choice == 'huber':
            eq_loss = tf.math.reduce_sum(Huber_loss(losses[eq_f], Huber_loss_delta))
        else: 
            eq_loss = tf.math.reduce_sum((losses[eq_f]) ** 2)
            
        if not horovod_worker:
            tf.summary.scalar('dev_' + eq_f, eq_loss)
        loss_val += eq_loss
        
    net_loss_val = loss_val
    loss_val += penalty_bounds_policy(state, policy_state)
    #normalize loss with number of equations
    no_eq = len(losses)
    
    tf.summary.scalar('dev_loss', loss_val/no_eq)
    tf.summary.scalar('dev_net_loss', net_loss_val/no_eq)    
    
    return loss_val/no_eq, net_loss_val/no_eq
