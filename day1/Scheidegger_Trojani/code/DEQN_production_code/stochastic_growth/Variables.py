# -*- coding: utf-8 -*-
import tensorflow as tf
import math

################################################
# constants from config file

constants = {'alpha': 0.36, 
            'beta': 0.9,
            'sigma_ax': 0.01,
            'rho': 0.95}
    

#################################################
### states 

### endogenous states 
K_x_state = []
K_x_state.append({'name':'K_x','activation': 'tf.nn.softplus'})


# total endogenous state space
end_state = K_x_state


### exogenous states
a_x_state = []
a_x_state.append({'name':'a_x'})


# total endogenous state space
ex_state  = a_x_state


###############################

### total state space
states = end_state + ex_state 

print("number of states", len(states))


#################################################
### Policies

K_y_policy = []
K_y_policy.append({'name':'K_y','activation': 'tf.nn.softplus'})


### total number of policies
policies = K_y_policy

print("number of policies", len(policies))


##################################################
### definitions

#pT_policy = []
#pT_policy.append({'name':'pT','activation': 'tf.nn.softplus'})

### total number of definitions
definitions = []


print("number of definitions", len(definitions)) 
