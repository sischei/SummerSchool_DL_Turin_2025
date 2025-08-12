#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 08:26:23 2020

@author: -
"""
import sys
import numpy as np
import tensorflow as tf
from Parameters import policy_states, policy_bounds_hard, policy

for i,policy_state in enumerate(policy_states):
    if (policy_state in policy_bounds_hard['lower']) or (policy_state in policy_bounds_hard['upper']):
        setattr(sys.modules[__name__],policy_state, 
                (lambda ind: lambda x: tf.clip_by_value(x[:,ind], policy_bounds_hard['lower'].get(policy_states[ind], np.NINF), policy_bounds_hard['upper'].get(policy_states[ind],np.Inf)))(i)
                )
    else:
        setattr(sys.modules[__name__],policy_state, (lambda ind: lambda x: x[:,ind])(i))
        
    # always add a 'raw' attribute as well - this can be used for penalties
    setattr(sys.modules[__name__],policy_state + "_RAW", (lambda ind: lambda x: x[:,ind])(i))

    # policy functions (where policy_state is explicitly calculated from current state using current policy)
    setattr(sys.modules[__name__],policy_state + "_POLICY_FROM_STATE", (lambda ind: lambda state: policy(state)[:,ind])(i))