import tensorflow as tf
import Definitions
import PolicyState
import State
from Parameters import alpha, beta, rho, sigma_ax


def equations(state, policy_state):
    E_t = State.E_t_gen(state, policy_state)
    
    loss_dict = {}
    
    # policy today
    Knext = PolicyState.K_y(policy_state)
    
    # output today
    Y = State.a_x(state) * State.K_x(state) ** alpha 
    
    # consumption today
    C = tf.math.maximum(Y - Knext, 1e-5)
    
    # function which gives the RHS of the Euler equation for a given states tomorrow
    def RHS(snext, psnext):
        
        # policy tomorrow
        Knextnext = PolicyState.K_y(psnext)
        
        # output tomorrow
        Ynext = State.a_x(snext) * State.K_x(snext) ** alpha 
        
        # Return tomorrow
        Rnext = alpha * State.a_x(snext) * State.K_x(snext) ** (alpha - 1.) 
        
        # consumption tomorrow
        Cnext = tf.math.maximum(Ynext - Knextnext, 1e-5)
        
        # RHS Euler Equation
        RHS = beta * (1. / Cnext) * Rnext
        
        return RHS
        
        
    # relative error
    loss_dict['eq_1'] = 1 / (C * E_t(lambda snext, psnext: RHS(snext, psnext))) - 1.0  
    
    #loss_dict['eq_1'] = E_t(lambda snext, psnext: RHS(snext, psnext)) - 1.0 / C  





    return loss_dict
