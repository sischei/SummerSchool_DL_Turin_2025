import Definitions
from State import E_t_gen
from Parameters import beta, eq_scale, gamma
import PolicyState

def equations(state, policy_state):
    E_t = E_t_gen(state, policy_state)
    
    loss_dict = {}
    
    for i in range(1,6):
         loss_dict['eq_' + str(i)] =  -1 + ((beta * E_t(lambda s, ps: Definitions.r(s,ps) * (getattr(Definitions,"c" + str(i+1))(s, ps)) ** (-gamma)))** (-1 / gamma)) / (getattr(Definitions,"c" + str(i))(state, policy_state))
            
    return loss_dict
