import Parameters
import PolicyState
import State
import sys
import Definitions

def K_total(state, policy_state=None):
    return State.K2(state) + State.K3(state) + State.K4(state) + State.K5(state) + State.K6(state)

def K_total_next(state, policy_state):
    return PolicyState.a1(policy_state) + PolicyState.a2(policy_state) + PolicyState.a3(policy_state) + PolicyState.a4(policy_state) + PolicyState.a5(policy_state)

def r(state, policy_state=None):
    return Parameters.alpha * State.TFP(state) * Definitions.K_total(state, None)**(Parameters.alpha - 1)  + (1 - State.depr(state))

def w(state, policy_state=None):
    return  (1 - Parameters.alpha) * State.TFP(state) * Definitions.K_total(state, None)**Parameters.alpha 

def Y(state, policy_state=None):
    return State.TFP(state) * Definitions.K_total(state, None)**Parameters.alpha + (1 - State.depr(state)) * Definitions.K_total(state, None)

# consumption definitions
for i in range(1,7):
    # only youngest generation has labour income
    if i == 1:
        setattr(sys.modules[__name__], "c" + str(i), lambda s, ps: w(s,ps) - PolicyState.a1(ps))
    if i > 1 and i < 6:
        setattr(sys.modules[__name__], "c" + str(i), (lambda ind: lambda s, ps: r(s, ps) * getattr(State, "K" + str(ind))(s) - getattr(PolicyState, "a" +  str(ind))(ps))(i))
    if i == 6:
        # consume everything
        setattr(sys.modules[__name__], "c" + str(i), lambda s, ps: r(s, ps) * State.K6(s))
        

