import Parameters
import PolicyState
import State
import matplotlib.pyplot as plt
import tensorflow as tf

def cycle_hook(state,i):
    A = 6   
    if i % 100 == 0:
        p = Parameters.policy(state)

        # a1
        plt.plot(Parameters.beta * State.w(state).numpy() * (1- Parameters.beta ** (A-1)) / (1 - Parameters.beta ** A), PolicyState.a1(p), 'bs')
        plt.savefig(Parameters.LOG_DIR + '/a1.png')
        plt.close()
        
        # a2
        plt.plot(Parameters.beta * (State.r(state) * State.K2(state)).numpy() * (1- Parameters.beta ** (A-2)) / (1 - Parameters.beta ** (A-1)), PolicyState.a2(p), 'bs')
        plt.savefig(Parameters.LOG_DIR + '/a2.png')
        plt.close()
    
        plt.plot(Parameters.beta * (State.r(state) * State.K3(state)).numpy() * (1- Parameters.beta ** (A-3)) / (1 - Parameters.beta ** (A-2)), PolicyState.a3(p), 'bs')
        plt.savefig(Parameters.LOG_DIR + '/a3.png')
        plt.close()
    
        plt.plot(Parameters.beta * (State.r(state) * State.K4(state)).numpy() * (1- Parameters.beta ** (A-4)) / (1 - Parameters.beta ** (A-3)), PolicyState.a4(p), 'bs')
        plt.savefig(Parameters.LOG_DIR + '/a4.png')
        plt.close()
    
        plt.plot(Parameters.beta * (State.r(state) * State.K5(state)).numpy() * (1- Parameters.beta ** (A-5)) / (1 - Parameters.beta ** (A-4)), PolicyState.a5(p), 'bs')
        plt.savefig(Parameters.LOG_DIR + '/a5.png')
        plt.close()    
        
    
def post_init():
    Parameters.starting_state.assign(State.update(Parameters.starting_state, "K2",tf.constant(0.5,shape=(Parameters.starting_state.shape[0],))))