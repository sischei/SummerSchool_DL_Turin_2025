import importlib
import tensorflow as tf
import Equilibrium
import Parameters
import gc

if Parameters.horovod:
    import horovod.tensorflow as hvd

Dynamics = importlib.import_module(Parameters.MODEL_NAME + ".Dynamics")
Hooks = importlib.import_module(Parameters.MODEL_NAME + ".Hooks")

"""
Overview:
    - Episode > Epochs (1 full scan of an episode) > Mini-batches > Adam
"""

@tf.function
def do_random_step(current_state):
    return Dynamics.total_step_random(current_state, Parameters.policy(current_state))

def run_episode(state_episode):
    """ Runs an episode starting from the begging of the state_episode. Results are saved into state_episode."""
    current_state = state_episode[0,:,:]
    
    # run optimization
    for i in range(1, state_episode.shape[0]):
        current_state = do_random_step(current_state)
        # replace above for deterministic results
        # current_state = BatchState.total_step_spec_shock(current_state, Parameters.policy(current_state),0)
        state_episode = tf.tensor_scatter_nd_update(state_episode, tf.constant([[ i ]]), tf.expand_dims(current_state, axis=0))
    
    return state_episode

@tf.function
def run_grads(state_sample, first_batch):
    """Runs a single gradient step using Adam for a minibatch"""
    with Parameters.writer.as_default():
        with tf.GradientTape() as tape:
            loss, net_loss = Equilibrium.loss(state_sample, Parameters.policy(state_sample))

    if Parameters.horovod:
        # Horovod: add Horovod Distributed GradientTape.
        tape = hvd.DistributedGradientTape(tape)
        
    grads = tape.gradient(loss, Parameters.policy_net.trainable_variables)
    Parameters.optimizer.apply_gradients(zip(grads, Parameters.policy_net.trainable_variables))
    
    # Note: broadcast should be done after the first gradient step to ensure optimizer
    # initialization.
    if first_batch:
        tf.print("Broadcasting variables....")
        hvd.broadcast_variables(Parameters.policy_net.variables, root_rank=0)
        hvd.broadcast_variables(Parameters.optimizer.variables(), root_rank=0)
    
    return loss, net_loss

def run_epoch(state_episode):
    # we have a larger effective sample size as we batch simulated
    effective_size = state_episode.shape[0] * state_episode.shape[1]
    if not Parameters.sorted_within_batch:
        batches =  tf.data.Dataset.from_tensor_slices(tf.reshape(state_episode, [effective_size,len(Parameters.states)])).shuffle(buffer_size=effective_size).batch(Parameters.N_minibatch_size, drop_remainder=True)
    else:
        batches = tf.data.Dataset.from_tensor_slices(tf.reshape(tf.transpose(state_episode,[1,0,2]), [effective_size,len(Parameters.states)])).batch(Parameters.N_minibatch_size, drop_remainder=True).shuffle(buffer_size=int(effective_size/Parameters.N_minibatch_size))
    epoch_loss = 0.0
    net_epoch_loss = 0.0
    
    for batch in batches:
        epoch_loss_1, net_epoch_loss_1 = run_grads(batch, Parameters.horovod and Parameters.optimizer.iterations == Parameters.optimizer_starting_iteration)    
        epoch_loss += epoch_loss_1
        net_epoch_loss += net_epoch_loss_1
            
    return epoch_loss, net_epoch_loss
 
def run_cycle(state_episode):
    """ Runs an iteration cycle startin from a given BatchState.
    
    It creates an episode and then runs N_epochs_per_episode epochs on the data.
    """
    
    state_episode = run_episode(state_episode)

    # starting learning phase - needed for DROPOUT layer to become active
    tf.keras.backend.set_learning_phase(1)

    file1 = open(Parameters.LOG_DIR + "/" + Parameters.error_filename,"a") 
    for e in range(Parameters.N_epochs_per_episode):
        #print("Current Time (before epoch) =", datetime.now().strftime("%H:%M:%S"))
        epoch_loss, net_epoch_loss = run_epoch(state_episode)
        MSE_epoch_loss = epoch_loss / (Parameters.N_episode_length * Parameters.N_sim_batch)
        Norm_epoch_loss = tf.math.sqrt(epoch_loss / (Parameters.N_episode_length * Parameters.N_sim_batch))
        MSE_epoch_no_penalty = net_epoch_loss / (Parameters.N_episode_length * Parameters.N_sim_batch)
        Norm_epoch_loss_no_penalty = tf.math.sqrt(net_epoch_loss / (Parameters.N_episode_length * Parameters.N_sim_batch))
        
        tf.print("----------------------------------")
        tf.print("Normalized MSE epoch loss: " + str(MSE_epoch_loss))
        tf.print("Normalized epoch loss: " + str(Norm_epoch_loss))
        tf.print("----------------------------------")
        tf.print("Normalized MSE epoch loss without penalties: " + str(MSE_epoch_no_penalty))
        tf.print("Normalized epoch loss without penalties: " + str(Norm_epoch_loss_no_penalty))
        tf.print("==================================")
        file1.write("MSE:            " + str(tf.get_static_value(MSE_epoch_loss)) + "  ")
        file1.write("MAE:            " + str(tf.get_static_value(Norm_epoch_loss)) + "  ")
        file1.write("MSE_no_penalty: " + str(tf.get_static_value(MSE_epoch_no_penalty)) + "  ")
        file1.write("MAE_no_penatly: " + str(tf.get_static_value(Norm_epoch_loss_no_penalty)) + "\n")
        
    file1.close() 


    # stopping learning phase
    tf.keras.backend.set_learning_phase(0)
        
    return state_episode

def run_cycles():
    if "post_init" in dir(Hooks) and Parameters.ckpt.current_episode < 2:
        print("Running post-init hook...")
        Hooks.post_init()
        print("Starting state after post-init:")
        print(Parameters.starting_state)

    state_episode = tf.tile(tf.expand_dims(Parameters.starting_state, axis = 0), [Parameters.N_episode_length, 1, 1])

    start_time = tf.timestamp()

    for i in range(Parameters.N_episodes):       
        tf.print("Running episode: " + str(Parameters.ckpt.current_episode.numpy()))
        state_episode = run_cycle(state_episode)
        # start again from previous last state
        if Parameters.initialize_each_episode:
            print("Running with states re-drawn after each episode!") 
            Parameters.starting_state.assign(Parameters.initialize_states())
        else:
            Parameters.starting_state.assign(state_episode[Parameters.N_episode_length-1,:,:])
            
        state_episode = tf.tensor_scatter_nd_update(state_episode, tf.constant([[ 0 ]]), tf.expand_dims(Parameters.starting_state, axis=0))
        
        # create checkpoint
        Parameters.ckpt.current_episode.assign_add(1)
        
        if not Parameters.horovod_worker:
            Parameters.manager.save()
            # run hooks
            with Parameters.writer.as_default():
                Hooks.cycle_hook(state_episode[0,:,:],i)
                
        tf.print("Elapsed time since start: ", tf.timestamp() - start_time)

        if i % 10 == 0:
            tf.print("Garbage collecting")
            tf.keras.backend.clear_session()
            gc.collect()
