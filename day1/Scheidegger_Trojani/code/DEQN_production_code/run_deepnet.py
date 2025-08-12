import tensorflow as tf
import Graphs

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

#### Run an episode ###
Graphs.run_cycles()
