import os
# os.environ['PYTHONHASHSEED'] = '0'  # Use a fixed value, not dynamic seed
# os.environ['TF_DETERMINISTIC_OPS'] = '1'
# os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
# os.environ['OMP_NUM_THREADS'] = '1'
# os.environ['TF_NUM_INTRAOP_THREADS'] = '1'
# os.environ['TF_NUM_INTEROP_THREADS'] = '1'

import tensorflow as tf
# gpus = tf.config.list_physical_devices('GPU')
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)

import json
import time
import datetime
from absl import app
from absl import flags
from param import KSParam
from dataset import KSInitDataSet
from value import ValueTrainer
from policy import KSPolicyTrainer
from util import print_elapsedtime
from util import set_random_seed

flags.DEFINE_string("config_path", "./configs/KS/game_nn_n50.json",
                    """The path to load json file.""",
                    short_name='c')
flags.DEFINE_string("exp_name", "test",
                    """The suffix used in model_path for save.""",
                    short_name='n')
flags.DEFINE_integer("seed_index", 0,
                    """The suffix chooses index of random seeds.""",
                    short_name='s')
FLAGS = flags.FLAGS

def main(argv):
    del argv
    with open(FLAGS.config_path, 'r') as f:
        config = json.load(f)

    # Set random seed from config
    if "random_seed" in config:
        seed = config["random_seed"][FLAGS.seed_index]
        set_random_seed(seed)
        print(f"Using seed {seed} (index {FLAGS.seed_index})")

    print("Solving the problem based on the config path {}".format(FLAGS.config_path))
    mparam = KSParam(config["n_agt"], config["beta"], config["mats_path"])
    # save config at the beginning for checking
    model_path = "../data/simul_results/KS/{}_{}_n{}_{}".format(
        "game" if config["policy_config"]["opt_type"] == "game" else "sp",
        config["dataset_config"]["value_sampling"],
        config["n_agt"],
        FLAGS.exp_name,
    )
    config["model_path"] = model_path
    config["current_time"] = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    os.makedirs(model_path, exist_ok=True)
    with open(os.path.join(model_path, "config_beg.json"), 'w') as f:
        json.dump(config, f)

    start_time = time.monotonic()

    # initial value training
    init_ds = KSInitDataSet(mparam, config)
    value_config = config["value_config"]
    if config["init_with_bchmk"]:
        init_policy = init_ds.k_policy_bchmk
        policy_type = "pde"
    else:
        init_policy = init_ds.c_policy_const_share
        policy_type = "nn_share"
    train_vds, valid_vds = init_ds.get_valuedataset(init_policy, policy_type, update_init=False)
    vtrainers = []
    for i in range(value_config["num_vnet"]):
        config["vnet_idx"] = str(i)
        vtrainers.append(ValueTrainer(config))
    for vtr in vtrainers:
        vtr.train(train_vds, valid_vds, value_config["num_epoch"], value_config["batch_size"])

    # iterative policy and value training
    policy_config = config["policy_config"]
    ptrainer = KSPolicyTrainer(vtrainers, init_ds)
    ptrainer.train(policy_config["num_step"], policy_config["batch_size"])

    # save config and models
    with open(os.path.join(model_path, "config.json"), 'w') as f:
        json.dump(config, f)
    for i, vtr in enumerate(vtrainers):
        vtr.save_model(os.path.join(model_path, "value{}.weights.h5".format(i)))

    ptrainer.save_model(os.path.join(model_path, "policy.weights.h5"))

    end_time = time.monotonic()
    print_elapsedtime(end_time - start_time)

    # Save computational time to a file
    elapsed_time = end_time - start_time
    
    time_log_path = os.path.join(model_path, "time.txt")
    with open(time_log_path, 'w') as time_file:
        time_file.write(f"Solving the problem based on the config path {FLAGS.config_path} took {elapsed_time:.2f} seconds.\n")

if __name__ == '__main__':
    app.run(main)
