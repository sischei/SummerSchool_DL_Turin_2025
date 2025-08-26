import numpy as np
import os
import tensorflow as tf
import util

DTYPE = "float64"
tf.keras.backend.set_floatx(DTYPE)
if DTYPE == "float64":
    NP_DTYPE = np.float64
elif DTYPE == "float32":
    NP_DTYPE = np.float32
else:
    raise ValueError("Unknown dtype.")
log_10 = tf.cast(tf.math.log(10.0), DTYPE)
eps = tf.cast(1e-8, DTYPE)

class ValueTrainer():
    def __init__(self, config):
        self.config = config
        self.value_config = config["value_config"]
        if self.config.get("full_state", False):
            self.d_in = config["n_basic"] + config["n_agt"]
        else:
            # TODO: d_in may be incorrect when use_logk = True
            self.d_in = config["n_basic"] + config["n_fm"] + config["n_gm"]
        self.model = util.FeedforwardModel(self.d_in, 1, self.value_config, name="v_net")
        if config["n_gm"] > 0:
            # TODO generalize to multi-dimensional agt_s
            self.gm_model = util.GeneralizedMomModel(1, config["n_gm"], config["gm_config"], name="v_gm")
        self.train_vars = None
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.value_config["lr"], epsilon=1e-8,
            beta_1=0.99, beta_2=0.99
        )
        self.use_log_k = self.config.get("use_log_k", False)
        # self.train_loss_metric = tf.keras.metrics.Mean('train_loss', dtype=log_10.dtype)
        # self.valid_loss_metric = tf.keras.metrics.Mean('valid_loss', dtype=log_10.dtype)
        # train_log_dir = os.path.join(config["model_path"], 'logs/', config["current_time"], 'vnet{}_train'.format(config["vnet_idx"]))
        # valid_log_dir = os.path.join(config["model_path"], 'logs/', config["current_time"], 'vnet{}_valid'.format(config["vnet_idx"]))
        # self.train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        # self.valid_summary_writer = tf.summary.create_file_writer(valid_log_dir)

    @tf.function
    def prepare_state(self, input_data):
        if self.use_log_k:
            log_k = tf.math.log(tf.cast(input_data["basic_s"][..., 0:1], DTYPE) + eps)/log_10
            log_k_mean = tf.math.reduce_mean(log_k, axis=1 ,keepdims=True)
            log_k_mean = tf.tile(log_k_mean, [1, input_data["basic_s"].shape[1], 1])
            basic_s = tf.concat([log_k, log_k_mean, tf.cast(input_data["basic_s"][..., 2:], DTYPE)], axis=-1)
            agt_s = tf.math.log(tf.cast(input_data["agt_s"], DTYPE))/log_10
        else:
            basic_s = tf.cast(input_data["basic_s"], DTYPE)
            agt_s = tf.cast(input_data["agt_s"], DTYPE)
        if self.config.get("full_state", False):
            state = tf.concat(
                [basic_s[..., 0:1], basic_s[..., 2:],
                 tf.repeat(tf.transpose(tf.cast(input_data["agt_s"], DTYPE), perm=[0, 2, 1]), self.config["n_agt"], axis=-2)],
                 axis=-1
            )
            # tf.print(tf.repeat(basic_s[..., 0:1], self.config["n_agt"], axis=-1) - tf.repeat(tf.cast(input_data["agt_s"], DTYPE), self.config["n_agt"], axis=-1))
        elif self.config["n_fm"] == 2:
            k_var = tf.math.reduce_variance(agt_s, axis=-2, keepdims=True)
            k_var = tf.tile(k_var, [1, agt_s.shape[1], 1])
            state = tf.concat([basic_s, k_var], axis=-1)
        elif self.config["n_fm"] == 0:
            state = tf.concat([basic_s[..., 0:1], basic_s[..., 2:]], axis=-1)
        elif self.config["n_fm"] == 1:  # so far always add k_mean in the basic_state
            state = basic_s
        if self.config["n_gm"] > 0:
            gm = self.gm_model(agt_s)
            state = tf.concat([state, gm], axis=-1)
        return state

    @tf.function
    def value_fn(self, input_data):
        state = self.prepare_state(input_data)
        value = self.model(state)
        return value

    @tf.function
    def loss(self, input_data):
        y_pred = self.value_fn(input_data)
        y = input_data["value"]
        loss = tf.reduce_mean(tf.square(y_pred - y))
        loss_dict = {"loss": loss}
        return loss_dict

    def grad(self, input_data):
        with tf.GradientTape(persistent=True) as tape:
            loss = self.loss(input_data)["loss"]
        # self.train_loss_metric(loss)
        train_vars = self.model.trainable_variables
        if self.config["n_gm"] > 0:
            train_vars += self.gm_model.trainable_variables
        self.train_vars = train_vars
        grad = tape.gradient(
            loss,
            train_vars,
            unconnected_gradients=tf.UnconnectedGradients.ZERO,
        )
        del tape
        return grad

    @tf.function
    def train_step(self, train_data):
        grad = self.grad(train_data)
        self.optimizer.apply_gradients(
            zip(grad, self.train_vars)
        )

    def train(self, train_dataset, valid_dataset, num_epoch=None, batch_size=None):
        train_dataset = train_dataset.batch(batch_size)

        for epoch in range(num_epoch+1):
            for train_data in train_dataset:
                self.train_step(train_data)
            if epoch % 20 == 0:
                for valid_data in valid_dataset:
                    val_loss = self.loss(valid_data)
                    # self.valid_loss_metric(val_loss["loss"])
                    print(
                        "Value function learning epoch: %d" % (epoch)
                    )                    

    def save_model(self, path="value_model.h5"):
        self.model.save_weights(path)
        if self.config["n_gm"] > 0:
            self.gm_model.save_weights(path.replace(".weights.h5", "_gm.weights.h5"))

    def load_model(self, path):
        self.model.load_weights_after_init(path)
        if self.config["n_gm"] > 0:
            self.gm_model.load_weights_after_init(path.replace(".weights.h5", "_gm.weights.h5"))
