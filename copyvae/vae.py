#!/usr/bin/env python3

import functools
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import *
from tensorflow.keras.initializers import *
from tensorflow.errors import *
from copyvae.preprocess import *

def validate_params(mu,theta,pi):
    try:
        tf.debugging.assert_non_negative(mu)
        tf.debugging.assert_non_negative(theta)
        #lower_tensor = tf.debugging.assert_greater_equal(pi, 0.0)
        #upper_tensor = tf.debugging.assert_less_equal(pi, 1.0)
    except InvalidArgumentError:
        return False
    return True


def zinb_pos(y_true, y_pred, eps=1e-8):

    """
    Parameters
    ----------
    x: Data
    mu: mean of the negative binomial (has to be positive support) (shape: minibatch x vars)
    theta: inverse dispersion parameter (has to be positive support) (shape: minibatch x vars)
    pi: logit of the dropout parameter (real support) (shape: minibatch x vars)
    eps: numerical stability constant
    """

    x = y_true
    mu = y_pred[0]
    theta = y_pred[1]
    pi = y_pred[2]

    arg_validated = validate_params(mu,theta,pi)
    if not arg_validated:
        print("invalid arguments for zinb!")
        return None

    softplus_pi = tf.math.softplus(-pi)
    log_theta_eps = tf.math.log(theta + eps)
    log_theta_mu_eps = tf.math.log(theta + mu + eps)
    pi_theta_log = -pi + theta * (log_theta_eps - log_theta_mu_eps)

    case_zero = tf.math.softplus(pi_theta_log) - softplus_pi
    mask1 = tf.cast(tf.math.less(x, eps), tf.float32)
    mul_case_zero = tf.math.multiply(mask1, case_zero)

    case_non_zero = (
        -softplus_pi
        + pi_theta_log
        + x * (tf.math.log(mu + eps) - log_theta_mu_eps)
        + tf.math.lgamma(x + theta)
        - tf.math.lgamma(theta)
        - tf.math.lgamma(x + 1)
    )
    mask2 = tf.cast(tf.math.greater(x, eps), tf.float32)
    mul_case_non_zero = tf.math.multiply(mask2, case_non_zero)
    res = mul_case_zero + mul_case_non_zero

    return tf.math.reduce_sum(res, axis=-1)


class FullyConnLayer(keras.layers.Layer):

    def __init__(self,
                  num_outputs,
                  STD=0.01,
                  keep_prob=None,
                  activation=None,
                  bn=False):
        super(FullyConnLayer, self).__init__()
        self.drop = keep_prob
        self.act = activation
        self.bn_on = bn
        self.fc = Dense(num_outputs, \
                        kernel_initializer = TruncatedNormal(stddev=STD))
        self.bn = BatchNormalization(momentum=0.01, epsilon=0.001)
        if self.drop:
          self.dropout = Dropout(self.drop)

    def call(self, inputs):
        if self.drop:
          inputs = self.dropout(inputs)
        x = self.fc(inputs)
        if self.bn_on:
          x = self.bn(x)
        if self.act:
          x = self.act(x)
        return x


class Sampling(keras.layers.Layer):

    def call(self, inputs):
        z_mean, z_log_var = inputs
        sample = tf.random.normal(tf.shape(z_mean),
                                  z_mean, tf.math.sqrt(z_log_var))
        return sample


class Encoder(layers.Layer):

    def __init__(self, latent_dim=32, intermediate_dim=64, name="encoder", **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)

        self.dense_proj1 = FullyConnLayer(intermediate_dim, activation= Activation('relu'), bn=True, keep_prob= .1)
        self.dense_proj2 = FullyConnLayer(intermediate_dim, activation= Activation('relu'), bn=True, keep_prob= .1)
        self.dense_proj3 = FullyConnLayer(intermediate_dim, activation= Activation('relu'), bn=True, keep_prob= .1)
        self.dense_mean = FullyConnLayer(latent_dim, activation=None, bn=False, keep_prob=None)
        self.dense_log_var = FullyConnLayer(latent_dim, activation= Activation('exponential'), bn=False, keep_prob=None)
        self.sampling = Sampling()

    def call(self, inputs):
        #x = tf.math.log(1 + inputs)
        x = inputs
        x = self.dense_proj1(x)
        x = self.dense_proj2(x)
        x = self.dense_proj3(x)
        z_mean = self.dense_mean(x)
        z_log_var = self.dense_log_var(x)
        z = self.sampling((z_mean, z_log_var))
        return z_mean, z_log_var, z


class Decoder(keras.layers.Layer):

    def __init__(self, original_dim, intermediate_dim, name="decoder", **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)
        self.dense_proj1 = FullyConnLayer(intermediate_dim, activation= Activation('relu'), bn=True, keep_prob=None)
        self.dense_proj2 = FullyConnLayer(intermediate_dim, activation= Activation('relu'), bn=True, keep_prob= .1)
        self.dense_proj3 = FullyConnLayer(intermediate_dim, activation= Activation('relu'), bn=True, keep_prob= .1)
        self.px_rate = FullyConnLayer(original_dim, activation= Activation('softmax'), bn=False, keep_prob=None)
        self.px_r = FullyConnLayer(original_dim, activation= None, bn=False, keep_prob=None)
        self.px_dropout = FullyConnLayer(original_dim, activation= None, bn=False, keep_prob=None)


    def call(self, inputs):
        x = self.dense_proj1(inputs)
        x = self.dense_proj2(x)
        x = self.dense_proj3(x)
        px_rate = self.px_rate(x)
        px_r = self.px_r(x)
        px_r = tf.math.exp(px_r)
        px_dropout = self.px_dropout(x)
        return [px_rate, px_r, px_dropout]


class VariationalAutoEncoder(keras.models.Model):
    """Combines the encoder and decoder into an end-to-end model for training."""

    def __init__(
        self,
        original_dim,
        intermediate_dim=64,
        latent_dim=10,
        name="autoencoder",
        **kwargs
    ):
        super(VariationalAutoEncoder, self).__init__(name=name, **kwargs)
        self.original_dim = original_dim
        self.encoder = Encoder(latent_dim=latent_dim, intermediate_dim=intermediate_dim)
        self.decoder = Decoder(original_dim, intermediate_dim=intermediate_dim)

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)
        # Add KL divergence regularization loss.
        kl_loss = 0.5 * tf.reduce_sum(
                        tf.square(z_mean) + z_log_var - tf.math.log(1e-8 + z_log_var) - 1, 1)
        self.add_loss(kl_loss)
        return reconstructed


def train_vae(data, original_dim = None, epochs = 10):

    if original_dim is None:
        original_dim = data.shape[-1]

    vae = VariationalAutoEncoder(original_dim, 128, 10)

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    mse_loss_fn = tf.keras.losses.MeanSquaredError()

    loss_metric = tf.keras.metrics.Mean()

    train_dataset = tf.data.Dataset.from_tensor_slices(data)
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(64)

    # Iterate over epochs.
    for epoch in range(epochs):
        print("Start of epoch %d" % (epoch,))

        # Iterate over the batches of the dataset.
        for step, x_batch_train in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                reconstructed = vae(x_batch_train)
                # Compute reconstruction loss
                recon = - zinb_pos(x_batch_train, reconstructed)
                loss = recon + sum(vae.losses)

            grads = tape.gradient(loss, vae.trainable_weights)
            optimizer.apply_gradients(zip(grads, vae.trainable_weights))
            #print(loss)
            loss_metric(loss)

            if step % 100 == 0:
                print("step %d: mean loss = %.4f" % (step, loss_metric.result()))

### example
"""
data_path_scvi = 'scvi_data/'
data_path_kat = 'copykat_data/txt_files/'
adata = load_cortex_txt(data_path_scvi + 'expression_mRNA_17-Aug-2014.txt')
X_train = adata.X.astype('float32')
train_vae(X_train)
"""
