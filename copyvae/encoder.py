#! /usr/bin/env python3

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Activation
from copyvae.layers import FullyConnLayer, GaussianSampling


class Encoder(keras.models.Model):
    """ SCVI encoder """

    def __init__(self, latent_dim=10, intermediate_dim=128, n_layer=2,
                 name="encoder", **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)
        self.eps = 1e-4
        self.n_layer = n_layer

        for i in range(self.n_layer):
            setattr(
                self,
                "dense%i" %
                i,
                FullyConnLayer(
                    intermediate_dim,
                    activation=Activation('relu'),
                    bn=True,
                    keep_prob=.1))
        self.dense_mean = Dense(latent_dim)
        self.dense_var = Dense(latent_dim)
        self.sampling = GaussianSampling()

    def call(self, inputs):
        x = inputs
        for i in range(self.n_layer):
            x = getattr(self, "dense%i" % i)(x)
        z_mean = self.dense_mean(x)
        z_var = tf.math.exp(self.dense_var(x)) + self.eps
        z = self.sampling((z_mean, z_var))

        return z_mean, z_var, z


class CNEncoder(keras.models.Model):
    """ copy number encoder """

    def __init__(self,
                 original_dim,
                 max_cp,
                 name="cnencoder", **kwargs):
        super(CNEncoder, self).__init__(name=name, **kwargs)
        self.max_cp = max_cp
        self.mu_nn = keras.Sequential([
                                        Dense(original_dim),
                                        Activation(keras.activations.sigmoid)
                                        ])

    def call(self, inputs):
        copy = self.mu_nn(inputs[1]) * self.max_cp
        copy_sum = tf.reduce_sum(copy, 1, keepdims=True)
        pseudo_sum = tf.reduce_sum(inputs[0], 1, keepdims=True)
        norm_copy = copy / copy_sum * pseudo_sum

        return norm_copy