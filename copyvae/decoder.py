#! /usr/bin/env python3

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Softmax, Activation
from copyvae.layers import FullyConnLayer

class Decoder(keras.models.Model):
    """ SCVI decoder """

    def __init__(
            self,
            original_dim,
            intermediate_dim,
            n_layer=2,
            name="decoder",
            **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)
        self.n_layer = n_layer
        for i in range(self.n_layer):
            setattr(
                self,
                "dense%i" %
                i,
                FullyConnLayer(
                    intermediate_dim,
                    activation=Activation('relu'),
                    bn=True))

        self.px_scale_decoder = keras.Sequential([
            Dense(original_dim),
            Softmax(axis=-1)
        ])
        self.px_r_decoder = Dense(original_dim)
        self.px_dropout_decoder = Dense(original_dim)

    def call(self, inputs):
        x = inputs[0]
        lib = inputs[1]
        for i in range(self.n_layer):
            x = getattr(self, "dense%i" % i)(x)
        px = x
        px_scale = self.px_scale_decoder(px)
        px_dropout = self.px_dropout_decoder(px)
        px_rate = lib * px_scale
        px_r = self.px_r_decoder(px)
        px_r = tf.math.exp(px_r)

        return [px_rate, px_r, px_dropout]


class CNDecoder(keras.models.Model):
    """ second decoder """

    def __init__(
            self,
            original_dim,
            intermediate_dim,
            n_layer=2,
            name="cndecoder",
            **kwargs):
        super(CNDecoder, self).__init__(name=name, **kwargs)
        self.n_layer = n_layer
        for i in range(self.n_layer):
            setattr(
                self,
                "dense%i" %
                i,
                FullyConnLayer(
                    intermediate_dim,
                    activation=Activation('relu'),
                    bn=True))

        self.px_r_decoder = Dense(original_dim)

    def call(self, inputs):
        x = inputs[0]
        px = inputs[1]

        px_r = self.px_r_decoder(px)
        px_r = tf.math.exp(px_r)

        return [x, px_r]