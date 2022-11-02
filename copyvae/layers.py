#! /usr/bin/env python3

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.initializers import TruncatedNormal


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
        self.fc = Dense(num_outputs,
                        kernel_initializer=TruncatedNormal(stddev=STD))
        self.bn = BatchNormalization(momentum=0.01, epsilon=0.001)
        if self.drop:
            self.dropout = Dropout(self.drop)

    def call(self, inputs):
        x = self.fc(inputs)
        if self.bn_on:
            x = self.bn(x)
        if self.act:
            x = self.act(x)
        if self.drop:
            x = self.dropout(x)
        return x


class GaussianSampling(keras.layers.Layer):
    """ Gaussian sampling """

    def call(self, inputs):
        z_mean, z_var = inputs
        sample = tf.random.normal(tf.shape(z_mean),
                                  z_mean, tf.math.sqrt(z_var))
        return sample
