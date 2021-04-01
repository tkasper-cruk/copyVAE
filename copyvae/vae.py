#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import *
from tensorflow.keras.initializers import *
from tensorflow.errors import *
from copyvae.preprocess import *


def validate_params(mu, theta):

    try:
        tf.debugging.assert_non_negative(mu)
    except InvalidArgumentError:
        print("Invalid mu")
        #print(mu)
        return False
  try:
      tf.debugging.assert_non_negative(theta)
  except InvalidArgumentError:
      print("Invalid theta")
      #print(theta)
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
    #### π in [0,1] ####
    pi = log(π/(1-π)) = log π - log(1-π)
    eps: numerical stability constant
    """

    x = y_true
    mu = y_pred[0]
    theta = y_pred[1]
    pi = y_pred[2]

    arg_validated = validate_params(mu,theta)
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
        x = self.fc(inputs)
        if self.bn_on:
          x = self.bn(x)
        if self.act:
          x = self.act(x)
        if self.drop:
          x = self.dropout(x)
        return x


class ScaleLayer(keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        super(ScaleLayer, self).__init__(*args, **kwargs)

    def build(self, input_shape):
        self.bias = self.add_weight('bias',
                                    shape=input_shape[1:],
                                    initializer='ones',
                                    trainable=True)
        self.act = Activation('softplus')

    def call(self, x):
        w = self.act(self.bias)
        x = tf.math.multiply(x, w)
        out = self.act(x)
        return out


class Sampling(keras.layers.Layer):

    def call(self, inputs):
        z_mean, z_log_var = inputs
        sample = tf.random.normal(tf.shape(z_mean),
                                  z_mean, tf.math.sqrt(z_log_var))
        return sample


class GumbelSoftmaxSampling(keras.layers.Layer):
    """ reparameterize categorical distribution """

    def call(self, inputs, temp=1.2, eps=1e-20):

        rho = tf.stack(inputs,axis=1)
        pi = tf.transpose(rho, [0, 2, 1])
        u = tf.random.uniform(tf.shape(pi) ,minval=0, maxval=1)
        g = - tf.math.log(- tf.math.log(u + eps) + eps)
        z = (tf.math.log(pi + eps) + g) / temp
        y = tf.nn.softmax(z)

        # one-hot map
        y_hard = tf.cast(tf.equal(y,tf.math.reduce_max(y,1,keepdims=True)),tf.float32)
        y = tf.stop_gradient(y_hard - y) + y

        # constract copy number matrix
        bat = tf.shape(pi)[0]
        gene = tf.shape(pi)[1]
        copy = tf.shape(pi)[2]
        a = tf.reshape(tf.range(copy), (-1, copy))
        b = tf.tile(a, (gene, 1))
        c = tf.reshape(b,(-1,gene,copy))
        cmat = tf.cast(tf.tile(c,(bat,1,1)),tf.float32)

        # copy number map
        y = tf.math.multiply(y,cmat)
        sample = tf.math.reduce_sum(y,2)

        return sample


class Encoder(keras.models.Model):

    def __init__(self, latent_dim=10, intermediate_dim=128, n_layer=1, name="encoder", **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)
        self.eps = 1e-4
        self.n_layer = n_layer

        for i in range(self.n_layer):
            setattr(self, "dense%i" % i, FullyConnLayer(intermediate_dim,
                                                        activation= Activation('relu'),
                                                        bn=True,
                                                        keep_prob= .1))
        self.dense_mean = Dense(latent_dim)
        self.dense_log_var = Dense(latent_dim)
        self.sampling = Sampling()

    def call(self, inputs):
        x = tf.math.log(1 + inputs)
        #x = inputs
        for i in range(self.n_layer):
            x = getattr(self, "dense%i" % i)(x)
        z_mean = self.dense_mean(x)
        z_log_var = tf.math.exp(self.dense_log_var(x)) + self.eps
        z = self.sampling((z_mean, z_log_var))
        return z_mean, z_log_var, z


class Decoder(keras.models.Model):

    def __init__(self, original_dim, intermediate_dim, n_layer=1, name="decoder", **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)
        self.n_layer = n_layer
        for i in range(self.n_layer):
            setattr(self, "dense%i" % i, FullyConnLayer(intermediate_dim,
                                                        activation= Activation('relu')))

        self.px_rate = Dense(original_dim, activation='exponential')
        self.px_r = Dense(original_dim)
        self.px_dropout = Dense(original_dim)


    def call(self, inputs):
        x = inputs
        for i in range(self.n_layer):
            x = getattr(self, "dense%i" % i)(x)
        px_rate = tf.clip_by_value(self.px_rate(x), clip_value_min=0, clip_value_max=12)
        px_r = self.px_r(x)
        px_r = tf.math.exp(px_r)
        px_dropout = self.px_dropout(x)
        return [px_rate, px_r, px_dropout]


class VariationalAutoEncoder(keras.models.Model):

    def __init__(
        self,
        original_dim,
        intermediate_dim=128,
        latent_dim=10,
        name="VAE",
        **kwargs
    ):
        super(VariationalAutoEncoder, self).__init__(name=name, **kwargs)
        self.original_dim = original_dim
        self.encoder = Encoder(latent_dim, intermediate_dim)
        self.decoder = Decoder(original_dim, intermediate_dim)

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)
        # Add KL divergence regularization loss.
        kl_loss = 0.5 * tf.reduce_sum(
                    tf.square(z_mean) + z_log_var \
                    - tf.math.log(1e-8 + z_log_var) - 1, 1)
        self.add_loss(kl_loss)
        return reconstructed


class DecoderCategorical(keras.models.Model):

    def __init__(self, original_dim, intermediate_dim,
                                            max_cp=6,
                                            n_layer=1,
                                            name="decoder_categorical", **kwargs):
        super(DecoderCategorical, self).__init__(name=name, **kwargs)

        self.max_cp = max_cp
        self.n_layer = n_layer
        for i in range(self.n_layer):
            setattr(self, "dense%i" % i, FullyConnLayer(intermediate_dim,
                                                        activation= Activation('relu'),
                                                        ))
        self.px_r = Dense(original_dim)
        self.px_dropout = Dense(original_dim)
        for i in range(self.max_cp):
            setattr(self, "rho%i" % i,FullyConnLayer(original_dim,
                                                    activation= Activation('softmax'),
                                                    bn=False,
                                                    keep_prob=None))
        self.k_layer = ScaleLayer()
        self.sampling = GumbelSoftmaxSampling()

    def call(self, inputs):
        x = inputs
        for i in range(self.n_layer):
            x = getattr(self, "dense%i" % i)(x)
        px_r = self.px_r(x)
        theta = tf.math.exp(px_r)
        pi = self.px_dropout(x)
        rho_list = []
        for i in range(self.max_cp):
            rho = getattr(self, "rho%i" % i)(x)
            rho_list.append(rho)
        rho_lst = tf.ones_like(rho) - tf.math.reduce_sum(rho_list, axis=0)
        rho_list.append(rho_lst)
        copy = self.sampling(rho_list)
        mu = self.k_layer(copy)
        return [mu, theta, pi]


class CopyVAE(VariationalAutoEncoder):

    def __init__(
        self,
        original_dim,
        intermediate_dim=128,
        latent_dim=10,
        name="CopyVAE",
        **kwargs):
        super().__init__(original_dim,
                         intermediate_dim,
                         latent_dim,
                         name)
        self.decoder = DecoderCategorical(original_dim, intermediate_dim)



def train_vae(vae, data, batch_size = 128, epochs = 10):

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    loss_metric = tf.keras.metrics.Mean()

    train_dataset = tf.data.Dataset.from_tensor_slices(data)
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

    # Iterate over epochs.
    for epoch in range(epochs):
        print("Start of epoch %d" % (epoch,))

        # Iterate over the batches of the dataset.
        for step, x_batch_train in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                reconstructed = vae(x_batch_train)
                # Compute reconstruction loss
                recon = - zinb_pos(x_batch_train, reconstructed)
                loss = recon + vae.losses   #sum(vae.losses)

            grads = tape.gradient(loss, vae.trainable_weights)
            optimizer.apply_gradients(zip(grads, vae.trainable_weights))
            loss_metric(loss)

            if step % 100 == 0:
                #print("step %d: mean loss = %.4f" % (step, loss_metric.result()))
                print("step %d: mean loss = %s" % (step, "{:.2e}".format(loss_metric.result())))
    return vae


### example
"""
data_path_scvi = 'scvi_data/'
data_path_kat = 'copykat_data/txt_files/'
adata = load_cortex_txt(data_path_scvi + 'expression_mRNA_17-Aug-2014.txt')
x_train = adata.X
#model = VariationalAutoEncoder(x_train.shape[-1], 128, 10)
model = CopyVAE(x_train.shape[-1], 128, 10)
copy_vae = train_vae(model, x_train, epochs = 400)
"""
