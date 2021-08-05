#! /usr/bin/env python3

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow.keras as keras
from tensorflow.keras.layers import *
from tensorflow.keras.initializers import *
from tensorflow.errors import InvalidArgumentError
from tqdm import tqdm


def validate_params(mu, theta):

    try:
        tf.debugging.assert_non_negative(mu)
    except InvalidArgumentError:
        print("Invalid mu")
        # print(mu)
        # return False
        raise
    try:
        tf.debugging.assert_non_negative(theta)
    except InvalidArgumentError:
        print("Invalid theta")
        # print(theta)
        # return False
        raise
    return True


def zinb_pos(y_true, y_pred, eps=1e-8):
    """ zero-inflated negative binomial reconstruction loss

    Args:
        y_true: true values
        y_pred: predicted values
        eps: numerical stability constant
    Parameters:
        x: Data
        mu: mean of the negative binomial (positive (batch x vars)
        theta: inverse dispersion parameter (positive) (batch x vars)
        pi: logit of the dropout parameter (real number) (batch x vars)
        #### π in [0,1] ####
        pi = log(π/(1-π)) = log π - log(1-π)
    Returns:
        loss
    """

    x = y_true
    mu = y_pred[0]
    theta = y_pred[1]
    pi = y_pred[2]

    arg_validated = validate_params(mu, theta)
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


def poisson_prior(batch_dim, genes_dim, max_cp=6, lam=2):
    """ poisson-like categorical distribution

    Args:
        batch_dim: number of example in minibatch
        genes_dim: number of genes
        max_cp: maximum copy number
        lam: poisson mean
    Returns:
        cat_dis: categorical distribution object
    """

    poi = tfp.distributions.Poisson(lam)
    poi_prob = poi.prob(np.arange(max_cp + 1))
    cat_prob = poi_prob / np.sum(poi_prob)
    a = tf.expand_dims(cat_prob, axis=0)
    b = tf.expand_dims(a, axis=0)
    c = tf.repeat(b, repeats=genes_dim, axis=1)
    d = tf.cast(tf.repeat(c, repeats=batch_dim, axis=0), tf.float32)
    cat_dis = tfp.distributions.Categorical(probs=d)

    return cat_dis


def dirichlet_prior(batch, genes):
    """ categorical distribution from a dirichlet prior

    Args:
        batch: number of example in minibatch
        genes: number of genes
    Returns:
        cat_dis: categorical distribution object
    """

    alpha = [1, 5, 10, 5, 1, 1, 1]
    dist = tfp.distributions.Dirichlet(alpha)
    d = dist.sample([batch, genes])
    cat_dis = tfp.distributions.Categorical(probs=d)
    return cat_dis


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


class ScaleLayer(keras.layers.Layer):

    def __init__(self, *args, **kwargs):
        super(ScaleLayer, self).__init__(*args, **kwargs)

    def build(self, input_shape):
        self.w = self.add_weight('weight',
                                 shape=input_shape[1:],
                                 initializer='random_normal',
                                 trainable=True)
        self.b = self.add_weight('bias',
                                 shape=input_shape[1:],
                                 initializer='random_normal',
                                 trainable=True)
        self.act = Activation('sigmoid')

    def call(self, x):
        w = self.w
        x = tf.math.multiply(x, w) + self.b
        out = self.act(x)
        return out


class Sampling(keras.layers.Layer):
    """ Gaussian sampling """

    def call(self, inputs):
        z_mean, z_log_var = inputs
        sample = tf.random.normal(tf.shape(z_mean),
                                  z_mean, tf.math.sqrt(z_log_var))
        return sample


class GumbelSoftmaxSampling(keras.layers.Layer):
    """ Reparameterize categorical distribution """

    def call(self, inputs, bin_size=25, temp=0.1, eps=1e-20):

        # reshape the dimensions (batch x gene x copies)
        rho = tf.stack(inputs, axis=1)
        #pi = tf.transpose(rho, [0, 2, 1])
        gene_rho = tf.repeat(rho, repeats=bin_size, axis=-1)
        pi = tf.transpose(gene_rho, [0, 2, 1])

        # sample from Gumbel(0, 1)
        u = tf.random.uniform(tf.shape(pi), minval=0, maxval=1)
        # Gumbel-Softmax
        g = - tf.math.log(- tf.math.log(u + eps) + eps)
        z = (tf.math.log(pi + eps) + g) / temp
        y = tf.nn.softmax(z, axis=-1)

        # one-hot map using argmax, but differentiate w.r.t. soft sample y
        y_hard = tf.cast(
            tf.equal(y,
                     tf.math.reduce_max(y, axis=-1, keepdims=True)
                     ),
            tf.float32)
        y = tf.stop_gradient(y_hard - y) + y

        # constract copy number matrix
        bat = tf.shape(pi)[0]
        gene = tf.shape(pi)[1]
        copy = tf.shape(pi)[2]
        a = tf.reshape(tf.range(copy), (-1, copy))
        b = tf.tile(a, (gene, 1))
        c = tf.reshape(b, (-1, gene, copy))
        cmat = tf.cast(tf.tile(c, (bat, 1, 1)), tf.float32)

        # copy number map
        y = tf.math.multiply(y, cmat)
        sample = tf.math.reduce_sum(y, axis=-1)

        return sample, pi


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
    """ SCVI decoder """

    def __init__(
            self,
            original_dim,
            intermediate_dim,
            n_layer=3,
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

        self.px_rate = Dense(original_dim, activation='exponential')
        self.px_r = Dense(original_dim)
        self.px_dropout = Dense(original_dim)

    def call(self, inputs):
        x = inputs
        for i in range(self.n_layer):
            x = getattr(self, "dense%i" % i)(x)
        px_rate = tf.clip_by_value(self.px_rate(x), clip_value_min=0,
                                   clip_value_max=12)
        px_r = self.px_r(x)
        px_r = tf.math.exp(px_r)
        px_dropout = self.px_dropout(x)
        return [px_rate, px_r, px_dropout]


class VariationalAutoEncoder(keras.models.Model):
    """ SCVI """

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
            tf.square(z_mean) + z_log_var
            - tf.math.log(1e-8 + z_log_var) - 1,
            1)
        self.add_loss(kl_loss)
        return reconstructed


class DecoderCategorical(keras.models.Model):
    """ CopyVAE decoder """

    def __init__(self, original_dim, intermediate_dim,
                 bin_size,
                 max_cp,
                 n_layer=2,
                 name="decoder_categorical", **kwargs):
        super(DecoderCategorical, self).__init__(name=name, **kwargs)

        self.max_cp = max_cp
        self.n_layer = n_layer
        self.bin_size = bin_size
        for i in range(self.n_layer):
            setattr(
                self,
                "dense%i" %
                i,
                FullyConnLayer(
                    intermediate_dim,
                    activation=Activation('relu'),
                    bn=True))
        self.px_r = Dense(original_dim)
        self.px_dropout = Dense(original_dim)
        for i in range(self.max_cp + 1):
            #setattr(self, "rho%i" % i, Dense(original_dim))
            setattr(self, "rho%i" % i, Dense(original_dim // bin_size))
        self.sampling = GumbelSoftmaxSampling()
        self.k_layer = ScaleLayer()

    def call(self, inputs):
        x = inputs
        for i in range(self.n_layer):
            x = getattr(self, "dense%i" % i)(x)

        px_r = self.px_r(x)
        theta = tf.math.exp(px_r)
        pi = self.px_dropout(x)

        # categorical parameters
        rho_list = []
        for i in range(self.max_cp + 1):
            rho = getattr(self, "rho%i" % i)(x)
            rho_list.append(rho)
        rho_list = tf.nn.softmax(rho_list, axis=0)
        copy, cat_prob = self.sampling(rho_list)

        mu = self.k_layer(copy)
        return [mu, theta, pi], copy, cat_prob


class CopyVAE(VariationalAutoEncoder):

    def __init__(
            self,
            original_dim,
            intermediate_dim=128,
            latent_dim=10,
            bin_size=25,
            max_cp=6,
            decoder_layers=2,
            name="CopyVAE",
            **kwargs):
        super().__init__(original_dim,
                         intermediate_dim,
                         latent_dim,
                         name)
        self.decoder = DecoderCategorical(original_dim,
                                          intermediate_dim,
                                          bin_size=bin_size,
                                          max_cp=max_cp,
                                          n_layer=decoder_layers
                                          )

    def call(self, inputs):

        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructed, copy, rho = self.decoder(z)
        # latent KL
        kl_loss = 0.5 * tf.reduce_sum(
            tf.square(z_mean) + z_log_var
            - tf.math.log(1e-8 + z_log_var) - 1,
            1)
        self.add_loss(kl_loss)
        # copy number KL
        cn_dis = tfp.distributions.Categorical(probs=rho)
        cn_prior = poisson_prior(rho.shape[0], rho.shape[1])
        kl_copy = 0.05 * tf.reduce_sum(
            tfp.distributions.kl_divergence(
                cn_dis,
                cn_prior),
            1)
        self.add_loss(kl_copy)

        return reconstructed


def train_vae(vae, data, batch_size=128, epochs=10):
    """ Train function

    Args:
        vae: VAE object
        data: training examples
        batch_size: number of example in minibatch
        epochs: epochs
    Returns:
        trained model
    """

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    loss_metric = tf.keras.metrics.Mean()

    train_dataset = tf.data.Dataset.from_tensor_slices(data)
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

    # Iterate over epochs.
    tqdm_progress = tqdm(range(epochs), desc='model training')
    for epoch in tqdm_progress:

        # Iterate over the batches of the dataset.
        for step, x_batch_train in enumerate(train_dataset):
            try:
                with tf.GradientTape() as tape:
                    reconstructed = vae(x_batch_train)
                    # Compute reconstruction loss
                    recon = - zinb_pos(x_batch_train, reconstructed)
                    loss = recon + vae.losses  # sum(vae.losses)

                grads = tape.gradient(loss, vae.trainable_weights)
                optimizer.apply_gradients(zip(grads, vae.trainable_weights))
                loss_metric(loss)
            except BaseException:
                return vae
            if step % 100 == 0:
                tqdm_progress.set_postfix_str(
                    s="loss={:.2e}".format(
                        loss_metric.result()), refresh=True)

    return vae
