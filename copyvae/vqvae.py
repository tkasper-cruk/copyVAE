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


class Encoder(keras.models.Model):
    """ SCVI encoder """

    def __init__(self, latent_dim=10, intermediate_dim=128, n_layer=3,
                 name="encoder", **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)
        #self.eps = 1e-4
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
        #self.dense = Dense(latent_dim)
        #self.dense_log_var = Dense(latent_dim)
        #self.sampling = Sampling()

    def call(self, inputs):
        x = tf.math.log(1 + inputs)
        for i in range(self.n_layer):
            x = getattr(self, "dense%i" % i)(x)
        #z = self.dense(x)
        #z_log_var = tf.math.exp(self.dense_log_var(x)) + self.eps
        #z = self.sampling((z_mean, z_log_var))
        return x


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



class VectorQuantizer(Layer):
    def __init__(self, num_embeddings, embedding_dim, beta=0.25, **kwargs):
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.beta = (
            beta  # This parameter is best kept between [0.25, 2] as per the paper.
        )

        # Initialize the embeddings which we will quantize.
        w_init = tf.random_uniform_initializer()
        self.embeddings = tf.Variable(
            initial_value=w_init(
                shape=(self.embedding_dim, self.num_embeddings), dtype="float32"
            ),
            trainable=True,
            name="embeddings_vqvae",
        )

    def call(self, x):
        # Calculate the input shape of the inputs and
        # then flatten the inputs keeping `embedding_dim` intact.
        #input_shape = tf.shape(x)
        #flattened = tf.reshape(x, [-1, self.embedding_dim])
        flattened = x

        # Quantization.
        encoding_indices = self.get_code_indices(flattened)
        encodings = tf.one_hot(encoding_indices, self.num_embeddings)
        quantized = tf.matmul(encodings, self.embeddings, transpose_b=True)
        #quantized = tf.reshape(quantized, input_shape)

        # Calculate vector quantization loss and add that to the layer. You can learn more
        # about adding losses to different layers here:
        # https://keras.io/guides/making_new_layers_and_models_via_subclassing/. Check
        # the original paper to get a handle on the formulation of the loss function.
        commitment_loss = self.beta * tf.reduce_mean(
            (tf.stop_gradient(quantized) - x) ** 2
        )
        codebook_loss = tf.reduce_mean((quantized - tf.stop_gradient(x)) ** 2)
        self.add_loss(commitment_loss + codebook_loss)

        # Straight-through estimator.
        quantized = x + tf.stop_gradient(quantized - x)
        return quantized

    def get_code_indices(self, flattened_inputs):
        # Calculate L2-normalized distance between the inputs and the codes.
        similarity = tf.matmul(flattened_inputs, self.embeddings)
        distances = (
            tf.reduce_sum(flattened_inputs ** 2, axis=1, keepdims=True)
            + tf.reduce_sum(self.embeddings ** 2, axis=0)
            - 2 * similarity
        )

        # Derive the indices for minimum distances.
        encoding_indices = tf.argmin(distances, axis=1)
        return encoding_indices



class VQVAE(keras.models.Model):
    """ VQVAE """

    def __init__(
        self,
        original_dim,
        intermediate_dim=128,
        latent_dim=10,
        num_embeddings=7,
        name="VQ_VAE",
        **kwargs
    ):
        super(VQVAE, self).__init__(name=name, **kwargs)
        self.original_dim = original_dim
        self.encoder = Encoder(latent_dim, intermediate_dim)
        self.decoder = Decoder(original_dim, intermediate_dim)
        self.vq_layer = VectorQuantizer(16, 128, name="vector_quantizer")

    def call(self, inputs):
        encoder_outputs = self.encoder(inputs)
        quantized_latents = self.vq_layer(encoder_outputs)
        #print(tf.shape(quantized_latents))
        reconstructed = self.decoder(quantized_latents)
        return reconstructed



def train_vae(vae, data, batch_size=128, epochs=10):
    """ Training function

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
            with tf.GradientTape() as tape:
                reconstructed = vae(x_batch_train)
                # Compute reconstruction loss
                recon = - zinb_pos(x_batch_train, reconstructed)
                loss = recon + sum(vae.losses)

            grads = tape.gradient(loss, vae.trainable_weights)
            optimizer.apply_gradients(zip(grads, vae.trainable_weights))
            loss_metric(loss)
            if step % 100 == 0:
                tqdm_progress.set_postfix_str(
                    s="loss={:.2e}".format(
                        loss_metric.result()), refresh=True)

    return vae


from copyvae.preprocess import *

data_path = 'bined_expressed_cell.csv'
adata = load_data(data_path)
x_train = adata.X

d = '/device:GPU:0'
with tf.device(d):

    model = VQVAE(x_train.shape[-1], 128, 10, 20)
    copy_vae = train_vae(model, x_train, epochs = 400)
    z = copy_vae.encoder.predict(adata.X)
    c = copy_vae.vq_layer(z)