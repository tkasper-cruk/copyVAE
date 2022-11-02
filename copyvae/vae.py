#! /usr/bin/env python3

import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow.keras as keras
from copyvae.loss_func import nb_pos, zinb_pos
from copyvae.encoder import Encoder, CNEncoder
from copyvae.decoder import Decoder, CNDecoder


class VAE(keras.models.Model):
    """ SCVI """

    def __init__(
        self,
        original_dim,
        intermediate_dim=128,
        latent_dim=10,
        name="VAE",
        **kwargs
    ):
        super(VAE, self).__init__(name=name, **kwargs)
        self.original_dim = original_dim
        self.encoder = Encoder(latent_dim, intermediate_dim)
        self.decoder = Decoder(original_dim, intermediate_dim)
        self.loss_metric = keras.metrics.Mean()

    def call(self, inputs):
        z_mean, z_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)
        
        # Add KL divergence regularization loss.
        p_dis = tfp.distributions.Normal(loc=z_mean, scale=tf.math.sqrt(z_var))
        q_dis = tfp.distributions.Normal(
            loc=tf.zeros_like(z_mean),
            scale=tf.ones_like(z_var))
        kl_loss = tf.reduce_sum(
            tfp.distributions.kl_divergence(
                p_dis, q_dis), 1)
        self.add_loss(kl_loss)

        return reconstructed


    def train_step(self, data):

        with tf.GradientTape() as tape:
            reconstructed = self(data)
            # Compute reconstruction loss
            recon = - zinb_pos(data, reconstructed)
            loss = recon + self.losses

        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.loss_metric(loss)

        return {"loss": self.loss_metric.result()}


class CopyVAE(VAE):

    def __init__(
            self,
            original_dim,
            intermediate_dim=128,
            latent_dim=10,
            max_cp=25,
            name="CopyVAE",
            **kwargs):
        super().__init__(original_dim,
                         intermediate_dim,
                         latent_dim,
                         name)
        self.max_cp = max_cp

        self.z_encoder = Encoder(latent_dim, intermediate_dim)

        self.encoder = CNEncoder(original_dim, max_cp=max_cp)
        self.decoder = CNDecoder(original_dim, intermediate_dim)

    def call(self, inputs):
        inputs_en = inputs
        z_mean, z_var, z = self.z_encoder(inputs_en)
        cp = self.encoder(z)
        l = tf.expand_dims(
                            tf.math.log(
                                        tf.math.reduce_sum(inputs_en, axis=1)
                                    ), 
                        axis=1)
        reconstructed, copy = self.decoder([cp,z,l])

        # Add KL divergence regularization loss.
        p_dis = tfp.distributions.Normal(
            loc=z_mean,
            scale=tf.math.sqrt(z_var)
            )
        q_dis = tfp.distributions.Normal(
            loc=tf.zeros_like(z_mean),
            scale=tf.ones_like(z_var)
            )
        kl_loss = tf.reduce_sum(
                        tfp.distributions.kl_divergence(
                            p_dis, q_dis),
                    axis=1) * 0.5
        self.add_loss(kl_loss)

        return reconstructed

    def train_step(self, data):

        with tf.GradientTape() as tape:
            reconstructed = self(data)
            # Compute reconstruction loss
            recon = - nb_pos(data, reconstructed)
            loss = recon + self.losses

        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.loss_metric(loss)

        return {"loss": self.loss_metric.result()}