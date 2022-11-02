#! /usr/bin/env python3

import tensorflow as tf
from tensorflow.errors import InvalidArgumentError


def validate_params(mu, theta):
    """ Validate parameters for distribution """
    try:
        tf.debugging.assert_non_negative(mu)
    except InvalidArgumentError:
        print("Invalid mu")
        raise
    try:
        tf.debugging.assert_non_negative(theta)
    except InvalidArgumentError:
        print("Invalid theta")
        raise
    return True


def nb_pos(y_true, y_pred, eps=1e-8):
    """ Negative binomial reconstruction loss

    Args:
        y_true: true values
        y_pred: predicted values
        eps: numerical stability constant
    Parameters:
        x: Data
        mu: mean of the negative binomial (positive (batch x vars)
        theta: inverse dispersion parameter (positive) (batch x vars)
    Returns:
        loss (log likelihood scalar)
    """
    x = y_true
    mu = y_pred[0]
    theta = y_pred[1]

    arg_validated = validate_params(mu, theta)
    if not arg_validated:
        print("invalid arguments for negative binomial!")
        return None

    log_theta_mu_eps = tf.math.log(theta + mu + eps)

    res = (
        theta * (tf.math.log(theta + eps) - log_theta_mu_eps)
        + x * (tf.math.log(mu + eps) - log_theta_mu_eps)
        + tf.math.lgamma(x + theta)
        - tf.math.lgamma(theta)
        - tf.math.lgamma(x + 1)
    )

    return tf.math.reduce_sum(res, axis=-1)


def zinb_pos(y_true, y_pred, eps=1e-8):
    """ Zero-inflated negative binomial reconstruction loss

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
        loss (log likelihood scalar)
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