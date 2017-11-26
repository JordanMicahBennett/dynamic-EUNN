import tensorflow as tf
import numpy as np
from tensorflow.python.ops import variable_scope as vs

def permute(x, hidden_size):
    x = tf.reshape(x, [-1, 2, hidden_size/2])
    x1, x2 = tf.unstack(x, axis=1)
    x = tf.stack([x2, x1], axis=1)
    x = tf.reshape(x, [-1, hidden_size])
    return x

def EUNN(inputs, hidden_size):
    theta_phi_initializer = tf.random_uniform_initializer(-np.pi, np.pi)
    thetaA = vs.get_variable("thetaA", [hidden_size/2], initializer=theta_phi_initializer)
    cos_thetaA = tf.cos(thetaA)
    sin_thetaA = tf.sin(thetaA)

    thetaB = vs.get_variable("thetaB", [hidden_size/2-1], initializer=theta_phi_initializer)
    cos_thetaB = tf.cos(thetaB)
    sin_thetaB = tf.sin(thetaB)

    diagA = tf.concat([cos_thetaA, cos_thetaA], axis=0)
    offA = tf.concat([-sin_thetaA, sin_thetaA], axis=0)

    diagB = tf.concat([[1], cos_thetaB, cos_thetaB, [1]], axis=0)
    offB = tf.concat([[0], -sin_thetaB, sin_thetaB, [0]], axis=0)

    out = diagA * inputs + permute(inputs, hidden_size) * offA
    out = diagB * out + permute(out, hidden_size) * offB

    return out

      