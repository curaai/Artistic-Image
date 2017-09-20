import tensorflow as tf
import numpy as np


STYLE_LAYERS = [
    ('conv1_1', 0.5),
    ('conv2_1', 1.0),
    ('conv3_1', 1.5),
    ('conv4_1', 3.0),
    ('conv5_1', 4.0)
]


def content_loss(sess, model):
    shape = sess.run(model['conv4_2']).shape
    N = shape[3]
    M = shape[1] * shape[2]
    return (1 / (4 * N * M)) * tf.reduce_sum(tf.pow(sess.run(model['conv4_2']) - model['conv4_2'], 2))


def style_loss(sess, model):
    def _gram_matrix(F, N, M):
        matrix = tf.reshape(F, (M, N))
        return tf.matmul(tf.transpose(matrix), matrix)

    def _loss(a, x):
        N = a.shape[3]
        M = a.shape[1] * a.shape[2]

        A = _gram_matrix(a, N, M)
        X = _gram_matrix(x, N, M)
        return (1 / (4 * N ** 2 * M ** 2)) * tf.reduce_mean(tf.pow(A - X, 2))

    style = sum([_loss(sess.run(model[layer_name]), model[layer_name]) * w for layer_name, w in STYLE_LAYERS])
    return style
