import tensorflow as tf
import numpy as np


STYLE_LAYERS = [
    ('conv1_1', 0.25),
    ('conv2_1', 0.25),
    ('conv3_1', 0.25),
    ('conv4_1', 0.25),
    ('conv5_1', 0.25)
]


def content_loss(sess, model):
    return 1 / 2 * tf.reduce_sum(tf.pow(sess.run(model['conv4_2']) - model['conv4_2'], 2))


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

    E = [_loss(sess.run(model[layer_name]), model[layer_name]) for layer_name, _ in STYLE_LAYERS]
    W = [w for _, w in STYLE_LAYERS]
    style = sum([W[l] * E[l] for l in range(len(STYLE_LAYERS))])
    return style
