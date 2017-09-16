import tensorflow as tf
import numpy as np


STYLE_LAYERS = [
    ('conv1_1', 0.2),
    ('conv2_1', 0.2),
    ('conv3_1', 0.2),
    ('conv4_1', 0.2),
    ('conv5_1', 0.2)
]


def content_loss(sess, model):
    return 1 / 2 * tf.pow(sess.run(model['conv4_1']) - model['conv4_1'], 2)


def style_loss(sess, model):
    def _gram_matrix(tensor):
        filter = tensor.get_shape()[3]

        matrix = tf.reshape(tensor, shape=[-1, filter])
        return tf.matmul(tf.transpose(matrix), matrix)

    loss_style = list()
    for layer, ratio in STYLE_LAYERS:
        layer_tensor = model[layer]
        shape = layer_tensor.get_shape()
        N = shape[3]
        M = shape[1] * shape[2]

        loss = (1/(4 * N ** 2 * M ** 2)) * \
               tf.pow(_gram_matrix(sess.run(layer_tensor)) - _gram_matrix(layer_tensor), 2)
        loss_style.append(loss * ratio)

    return sum(loss_style)
