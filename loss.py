import tensorflow as tf


STYLE_WEIGHTS = [0.5, 1.0, 1.5, 3.0, 4.0]


def content_loss(image, content):
    return 1/2 * tf.reduce_sum(tf.pow(image - content, 2))


def style_loss(image_layers, style_layers):
    def _gram_matrix(F, N, M):
        matrix = tf.reshape(F, (M, N))
        return tf.matmul(tf.transpose(matrix), matrix)

    def _loss(a, x):
        N = int(a.shape[3])
        M = int(a.shape[1] * a.shape[2])

        A = _gram_matrix(a, N, M)
        X = _gram_matrix(x, N, M)
        return (1 / (4 * N ** 2 * M ** 2)) * tf.reduce_mean(tf.pow(A - X, 2))

    losses = []
    for image, style, weight in zip(image_layers, style_layers, STYLE_WEIGHTS):
        losses.append(_loss(image, style) * weight)
    return sum(losses)