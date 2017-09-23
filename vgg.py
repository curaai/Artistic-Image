import tensorflow as tf
import numpy as np
from scipy.io import loadmat


class Model:
    def __init__(self, model_path, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNEL=3):
        self.model_path = model_path
        self.IMAGE_WIDTH = IMAGE_WIDTH
        self.IMAGE_HEIGHT = IMAGE_HEIGHT
        self.IMAGE_CHANNEL = IMAGE_CHANNEL

    def build(self):
        vgg = loadmat(self.model_path)
        vgg_layers = vgg['layers']

        vgg_dict = {vgg_layers[0][i][0][0][0][0]: i for i in range(len(vgg_layers[0]))}

        # original vgg structure
        '''
        conv1_1 = 0
        relu1_1 = 1
        conv1_2 = 2
        relu1_2 = 3
        pool1   = 4
        conv2_1 = 5
        relu2_1 = 6
        conv2_2 = 7
        relu2_2 = 8
        pool2   = 9
        conv3_1 = 10
        relu3_1 = 11
        conv3_2 = 12
        relu3_2 = 13
        conv3_3 = 14
        relu3_3 = 15
        conv3_4 = 16
        relu3_4 = 17
        pool3   = 18
        conv4_1 = 19
        relu4_1 = 20
        conv4_2 = 21
        relu4_2 = 22
        conv4_3 = 23
        relu4_3 = 24
        conv4_4 = 25
        relu4_4 = 26
        pool4   = 27
        conv5_1 = 28
        relu5_1 = 29
        conv5_2 = 30
        relu5_2 = 31
        conv5_3 = 32
        relu5_3 = 33
        conv5_4 = 34
        relu5_4 = 35
        pool5   = 36
        fc6     = 37
        relu6   = 38
        fc7     = 39
        relu7   = 40
        fc8     = 41
        prob    = 42
        '''

        def _weights_and_bias(layer):
            W = vgg_layers[0][layer][0][0][2][0][0].astype('float32')
            b = vgg_layers[0][layer][0][0][2][0][1].astype('float32')
            return W, b

        def _relu(conv2d_layer):
            return tf.nn.relu(conv2d_layer)

        def _conv2d(prev_layer, layer, name):
            W, b = _weights_and_bias(layer)

            W = tf.get_variable(name + "_weight", W.shape, dtype='float32', initializer=tf.constant_initializer(W))
            b = tf.get_variable(name + "_bias", b.size, dtype='float32', initializer=tf.constant_initializer(b))
            return tf.nn.bias_add(tf.nn.conv2d(
                prev_layer, W, strides=[1, 1, 1, 1], padding='SAME'
                ), b)

        def _conv2d_relu(prev_layer, layer, name):
            return _relu(_conv2d(prev_layer, layer, name))

        def _avgpool(layer):
            return tf.nn.avg_pool(
                layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # each weight size (1, width, height, filter)
        graph = dict()
        graph['input'] = tf.placeholder(tf.float32, [None, self.IMAGE_HEIGHT, self.IMAGE_WIDTH, self.IMAGE_CHANNEL])
        with tf.variable_scope("vggnet"):
            with tf.name_scope("conv1"):
                graph['conv1_1']  = _conv2d_relu(graph['input'],    vgg_dict['conv1_1'], 'conv1_1')
                graph['conv1_2']  = _conv2d_relu(graph['conv1_1'],  vgg_dict['conv1_2'], 'conv1_2')
                graph['avgpool1'] = _avgpool(graph['conv1_2'])
            with tf.name_scope("conv2"):
                graph['conv2_1']  = _conv2d_relu(graph['avgpool1'], vgg_dict['conv2_1'], 'conv2_1')
                graph['conv2_2']  = _conv2d_relu(graph['conv2_1'],  vgg_dict['conv2_2'], 'conv2_2')
                graph['avgpool2'] = _avgpool(graph['conv2_2'])
            with tf.name_scope("conv3"):
                graph['conv3_1']  = _conv2d_relu(graph['avgpool2'], vgg_dict['conv3_1'], 'conv3_1')
                graph['conv3_2']  = _conv2d_relu(graph['conv3_1'],  vgg_dict['conv3_2'], 'conv3_2')
                graph['conv3_3']  = _conv2d_relu(graph['conv3_2'],  vgg_dict['conv3_3'], 'conv3_3')
                graph['conv3_4']  = _conv2d_relu(graph['conv3_3'],  vgg_dict['conv3_4'], 'conv3_4')
                graph['avgpool3'] = _avgpool(graph['conv3_4'])
            with tf.name_scope("conv4"):
                graph['conv4_1']  = _conv2d_relu(graph['avgpool3'], vgg_dict['conv4_1'], 'conv4_1')
                graph['conv4_2']  = _conv2d_relu(graph['conv4_1'],  vgg_dict['conv4_2'], 'conv4_2')
                graph['conv4_3']  = _conv2d_relu(graph['conv4_2'],  vgg_dict['conv4_3'], 'conv4_3')
                graph['conv4_4']  = _conv2d_relu(graph['conv4_3'],  vgg_dict['conv4_4'], 'conv4_4')
                graph['avgpool4'] = _avgpool(graph['conv4_4'])
            with tf.name_scope('conv5'):
                graph['conv5_1']  = _conv2d_relu(graph['avgpool4'], vgg_dict['conv5_1'], 'conv5_1')
                graph['conv5_2']  = _conv2d_relu(graph['conv5_1'],  vgg_dict['conv5_2'], 'conv5_2')
                graph['conv5_3']  = _conv2d_relu(graph['conv5_2'],  vgg_dict['conv5_3'], 'conv5_3')
                graph['conv5_4']  = _conv2d_relu(graph['conv5_3'],  vgg_dict['conv5_4'], 'conv5_4')
                graph['avgpool5'] = _avgpool(graph['conv5_4'])

        self.graph = graph
