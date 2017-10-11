import tensorflow as tf
import os
import argparse

import vgg
import util


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--image_path', default='image/test.jpg', help='Image path to describe also possible image dir')
    parser.add_argument('--output_path', default='output/', help='Described image out')
    parser.add_argument('--image_width', type=int, default=800, help='image width')
    parser.add_argument('--image_height', type=int, default=600, help='image height')
    parser.add_argument('--model_path', default='model/imagenet-vgg-verydeep-19.mat', help='Trained model path')
    parser.add_argument('--save_path', default='save/test.ckpt', help='Load saver file')

    args = parser.parse_args()

    with tf.Session() as sess:
        pass