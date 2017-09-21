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
        model = vgg.load_vgg_model(args.model_path, args.image_width, args.image_height)
        before = model

        saver = tf.train.Saver()
        saver.restore(sess, args.save_path)

        if before == model:
            print(" didnt load well")

        image_list = []
        if os.path.isdir(args.image_path):
            image_list = [args.image_path + path for path in os.listdir(args.image_path)]
        else:
            image_list = [args.image_path]

        # make output dir if not exist
        if not os.path.isdir(args.output_path):
            os.makedirs(args.output_path)

        for file in image_list:
            sess.run(tf.global_variables_initializer())

            input_image = util.load_image(file, args.image_width, args.image_height)
            sess.run(model['input'].assign(input_image))

            # generate artistic image
            generated = sess.run(model['input'])
            util.save_image(args.output_path + "artistic_" + os.path.basename(file), generated)
