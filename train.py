import tensorflow as tf
import argparse
import os

import util
import loss
import vgg


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model_path', default='model/imagenet-vgg-verydeep-19.mat', help='insert vgg model path')
    parser.add_argument('--content', help='Insert Content Image_path')
    parser.add_argument('--style', help='Insert style Image_path')
    parser.add_argument('--image_width', type=int, help='image width')
    parser.add_argument('--image_height', type=int, help='image height')
    parser.add_argument('--output', help='Train result output')
    parser.add_argument('--save_model', help='Save Trained Model')

    parser.add_argument('--ALPHA', type=int, default=5, help='Used in train Content loss')
    parser.add_argument('--BETA', type=int, default=100, help='Used in train Style loss')
    parser.add_argument('--learning_rate', type=float, default=2.0, help='Learning rate ...')
    parser.add_argument('--iteration', type=int, default=1000, help='Train iteration count')

    args = parser.parse_args()

    with tf.Session() as sess:
        width, height = args.image_width, args.image_heigth
        content_image = util.load_image(args.content, width, height)
        style_image = util.load_image(args.style, width, height)
        input_image = util.generate_noise_image(content_image, width, height)
        model = vgg.load_vgg_model(args.model_path, width, height)

        sess.run(tf.global_variables_initializer())

        sess.run(model['input'].assign(content_image))
        content_loss = loss.content_loss(sess, model)
        sess.run(model['input'].assign(style_image))
        style_loss = loss.style_loss(sess, model)
        total_loss = args.alpha * content_loss + args.beta * style_loss

        saver = tf.train.Saver()

        optimizer = tf.train.AdamOptimizer(args.learning_rate).minimize(total_loss)

        # train
        print('Training Start !!!')
        sess.run(tf.global_variables_initializer())
        sess.run(model['input'].assign(input_image))
        for i in range(args.iteration):
            sess.run(optimizer)
            if i % 100 == 0:
                artistic_image = sess.run(model['input'])
                print("iteration:", str(i))
                print("cost:", sess.run(total_loss))

        # save image
        util.save_image('result.jpg', artistic_image)

        if not os.path.isdir(args.save_model):
            os.makedirs(args.save_model)
        saver.save(sess, args.save_model)
