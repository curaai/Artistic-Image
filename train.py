import tensorflow as tf
import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import util
import loss
import vgg
import numpy


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model_path', default='model/imagenet-vgg-verydeep-19.mat', help='insert vgg model path')
    parser.add_argument('--content', default='image/content.jpg', help='Insert Content Image_path')
    parser.add_argument('--style', default='image/style.jpg', help='Insert style Image_path')
    parser.add_argument('--image_width', type=int, default=800, help='image width')
    parser.add_argument('--image_height', type=int, default=600, help='image height')
    parser.add_argument('--save_model', default='save/model', help='Save Trained Model')

    parser.add_argument('--ALPHA', type=int, default=1, help='Used in train Content loss')
    parser.add_argument('--BETA', type=int, default=100, help='Used in train Style loss')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate ...')
    parser.add_argument('--iteration', type=int, default=1000, help='Train iteration count')

    args = parser.parse_args()

    with tf.Graph().as_default(), tf.Session() as sess:
        width, height = args.image_width, args.image_height
        model_path = args.model_path

        image_content = util.load_image(args.content, width, height)
        image_style = util.load_image(args.style, width, height)
        image_input = util.generate_noise_image(image_content, width, height)

        sess.run(tf.global_variables_initializer())

        vgg_net = vgg.Model(model_path, width, height, args.ALPHA, args.BETA, args.learning_rate)
        vgg_net.build(sess, image_input, image_content, image_style)
        
        # train
        print('Training Start !!!')
        sess.run(tf.global_variables_initializer())
        for i in range(args.iteration):
            vgg_net.pre_train()
            _, artistic_image, loss = vgg_net.train()
            
            if i % 50 == 0:
                print("cost:", cost)
                util.save_image(str(i) + '.jpg', artistic_image)
                print("iteration:", str(i))

        if args.save_model == 'save/model' and not os.path.isdir('save'):
            os.makedirs('save')            
        saver.save(sess, args.save_model)
