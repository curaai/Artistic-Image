import tensorflow as tf
import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import util
import loss
import vgg


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model_path', default='model/imagenet-vgg-verydeep-19.mat', help='insert vgg model path')
    parser.add_argument('--content', default='image/content.jpg', help='Insert Content Image_path')
    parser.add_argument('--style', default='image/style.jpg', help='Insert style Image_path')
    parser.add_argument('--image_width', type=int, default=800, help='image width')
    parser.add_argument('--image_height', type=int, default=600, help='image height')
    parser.add_argument('--save_model', default='save/model', help='Save Trained Model')

    parser.add_argument('--ALPHA', type=int, default=5, help='Used in train Content loss')
    parser.add_argument('--BETA', type=int, default=100, help='Used in train Style loss')
    parser.add_argument('--learning_rate', type=float, default=2.0, help='Learning rate ...')
    parser.add_argument('--iteration', type=int, default=1, help='Train iteration count')

    args = parser.parse_args()

    with tf.Graph().as_default(), tf.Session() as sess:
        width, height = args.image_width, args.image_height
        content_image = util.load_image(args.content, width, height)
        style_image = util.load_image(args.style, width, height)
        input_image = util.generate_noise_image(content_image, width, height)
        model = vgg.load_vgg_model(args.model_path, width, height)

        sess.run(tf.global_variables_initializer())

        x_content = model['conv4_2']
        x_style = [model['conv' + str(i) + '_1'] for i in range(1, 6)]

        sess.run(model['input'].assign(content_image))
        y_content = model['conv4_2']
        content_loss = loss.content_loss(x_content, y_content)

        sess.run(model['input'].assign(style_image))
        y_style = [model['conv' + str(i) + '_1'] for i in range(1, 6)]
        style_loss = loss.style_loss(x_style, y_style)

        total_loss = args.ALPHA * content_loss + args.BETA * style_loss

        optimizer = tf.train.AdamOptimizer(args.learning_rate).minimize(total_loss)

        saver = tf.train.Saver()

        # train
        print('Training Start !!!')
        sess.run(tf.global_variables_initializer())
        sess.run(model['input'].assign(input_image))
        for i in range(args.iteration):
            sess.run(optimizer)
            print("cost:", sess.run(total_loss))
            if i % 100 == 0:
                artistic_image = sess.run(model['input'])
                print("iteration:", str(i))
                print("cost:", sess.run(total_loss))
            if i % 3 == 0:
                print("cost:", sess.run(total_loss))

        # save image
        util.save_image('result.jpg', artistic_image)

        if args.save_model == 'save/model' and not os.path.isdir('save'):
            os.makedirs('save')            
        saver.save(sess, args.save_model)
