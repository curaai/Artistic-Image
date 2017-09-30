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
    parser.add_argument('--BETA', type=int, default=1000, help='Used in train Style loss')
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

        pred_image = tf.Variable(image_input)
        style_image = tf.constant(image_style)
        content_image = tf.constant(image_content)

        vgg_net = vgg.Model(model_path, width, height)
        pred_net    = vgg_net.build(pred_image, 'pred', 0)
        style_net   = vgg_net.build(style_image, 'style', 1)
        content_net = vgg_net.build(content_image, 'content', 1)


        x_content = pred_net['conv4_2']
        x_style = [pred_net['conv' + str(i) + '_1'] for i in range(1, 6)]

        y_content = content_net['conv4_2']
        content_loss = loss.content_loss(x_content, y_content)

        y_style = [style_net['conv' + str(i) + '_1'] for i in range(1, 6)]
        style_loss = loss.style_loss(x_style, y_style)

        total_loss = args.ALPHA * content_loss + args.BETA * style_loss

        default_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        vgg_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='vggnet')

        optimizer = tf.train.AdamOptimizer(args.learning_rate).minimize(loss=total_loss, var_list=default_vars + vgg_vars)

        saver = tf.train.Saver()

        # train
        print('Training Start !!!')
        sess.run(tf.global_variables_initializer())
        for i in range(args.iteration):
            _, cost, artistic_image = sess.run([optimizer, total_loss, pred_image])
            print("cost:", cost)
            util.save_image(str(i) + '.jpg', artistic_image)

            if i % 50 == 0:
                artistic_image = sess.run(pred_image)
                print("iteration:", str(i))

                # save image

        if args.save_model == 'save/model' and not os.path.isdir('save'):
            os.makedirs('save')            
        saver.save(sess, args.save_model)
