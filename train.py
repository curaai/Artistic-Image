import tensorflow as tf
import util
import numpy as np
import loss
import vgg
import os


if __name__ == '__main__':
    model_path = 'model/imagenet-vgg-verydeep-19.mat'
    content_path = "image/content.jpg"
    style_path = "image/style.jpg"
    output_path = "output/"

    ALPHA = 5
    BETA = 100

    learning_rate = 0.0001
    ITERATION = 1000

    with tf.InteractiveSession() as sess:
        content_image = util.load_image(content_path)
        style_image = util.load_image(style_path)
        input_image = util.generate_noise_image(content_image)
        model = vgg.load_vgg_model(model_path)

        sess.run(tf.initialize_all_variables())

        sess.run(model['input'].assign(content_image))
        content_loss = loss.content_loss(sess, model)
        sess.run(model['input'].assign(style_image))
        style_loss = loss.style_loss(sess, model)
        total_loss = ALPHA * content_loss + BETA * style_loss

        saver = tf.train.Saver()

        optimizer = tf.train.AdamOptimizer(2.0).minimize(total_loss)

        # train
        print('Training Start !!!')
        sess.run(model['input'].assign(input_image))
        for i in range(ITERATION):
            sess.run(optimizer)
            if i % 100 == 0:
                artistic_image = sess.run(model['input'])
                print("iteration:", str(i))
                print("cost:", sess.run(total_loss))

                if not os.path.exists(output_path):
                    os.mkdir(output_path)

                # save image
                util.save_image(output_path + str(i) + '.jpg', artistic_image)

        saver.save(sess, "model.ckpt")
