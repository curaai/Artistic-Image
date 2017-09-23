import numpy as np
from scipy.misc import imread, imresize, imsave

# mean of content, style
MEAN = np.array([123.68, 116.779, 103.939])


# TODO GENERATE_NOISE_IMAGE는 복붙했음 일단 결과부터 보고 나중에 공부하자
def generate_noise_image(content_image, IMAGE_WIDTH, IMAGE_HEIGHT, noise_ratio = 0.6):
    """
    Returns a noise image intermixed with the content image at a certain ratio.
    """
    noise_image = np.random.uniform(
            -20, 20,
            (IMAGE_HEIGHT, IMAGE_WIDTH, 3)).astype('float32')
    # White noise image from the content representation. Take a weighted average
    # of the values
    input_image = noise_image * noise_ratio + content_image * (1 - noise_ratio)
    return input_image


def load_image(image, IMAGE_WIDTH, IMAGE_HEIGHT):
    image = imread(image)
    image = imresize(image, (IMAGE_HEIGHT, IMAGE_WIDTH))
    image = image.astype('float32')

    # image normalization (mean subtraction)
    image -= MEAN.reshape((1, 1, 3))

    return image


def save_image(path, image):
    image += MEAN.reshape((1, 1, 1, 3))

    image = image[0]
    image = np.clip(image, 0, 255).astype('uint8')

    imsave(path, image)

