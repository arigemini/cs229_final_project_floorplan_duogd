import numpy as np
import tensorflow as tf
from PIL import Image


def load(image_file):
    image = tf.io.read_file(image_file)
    image = tf.image.decode_jpeg(image)

    w = tf.shape(image)[1]
    w = w // 3

    input_image = image[:, :w, :]
    intermediate_image = image[:, w:(2 * w), :]
    real_image = image[:, (2 * w):, :]

    input_image = tf.cast(input_image, tf.float32)
    intermediate_image = tf.cast(intermediate_image, tf.float32)
    real_image = tf.cast(real_image, tf.float32)

    return input_image, intermediate_image, real_image


def resize(input_image, intermediate_image, real_image, height, width):
    return [
        tf.image.resize(image, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        for image in [input_image, intermediate_image, real_image]
    ]


def tensor_to_image(tensor):
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return Image.fromarray(tensor)


def random_crop(input_image, intermediate_image, real_image, height, width):
    stacked_image = tf.stack([input_image, intermediate_image, real_image], axis=0)
    cropped_image = tf.image.random_crop(stacked_image, size=[3, height, width, 3])
    return cropped_image[0], cropped_image[1], cropped_image[2]


def normalize(input_image, intermediate_image, real_image):
    input_image = (input_image / 127.5) - 1
    intermediate_image = (intermediate_image / 127.5) - 1
    real_image = (real_image / 127.5) - 1
    return input_image, intermediate_image, real_image


@tf.function()
def random_jitter(input_image, intermediate_image, real_image, h1, w1, h2, w2):
    input_image, intermediate_image, real_image = resize(input_image, intermediate_image, real_image, h1, w1)
    input_image, intermediate_image, real_image = random_crop(input_image, intermediate_image, real_image, h2, w2)

    if tf.random.uniform(()) > 0.5:
        # random mirroring
        input_image = tf.image.flip_left_right(input_image)
        intermediate_image = tf.image.flip_left_right(intermediate_image)
        real_image = tf.image.flip_left_right(real_image)

    return input_image, intermediate_image, real_image
