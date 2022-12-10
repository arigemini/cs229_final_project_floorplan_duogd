import tensorflow as tf
from keras.models import load_model

from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input

import os
import time

from matplotlib import pyplot as plt
from IPython import display
import numpy as np

import scipy
from scipy import linalg
import cv2 as cv
import collections

import utils

BUFFER_SIZE = 400
BATCH_SIZE = 1
IMG_WIDTH = 256
IMG_HEIGHT = 256


def get_color_map():
    color_map = {'living_room': (255, 255, 27),
                 'bedroom': (255, 27, 255),
                 'kitchen': (255, 27, 27),
                 'bathroom': (27, 27, 255),
                 'balcony': (27, 255, 27),
                 'storage': (27, 255, 255)}
    return color_map


def load(image_file):
    input_image, _, real_image = utils.load(image_file)
    return input_image, real_image


def resize(input_image, real_image, height, width):
    input_image = tf.image.resize(input_image, [height, width],
                                  method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    real_image = tf.image.resize(real_image, [height, width],
                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    return input_image, real_image


def random_crop(input_image, real_image):
    stacked_image = tf.stack([input_image, real_image], axis=0)
    cropped_image = tf.image.random_crop(
        stacked_image, size=[2, IMG_HEIGHT, IMG_WIDTH, 3])

    return cropped_image[0], cropped_image[1]


# normalizing the images to [-1, 1]

def normalize(input_image, real_image):
    input_image = (input_image / 127.5) - 1
    real_image = (real_image / 127.5) - 1

    return input_image, real_image


def load_image_test(image_file):
    input_image, real_image = load(image_file)
    input_image, real_image = resize(input_image, real_image, IMG_HEIGHT, IMG_WIDTH)
    input_image, real_image = normalize(input_image, real_image)

    return input_image, real_image


def generate_images(model1, model2, test_input, tar, epoch, save_path):
    if model2 is not None:
        prediction1 = model1(test_input, training=True)
        prediction2 = model2(prediction1, training=True)
    else:
        prediction2 = model1(test_input, training=True)
    plt.figure(figsize=(15, 15))

    display_list = [test_input[0], tar[0], prediction2[0]]
    title = ['Input Image', 'Ground Truth', 'Predicted Image']

    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.title(title[i])
        # getting the pixel values between [0, 1] to plot it.
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
    # plt.show()
    plt.savefig(save_path + 'example' + str(epoch))
    plt.close()
    return tar[0], prediction2[0]


def calculate_ratios(image):
    color_map = get_color_map()
    # program_sum = {'living_room': 0,
    #              'bedroom': 0,
    #              'kitchen': 0,
    #              'bathroom': 0,
    #              'balcony': 0,
    #              'storage': 0}
    living_room_val = color_map['living_room']
    bedroom_val = color_map['bedroom']
    bathroom_val = color_map['bathroom']
    balcony_val = color_map['balcony']
    kitchen_val = color_map['kitchen']
    storage_val = color_map['storage']

    # img = cv.imread(image)  # you can read in images with opencv
    img = np.array(image)
    # img_hsv = cv.cvtColor(img)
    # attempy to speed up
    living_room_range_lo = np.asarray([living_room_val[0] - 10, living_room_val[1] - 10, living_room_val[2] - 10])
    living_room_range_hi = np.asarray([living_room_val[0] + 10, living_room_val[1] + 10, living_room_val[2] + 10])
    bedroom_range_lo = np.asarray([bedroom_val[0] - 10, bedroom_val[1] - 10, bedroom_val[2] - 10])
    bedroom_range_hi = np.asarray([bedroom_val[0] + 10, bedroom_val[1] + 10, bedroom_val[2] + 10])
    bathroom_range_lo = np.asarray([bathroom_val[0] - 10, bathroom_val[1] - 10, bathroom_val[2] - 10])
    bathroom_range_hi = np.asarray([bathroom_val[0] + 10, bathroom_val[1] + 10, bathroom_val[2] + 10])
    balcony_range_lo = np.asarray([balcony_val[0] - 10, balcony_val[1] - 10, balcony_val[2] - 10])
    balcony_range_hi = np.asarray([balcony_val[0] + 10, balcony_val[1] + 10, balcony_val[2] + 10])
    kitchen_range_lo = np.asarray([kitchen_val[0] - 10, kitchen_val[1] - 10, kitchen_val[2] - 10])
    kitchen_range_hi = np.asarray([kitchen_val[0] + 10, kitchen_val[1] + 10, kitchen_val[2] + 10])
    storage_range_lo = np.asarray([storage_val[0] - 10, storage_val[1] - 10, storage_val[2] - 10])
    storage_range_hi = np.asarray([storage_val[0] + 10, storage_val[1] + 10, storage_val[2] + 10])
    living_room_mask = cv.inRange(img, living_room_range_lo, living_room_range_hi)
    bedroom_mask = cv.inRange(img, bedroom_range_lo, bedroom_range_hi)
    bathroom_mask = cv.inRange(img, bathroom_range_lo, bathroom_range_hi)
    balcony_mask = cv.inRange(img, balcony_range_lo, balcony_range_hi)
    kitchen_mask = cv.inRange(img, kitchen_range_lo, kitchen_range_hi)
    storage_mask = cv.inRange(img, storage_range_lo, storage_range_hi)
    program_sum = {'living_room': np.count_nonzero(living_room_mask),
                   'bedroom': np.count_nonzero(bedroom_mask),
                   'kitchen': np.count_nonzero(kitchen_mask),
                   'bathroom': np.count_nonzero(bathroom_mask),
                   'balcony': np.count_nonzero(balcony_mask),
                   'storage': np.count_nonzero(storage_mask)}
    for key in program_sum.keys():
        if program_sum[key] < 10: program_sum[key] = 0
    # room_ratios = collections.defaultdict(float)
    #
    #
    # for key in program_sum.keys():
    #     for yek in program_sum.keys():
    #         if key != yek:
    #             if program_sum[key] != 0 and program_sum[yek] != 0:
    #                 room_ratios[str(key) + '/' + str(yek)] = program_sum[key] / program_sum[yek]
    #             else:
    #                 room_ratios[str(key) + '/' + str(yek)] = 0
    return program_sum


def room_ratio_accuracy(target_image, gen_image):
    # plt.figure(figsize=(15, 15))

    target = tf.keras.utils.array_to_img(target_image[0])
    plt.figure(figsize=(15, 15))
    plt.imshow(target)
    plt.close()

    gen = tf.keras.utils.array_to_img(gen_image[0])
    target_sum = calculate_ratios(target)
    gen_sum = calculate_ratios(gen)
    plt.close()
    ratio = []
    for key in target_sum.keys():
        weight = target_sum[key]
        if target_sum[key] != 0 and target_sum[key] > gen_sum[key]:
            ratio.append(weight * abs(gen_sum[key] / target_sum[key]))
        if gen_sum[key] != 0 and gen_sum[key] > target_sum[key]:
            ratio.append(weight * abs(target_sum[key] / gen_sum[key]))
    ratio_sum = sum(ratio)
    for i in range(len(ratio)):
        ratio[i] = ratio[i] / ratio_sum

    ave_accuracy = sum(ratio) / len(ratio)
    return ave_accuracy


def loss_assess(real_image, generated_image):
    perpixel_accuracy = 1 - tf.reduce_mean(tf.abs(real_image - generated_image))
    ratio_accuracy = room_ratio_accuracy(real_image, generated_image)
    return perpixel_accuracy, ratio_accuracy


# frechet inception distance
def calculate_fid(images1, images2):
    images1 = tf.stack(images1)
    images2 = tf.stack(images2)

    assert images1.shape == images2.shape

    model = InceptionV3(include_top=False, pooling='avg', input_shape=images1.shape[1:])

    images1 = preprocess_input(images1)
    images2 = preprocess_input(images2)

    act1 = model.predict(images1)
    act2 = model.predict(images2)

    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)

    diff = np.sum((mu1 - mu2) ** 2.0)

    covmean = scipy.linalg.sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    return diff + np.trace(sigma1 + sigma2 - 2.0 * covmean)


def main():
    # loading test files (we're testing loss right now, if we're not testing loss we
    # can switch to another dataset without desired output)
    test_data_dir = '2bedroom_data'
    test_dataset = tf.data.Dataset.list_files(test_data_dir + '/TEST/*.png')
    test_dataset = test_dataset.map(load_image_test)
    test_dataset = test_dataset.batch(BATCH_SIZE)

    # duo gen disc model
    saved_model1 = load_model('improved_model/1generator_model.h5')
    saved_model2 = load_model('improved_model/2generator_model.h5')
    save_path = 'improved_model_predictions/predicted'
    accuracy_file = 'duogendisc_model_accuracy_quantified.txt'

    # baseline pix2pix
    # saved_model1 = load_model('baseline_generator_model.h5')
    # saved_model2 = None
    # save_path = 'baseline_model_predictions/predicted'
    # accuracy_file = 'baseline_model_accuracy_quantified.txt'

    i = 0
    perpix = 0
    ratio = 0
    real_images = []
    fake_images = []
    for example_input, example_target in test_dataset:
        real, fake = generate_images(saved_model1, saved_model2, example_input, example_target, i, save_path)
        real_images.append(real)
        fake_images.append(fake)

        real = np.expand_dims(real, axis=0)
        fake = np.expand_dims(fake, axis=0)
        perpixel, room_ratio = loss_assess(real, fake)
        perpix += perpixel
        ratio += room_ratio

        # after reading checkpoint, change back to just epoch later
        i += 1
        print("finished example", str(i))
        # if i == 10:
        #     break

    ave_perpix = perpix / i
    ave_ratio = ratio / i

    # calculate FID
    print('calculating FID')
    fid = calculate_fid(real_images, fake_images)
    print(ave_perpix, ave_ratio, fid)

    with open(accuracy_file, 'w') as f:
        f.write(f'acc_perpix accuracy: {ave_perpix} ave_ratio accuracy: {ave_ratio} fid: {fid}\n')

    # f.writelines([acc_perpix, acc_ratio])
    return


if __name__ == "__main__":
    main()
