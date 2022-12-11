import os
import time
import collections
import datetime

from IPython import display

import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import cv2 as cv
import tqdm

from tensorflow.python.keras.callbacks import TensorBoard

from pix2pix_discriminator import Discriminator
from pix2pix_generator import upsample, downsample, Generator
from utils import load, resize, normalize, random_jitter


print(tf.config.list_physical_devices())

PATH = '2bedroom_data'

SAMPLE_PATH = os.path.join(PATH, 'TEST', '3.png')

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
    # for x in range(image.width):
    #     for y in range(image.height):
    #         cur_pixel = image.getpixel((x, y))
    #         r, g, b = cur_pixel[0], cur_pixel[1], cur_pixel[2]
    #         label=None
    #         if abs(r - living_room_val[0]) < 10 and abs(g - living_room_val[1]) < 10 and abs(b - living_room_val[2]) < 10 :
    #             label = 'living_room'
    #         if abs(r - bedroom_val[0]) < 10 and abs(g - bedroom_val[1]) < 10 and abs(b - bedroom_val[2]) < 10 :
    #             label = 'bedroom'
    #         if abs(r - bathroom_val[0]) < 10 and abs(g - bathroom_val[1]) < 10 and abs(b - bathroom_val[2]) < 10 :
    #             label = 'bathroom'
    #         if abs(r - kitchen_val[0]) < 10 and abs(g - kitchen_val[1]) < 10 and abs(b - kitchen_val[2]) < 10 :
    #             label = 'kitchen'
    #         if abs(r - balcony_val[0]) < 10 and abs(g - balcony_val[1]) < 10 and abs(b - balcony_val[2]) < 10 :
    #             label = 'balcony'
    #         if abs(r - storage_val[0]) < 10 and abs(g - storage_val[1]) < 10 and abs(b - storage_val[2]) < 10 :
    #             label = 'storage'
    #         if label != None:
    #             program_sum[label] += 1
    #

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
    room_ratios = collections.defaultdict(float)

    for key in program_sum.keys():
        for yek in program_sum.keys():
            if key != yek:
                if program_sum[key] != 0 and program_sum[yek] != 0:
                    room_ratios[str(key) + '/' + str(yek)] = program_sum[key] / program_sum[yek]
                else:
                    room_ratios[str(key) + '/' + str(yek)] = 0
    return room_ratios


def get_sample():
    return load(SAMPLE_PATH)


def show_sample():
    inp, intm, re = get_sample()
    plt.figure()
    plt.imshow(inp / 255.0)
    plt.figure()
    plt.imshow(intm / 255.0)
    plt.figure()
    plt.imshow(re / 255.0)


show_sample()


def show_jitter():
    inp, intm, re = get_sample()

    plt.figure(figsize=(6, 6))
    for i in range(4):
        rj_inp, rj_intm, rj_re = random_jitter(inp, intm, re, 286, 286, IMG_HEIGHT, IMG_WIDTH)
        plt.subplot(2, 2, i + 1)
        plt.imshow(rj_re / 255.0)
        plt.axis('off')
    plt.show()


show_jitter()


def load_image_train(image_file):
    input_image, intermediate_image, real_image = load(image_file)
    input_image, intermediate_image, real_image = random_jitter(
        input_image, intermediate_image, real_image, 286, 286, IMG_HEIGHT, IMG_WIDTH)
    input_image, intermediate_image, real_image = normalize(input_image, intermediate_image, real_image)
    return input_image, intermediate_image, real_image


def load_image_test(image_file):
    input_image, intermediate_image, real_image = load(image_file)
    input_image, intermediate_image, real_image = resize(input_image, intermediate_image, real_image, IMG_HEIGHT,
                                                         IMG_WIDTH)
    input_image, intermediate_image, real_image = normalize(input_image, intermediate_image, real_image)
    return input_image, intermediate_image, real_image


train_dataset = tf.data.Dataset.list_files(PATH + '/TRAIN/*.png')
train_dataset = train_dataset.map(load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.batch(BATCH_SIZE)

test_dataset = tf.data.Dataset.list_files(PATH + '/TEST/*.png')
test_dataset = test_dataset.map(load_image_test)
test_dataset = test_dataset.batch(BATCH_SIZE)


def show_testdataset():
    for inp, _, _ in test_dataset.take(1):
        print(np.asarray(inp))


show_testdataset()

"""
    Generator: modified U-Net.
        Encoder: Conv -> Batchnorm -> Leaky ReLU
        Decoder: Transposed Conv -> Batchnorm -> Dropout(applied to the first 3 blocks) -> ReLU
        Skip connections between the encoder and decoder.
"""


def show_down_and_up_sample():
    inp, _, _ = get_sample()
    down_model = downsample(3, 4)
    down_result = down_model(tf.expand_dims(inp, 0))
    print(down_result.shape)

    up_model = upsample(3, 4)
    up_result = up_model(down_result)
    print(up_result.shape)


show_down_and_up_sample()

first_generator = Generator()
second_generator = Generator()
# tf.keras.utils.plot_model(generator, show_shapes=True, dpi=64)

"""
    Generator loss
    sigmoid cross entropy loss of generated images and an **array of ones**.
    The [paper](https://arxiv.org/abs/1611.07004) also includes L1 loss which is MAE (mean absolute error) between the generated image and the target image.
    This allows the generated image to become structurally similar to the target image.
    total generator loss = gan_loss + LAMBDA * l1_loss, where LAMBDA = 100. This value was decided by the authors of the [paper](https://arxiv.org/abs/1611.07004).
    We have included a room_ratio_loss to tailor the model to our application. We also set LAMBDA=50 as that improved metrics.
"""

LAMBDA = 100
BETA = 1000

def room_ratio_loss(target_image, gen_image):
    # plt.figure(figsize=(15, 15))

    target = tf.keras.utils.array_to_img(target_image[0])
    plt.figure(figsize=(15, 15))
    plt.imshow(target)
    plt.close()

    gen = tf.keras.utils.array_to_img(gen_image[0])
    target_ratio = calculate_ratios(target)
    gen_ratio = calculate_ratios(gen)
    plt.close()
    difference = []
    for key in target_ratio.keys():
        difference.append(abs(target_ratio[key] - gen_ratio[key]))
    return tf.convert_to_tensor(difference)


def generator_loss(disc_generated_output, gen_output, target, second_gen_bool):
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

    if second_gen_bool:
        ratio_loss = tf.reduce_mean(room_ratio_loss(target, gen_output))

        total_gen_loss = gan_loss + (LAMBDA * l1_loss) + (BETA * ratio_loss)

        return total_gen_loss, gan_loss, l1_loss, ratio_loss

    total_gen_loss = gan_loss + (LAMBDA * l1_loss)
    return total_gen_loss, gan_loss, l1_loss, None


first_discriminator = Discriminator()
second_discriminator = Discriminator()
tf.keras.utils.plot_model(first_discriminator, show_shapes=True, dpi=64)


def show_first_disc_output():
    inp, _, _ = get_sample()
    first_gen_output = first_generator(inp[tf.newaxis, ...], training=False)
    plt.imshow(first_gen_output[0, ...])

    disc_out = first_discriminator([inp[tf.newaxis, ...], first_gen_output], training=False)
    plt.imshow(disc_out[0, ..., -1], vmin=-20, vmax=20, cmap='RdBu_r')
    plt.colorbar()


show_first_disc_output()

loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
    return real_loss + generated_loss


# Optimizers

first_generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
first_discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
second_generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
second_discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

training_outputdir = 'dual_gendisc_training_output'

checkpoint_dir = f'./{training_outputdir}/training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(first_generator_optimizer=first_generator_optimizer,
                                 first_discriminator_optimizer=first_discriminator_optimizer,
                                 second_generator_optimizer=second_generator_optimizer,
                                 second_discriminator_optimizer=second_discriminator_optimizer,
                                 first_generator=first_generator,
                                 second_generator=second_generator,
                                 first_discriminator=first_discriminator,
                                 second_discriminator=second_discriminator)


def generate_images(first_model, second_model, test_input, example_itm_target, tar, epoch):
    first_prediction = first_model(test_input, training=True)
    second_prediction = second_model(first_prediction, training=True)
    plt.figure(figsize=(15, 15))

    display_list = [test_input[0], tar[0], first_prediction[0], second_prediction[0]]
    title = ['Input Image', 'Ground Truth', 'First Predicted Image', 'Final Predicted Image']

    for i in range(4):
        plt.subplot(1, 4, i + 1)
        plt.title(title[i])
        # getting the pixel values between [0, 1] to plot it.
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
    # plt.show()
    os.makedirs(f'{training_outputdir}/iteration_images/', exist_ok=True)
    plt.savefig(f'{training_outputdir}/iteration_images/dual-gendisc model iteration' + str(epoch))


def show_images():
    for example_input, example_itm_target, example_target in test_dataset.take(3):
        generate_images(first_generator,
                        second_generator,
                        example_input,
                        example_itm_target,
                        example_target,
                        epoch=1)


show_images()

""" Training """

EPOCHS = 400

log_dir = f"{training_outputdir}/logs/"

summary_writer = tf.summary.create_file_writer(log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))


# @tf.function
def train_step(input_image, intermediate_target, target, epoch, cur_ratioloss):
    # first step to training: footprint -> wall boundaries
    with tf.GradientTape() as first_gen_tape, tf.GradientTape() as first_disc_tape:
        first_gen_output = first_generator(input_image, training=True)

        first_disc_real_output = first_discriminator([input_image, intermediate_target], training=True)
        first_disc_generated_output = first_discriminator([input_image, first_gen_output], training=True)

        first_gen_total_loss, first_gen_gan_loss, first_gen_l1_loss, first_gen_ratio_loss = generator_loss(
            first_disc_generated_output, first_gen_output, intermediate_target, False)
        first_gen_total_loss += cur_ratioloss
        first_disc_loss = discriminator_loss(first_disc_real_output, first_disc_generated_output)

    # acquire generated floor plan
    intermediate_generated_output = first_generator(input_image, training=True)

    # first step to training: wall boundaries -> programs
    with tf.GradientTape() as second_gen_tape, tf.GradientTape() as second_disc_tape:
        second_gen_output = second_generator(intermediate_generated_output, training=True)

        second_disc_real_output = second_discriminator([intermediate_generated_output, target], training=True)
        second_disc_generated_output = second_discriminator([intermediate_generated_output, second_gen_output],
                                                            training=True)

        second_gen_total_loss, second_gen_gan_loss, second_gen_l1_loss, second_gen_ratio_loss = generator_loss(
            second_disc_generated_output, second_gen_output, target, True)
        second_disc_loss = discriminator_loss(second_disc_real_output, second_disc_generated_output)

    # first_gen_total_loss = sum([first_gen_total_loss, second_gen_total_loss])

    first_generator_gradients = first_gen_tape.gradient(first_gen_total_loss,
                                                        first_generator.trainable_variables)
    first_discriminator_gradients = first_disc_tape.gradient(first_disc_loss,
                                                             first_discriminator.trainable_variables)
    second_generator_gradients = second_gen_tape.gradient(second_gen_total_loss,
                                                          second_generator.trainable_variables)
    second_discriminator_gradients = second_disc_tape.gradient(second_disc_loss,
                                                               second_discriminator.trainable_variables)

    first_generator_optimizer.apply_gradients(zip(first_generator_gradients,
                                                  first_generator.trainable_variables))
    first_discriminator_optimizer.apply_gradients(zip(first_discriminator_gradients,
                                                      first_discriminator.trainable_variables))

    second_generator_optimizer.apply_gradients(zip(second_generator_gradients,
                                                   second_generator.trainable_variables))
    second_discriminator_optimizer.apply_gradients(zip(second_discriminator_gradients,
                                                       second_discriminator.trainable_variables))
    # print('epoch'+ str(epoch) + 'second_gen_l1_loss: ' + str(second_gen_l1_loss))
    # print('epoch'+ str(epoch) + 'second_gen_total_loss: ' + str(second_gen_total_loss))
    with summary_writer.as_default():
        tf.summary.scalar('first_gen_total_loss', first_gen_total_loss, step=epoch)
        tf.summary.scalar('first_gen_gan_loss', first_gen_gan_loss, step=epoch)
        tf.summary.scalar('first_gen_l1_loss', first_gen_l1_loss, step=epoch)
        # tf.summary.scalar('first_gen_ratioloss', first_gen_ratio_loss, step=epoch)
        tf.summary.scalar('first_disc_loss', first_disc_loss, step=epoch)
        tf.summary.scalar('second_gen_total_loss', second_gen_total_loss, step=epoch)
        tf.summary.scalar('second_gen_gan_loss', second_gen_gan_loss, step=epoch)
        tf.summary.scalar('second_gen_l1_loss', second_gen_l1_loss, step=epoch)
        tf.summary.scalar('first_gen_ratioloss', second_gen_ratio_loss, step=epoch)
        tf.summary.scalar('second_disc_loss', second_disc_loss, step=epoch)
    return second_gen_ratio_loss


def fit(train_ds, epochs, test_ds, resumecheck=0):  # changed a bit, since the dataset is too big, we're using batch

    for epoch in range(epochs):
        start = time.time()

        display.clear_output(wait=True)

        for example_input, example_itm_target, example_target in test_ds.take(1):
            generate_images(first_generator,
                            second_generator,
                            example_input,
                            example_itm_target,
                            example_target,
                            epoch + resumecheck)
        print("Epoch: ", epoch + resumecheck)

        cur_train_batch = train_ds.take(700)
        # Train
        cur_ratioloss = 0
        for n, (input_image, intermediate_target, target) in tqdm.tqdm(cur_train_batch.enumerate()):
            cur_ratioloss = train_step(input_image, intermediate_target, target, epoch + resumecheck, cur_ratioloss)

        # saving (checkpoint) the model every 20 epochs
        if (epoch + resumecheck + 1) % 20 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        print('Time taken for epoch {} is {} sec\n'.format(epoch + resumecheck + 1, time.time() - start))
    checkpoint.save(file_prefix=checkpoint_prefix)
    # New save method for h5 format:
    # Uncomment for training in Colab instances:
    h5_dir = f'/Users/simbaxu/Desktop/229_pix2pix/{training_outputdir}/'
    first_h5_name = '1generator_model.h5'
    second_h5_name = '2generator_model.h5'
    first_generator.save(h5_dir + first_h5_name)
    second_generator.save(h5_dir + second_h5_name)


# #tensorboard = TensorBoard(log_dir= f'{training_outputdir}/logs/tb/{}'.format(time()))
fit(train_dataset, EPOCHS, test_dataset, 0)

# Restore from checkpoint

# checkpoint_dir = f'/Users/simbaxu/Desktop/229_pix2pix/{training_outputdir}/training_checkpoints'
# print(checkpoint_dir)
# checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

# first_discriminator = checkpoint.first_discriminator
# first_discriminator_optimizer = checkpoint.first_discriminator_optimizer
# first_generator = checkpoint.first_generator
# first_generator_optimizer = checkpoint.first_generator_optimizer
# second_discriminator = checkpoint.second_discriminator
# second_discriminator_optimizer = checkpoint.second_discriminator_optimizer
# second_generator = checkpoint.second_generator
# second_generator_optimizer = checkpoint.second_generator_optimizer
# fit(train_dataset, EPOCHS - 20, test_dataset, 20)
