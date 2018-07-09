from datasets import ucf11
import tensorflow as tf
import numpy as np
from matplotlib import image

from preprocessing import off_preprocessing

BATCH_SIZE = 1
_ALPHA = 2
_BETA = 8

_OUTPUT_SHAPES = [_ALPHA, 240, 240, 3]
_ORIGINAL_OUTPUT_SHAPES = [_ALPHA, 240, 240, 3]


sobel_x = tf.constant([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], tf.float32)
sobel_x_filter = tf.reshape(sobel_x, [3, 3, 1, 1])
sobel_y_filter = tf.transpose(sobel_x_filter, [1, 0, 2, 3])


def sobel(feature):
    with tf.variable_scope('sobel'):
        channels = tf.unstack(feature, axis=3)
        fx = []
        fy = []

        for channel in channels:
            channel = tf.expand_dims(channel, axis=3)
            filtered_x = tf.nn.conv2d(channel, sobel_x_filter, [1, 1, 1, 1], padding='SAME')
            filtered_y = tf.nn.conv2d(channel, sobel_y_filter, [1, 1, 1, 1], padding='SAME')
            fx.append(filtered_x)
            fy.append(filtered_y)

        return tf.concat(fx, axis=3), tf.concat(fy, axis=3)


def test():
    print('Test read data')
    dataset = ucf11.get_split('test', 'data/UCF11-tfrecord')
    input, label = ucf11.build_data(dataset, 'train',batch_size=BATCH_SIZE)

    new_input = off_preprocessing.preprocess_image(input, 240, 240, is_training=True)

    example_queue = tf.FIFOQueue(
        3 * BATCH_SIZE,
        dtypes=[tf.float32, tf.uint8, tf.int32],
        shapes=[_OUTPUT_SHAPES, _ORIGINAL_OUTPUT_SHAPES, []]
    )
    num_threads = 1

    example_queue_op = example_queue.enqueue([new_input, input, label])
    tf.train.add_queue_runner(tf.train.queue_runner.QueueRunner(
        example_queue, enqueue_ops=[example_queue_op] * num_threads))

    new_inputs, inputs, labels = example_queue.dequeue_many(BATCH_SIZE)

    new_images = tf.unstack(new_inputs, axis=0)

    print(new_images[0])

    fx, fy = sobel(new_images[0])

    spax = tf.unstack(fx, axis=0)
    spay = tf.unstack(fy, axis=0)

    print(fx)
    print(fy)
    print(spax)

    with tf.Session() as sess:
        tf.train.start_queue_runners(sess)
        sx, sy, l = sess.run([spax[0], spay[0], labels])
        for i in range(sx.shape[2]):
            image.imsave('%d_fx%d' % (l[0], i), sx[:,:,i], cmap='gray')
            image.imsave('%d_fy%d' % (l[0], i), sy[:,:,i], cmap='gray')

test()