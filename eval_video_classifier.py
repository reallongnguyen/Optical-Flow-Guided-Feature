from __future__ import print_function, division, unicode_literals, absolute_import
import tensorflow as tf
import numpy as np
import cv2

from nets import nets_factory
from preprocessing import preprocessing_factory
from nets.off_v2 import off
from utlis.datasets import build_datasets

from utlis.make_gif import make_gif

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'video',
    None,
    'Duong dan cua video ban muon du doan.')

tf.app.flags.DEFINE_string(
    'checkpoint_path', 'results/ucf11-off/train',
    'The directory where the model was written to or an absolute path to a '
    'checkpoint file.')

tf.app.flags.DEFINE_string(
    'eval_dir', 'results/ucf11-off/test', 'Directory where the results are saved to.')

tf.app.flags.DEFINE_string(
    'model_name', 'inception_v2', 'The name of the architecture to evaluate.')

tf.app.flags.DEFINE_string(
    'resnet_model_name', 'resnet_v2_26', 'The name of the resnet model to evaluate')

tf.app.flags.DEFINE_string(
    'preprocessing_name', 'off', 'The name of the preprocessing to use. If left '
                                 'as `None`, then the model_name flag is used.')

tf.app.flags.DEFINE_integer(
    'eval_image_size', 224, 'Eval image size')

WIDTH = 240
HEIGHT = 240
_BETA = 11

NUM_CLASSES = 11

X = tf.placeholder(tf.uint8, [_BETA, HEIGHT, WIDTH, 3], name='samples')


def get_basic_feature(endpoints):
    f_k = [
        endpoints['Conv2d_1a_7x7'],
    ]

    # Feature with size k/2
    f_k2 = [
        endpoints['MaxPool_2a_3x3'],
        endpoints['Conv2d_2b_1x1'],
        endpoints['Conv2d_2c_3x3'],
    ]

    # Feature with size k/4
    f_k4 = [
        endpoints['MaxPool_3a_3x3'],
        endpoints['Mixed_3b'],
        endpoints['Mixed_3c'],
    ]
    return f_k, f_k2, f_k4


def read_video(url):
    video = cv2.VideoCapture(url)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    # ori_width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
    # ori_height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)

    frames = np.empty((frame_count, HEIGHT, WIDTH, 3), dtype=np.uint8)
    fc = 0
    ret = True

    while fc < frame_count and ret:
        ret, image = video.read()
        image = cv2.resize(image, (HEIGHT, WIDTH))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        frames[fc] = image
        del (image)
        fc += 1

    video.release()

    return frames


def get_sample(frames):
    frame_count = frames.shape[0]
    delta = frame_count // _BETA
    index = np.arange(0, _BETA * delta, delta)
    sample = frames[index]
    return sample


network_fn = nets_factory.get_network_fn(
    FLAGS.model_name,
    num_classes=11,
    is_training=False)
preprocessing_name = FLAGS.preprocessing_name
image_preprocessing_fn = preprocessing_factory.get_preprocessing(
    preprocessing_name,
    is_training=False)


def network():
    eval_image_size = FLAGS.eval_image_size or network_fn.default_image_size

    images = image_preprocessing_fn(X, eval_image_size, eval_image_size)
    images = tf.expand_dims(images, axis=0)
    images = tf.unstack(images, axis=1)

    def off_rgb(image1, image2, name):
        logits1, endpoints1 = network_fn(image1)
        logits2, endpoints2 = network_fn(image2)

        # Feature with size maximum k
        f_k_1, f_k2_1, f_k4_1 = get_basic_feature(endpoints1)
        f_k_2, f_k2_2, f_k4_2 = get_basic_feature(endpoints2)

        logits_off, end_point_off = off(
            f_k_1, f_k_2,
            f_k2_1, f_k2_2,
            f_k4_1, f_k4_2,
            num_classes=11,
            resnet_model_name=FLAGS.resnet_model_name,
            resnet_weight_decay=0.0,
            is_training=False
        )

        logits_gen = tf.reduce_mean(tf.stack([
            logits1,
            logits2
        ], axis=2), axis=2)

        logits = tf.multiply(logits_gen, logits_off, name='logits' + name)

        return logits

    logits_arr = []

    for i in range(0, _BETA - 1, 1):
        logits = off_rgb(images[i], images[i + 1], str(i))
        logits_arr.append(logits)

    return tf.reduce_mean(tf.stack(logits_arr, axis=2), axis=2)


logits = network()
prediction = tf.nn.softmax(logits)


def format_name(name):
    return name.replace('_', ' ').title()


def main(_):
    datasets = build_datasets('data/UCF11', 'trainlist.txt', 'testlist.txt', 'labels.txt')

    saver = tf.train.Saver()

    if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
        checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
    else:
        checkpoint_path = FLAGS.checkpoint_path

    tf.logging.info('Evaluating %s' % checkpoint_path)

    confusionlist = []
    count_true = 0
    confusion_matrix = np.zeros([NUM_CLASSES, NUM_CLASSES], dtype=np.int32)

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, checkpoint_path)

        for iter in range(datasets.test.num_examples):
            if iter % 50 == 0:
                print('Eval %d/%d' % (iter, datasets.test.num_examples))

            files, labels = datasets.test.next_batch(1)
            frames = read_video(files[0])
            sample = get_sample(frames)
            pred = sess.run(prediction, feed_dict={X: sample})

            truth = np.argmax(pred)
            if truth == labels[0]:
                count_true += 1
            else:
                confusionlist.append(files[0])
            confusion_matrix[labels, truth] += 1

        print('Accuracy: %f' % (count_true / datasets.test.num_examples))
        print(confusion_matrix)

        with tf.gfile.Open('confusionlist.txt', 'w') as file:
            for url in confusionlist:
                file.write(url + '\n')
            print('confusionlist.txt has written')


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
