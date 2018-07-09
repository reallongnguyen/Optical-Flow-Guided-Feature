from __future__ import print_function, division, unicode_literals, absolute_import
import tensorflow as tf
import numpy as np
import cv2

from nets import nets_factory
from preprocessing import preprocessing_factory
from nets.off_v2 import off
from datasets.dataset_utils import read_label_file, read_split_file

from utlis.make_gif import make_gif

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'video',
    None,
    'Video you want to prediction.')

tf.app.flags.DEFINE_string(
    'video_list',
    None,
    'List file you want to prediction.')

tf.app.flags.DEFINE_string(
    'dataset_dir',
    '.',
    'Dataset directory.')

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

X = tf.placeholder(tf.uint8, [11, 240, 240, 3], name='input')


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
    logits_stream = []

    for i in range(0, 10, 1):
        logits = off_rgb(images[i], images[i + 1], str(i))
        logits_arr.append(logits)
        logits_stream.append(tf.reduce_mean(tf.stack(logits_arr, axis=2), axis=2))

    return logits_stream, logits_arr


logits_stream, logits_arr = network()
predictions = []

for logits in logits_stream:
    predictions.append(tf.nn.softmax(logits))


def get_3_max_arg_pred(pred):
    arg_max3 = pred.argsort()[-3:][::-1]
    return arg_max3


def format_name(name):
    return name.replace('_', ' ').title()


def main(_):
    if not FLAGS.video and not FLAGS.video_list:
        raise ValueError('You must supply video you want to prediction with --video or --video_list')

    if FLAGS.video_list:
        video_list = read_split_file(FLAGS.dataset_dir, FLAGS.video_list)
    else:
        video_list = [FLAGS.video]

    saver = tf.train.Saver()

    label_to_name = read_label_file('data/UCF11-tfrecord', 'labels.txt')

    if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
        checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
    else:
        checkpoint_path = FLAGS.checkpoint_path

    tf.logging.info('Evaluating %s' % checkpoint_path)

    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (20, 200)
    fontScale = 0.6
    fontColor = (255, 255, 255)
    lineType = 2

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, checkpoint_path)

        for file in video_list:
            frames = read_video(file)
            sample = get_sample(frames)

            name_stream = []
            pred_stream = []

            preds, ls, la = sess.run([predictions, logits_stream, logits_arr], feed_dict={X: sample})

            for pred in preds:
                pred = np.squeeze(pred)
                label = np.argmax(pred)
                name = label_to_name[label]
                name_stream.append(name)
                pred_stream.append(pred)

            class_name = file.split('/')[3]

            # max3 = pred.argsort()[-3:][::-1]
            # for item in max3:
            #     name = label_to_name[item]
            #     name = name.replace('_', ' ').title()
            #     print('%s %d' % (name, pred[item]))

            sample_length = sample.shape[0]

            for i in range(sample_length):
                frame = sample[i]
                frame = cv2.resize(frame, (320, 224))

                pred = pred_stream[i if i < len(name_stream) else len(name_stream) - 1]
                max_arg_pred = get_3_max_arg_pred(pred)

                cv2.putText(frame,
                            'Frame %2d/%2d' % (i + 1, sample_length),
                            (20, 160),
                            font,
                            0.5,
                            fontColor,
                            1)

                for j in range(3):
                    label = max_arg_pred[j]
                    action = label_to_name[label]
                    action = format_name(action)

                    cv2.putText(frame,
                                '%.2f' % (pred[label]),
                                (20, 180 + j * 15),
                                font,
                                0.5,
                                fontColor,
                                1)

                    cv2.putText(frame,
                                action,
                                (70, 180 + j * 15),
                                font,
                                0.5,
                                fontColor,
                                1)

                video_class_name = format_name(class_name)

                # get boundary of this text
                textsize = cv2.getTextSize(video_class_name, font, fontScale, lineType)[0]

                # get coords based on boundary
                textX = (frame.shape[1] - textsize[0]) // 2
                textY = (frame.shape[0] + textsize[1]) // 2

                cv2.putText(frame, video_class_name,
                            (textX, 20),
                            font,
                            fontScale,
                            fontColor,
                            lineType)

                cv2.imwrite('tmp_%d.png' % (i), frame)

                # cv2.imshow('win', frame)
                # cv2.waitKey(0)
            make_gif(class_name, fps=6)
            # cv2.destroyAllWindows()


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
