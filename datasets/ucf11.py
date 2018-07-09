"""Provides data for the ucf11 dataset.

The dataset scripts used to create the dataset can be found at: convert_ucf11.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf

from datasets import dataset_utils
from collections import namedtuple

_FILE_PATTERN = 'ucf11_%s_*.tfrecord'

SPLITS_TO_SIZES = {'train': 1147, 'test': 449}

_NUM_CLASSES = 11

_ITEMS_TO_DESCRIPTIONS = {
    'video': 'A sequent of color image of varying size.',
    'label': 'A single integer between 0 and 10',
}

_ALPHA = 2
_BETA = 11

KEY_TO_FEATURES = {
    'video/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
    # 'video/format': tf.FixedLenFeature((), tf.string, default_value='mpg'),
    'video/class/label': tf.FixedLenFeature(
        [], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
    'video/frame_count': tf.FixedLenFeature(
        [], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
    'video/height': tf.FixedLenFeature(
        [], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
    'video/width': tf.FixedLenFeature(
        [], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
    'video/channel': tf.FixedLenFeature(
        [], tf.int64, default_value=[3]),
}

Dataset = namedtuple('Dataset', ['filenames', 'num_samples', 'num_classes',
                                 'labels_to_names', 'items_to_descriptions'])


def get_split(split_name, dataset_dir, file_pattern=None):
    """Gets a dataset tuple with instructions for reading flowers.

    Args:
      split_name: A train/validation split name.
      dataset_dir: The base directory of the dataset sources.
      file_pattern: The file pattern to use when matching the dataset sources.
        It is assumed that the pattern contains a '%s' string so that the split
        name can be inserted.
      reader: The TensorFlow reader type.

    Returns:
      A `Dataset` namedtuple.

    Raises:
      ValueError: if `split_name` is not a valid train/validation split.
    """
    if split_name not in SPLITS_TO_SIZES:
        raise ValueError('split name %s was not recognized.' % split_name)

    if not file_pattern:
        file_pattern = _FILE_PATTERN
    file_pattern = os.path.join(dataset_dir, file_pattern % split_name)

    # Allowing None in the signature so that dataset_factory can use the default.
    # if reader is None:
    #     reader = tf.TFRecordReader

    labels_to_names = None
    if dataset_utils.has_labels(dataset_dir):
        labels_to_names = dataset_utils.read_label_file(dataset_dir)

    filenames = tf.gfile.Glob(file_pattern)

    return Dataset(
        filenames=filenames,
        num_samples=SPLITS_TO_SIZES[split_name],
        num_classes=_NUM_CLASSES,
        labels_to_names=labels_to_names,
        items_to_descriptions=_ITEMS_TO_DESCRIPTIONS
    )


def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(serialized_example, KEY_TO_FEATURES)

    encoded = tf.decode_raw(features['video/encoded'], tf.uint8)
    frame_count = tf.cast(features['video/frame_count'], tf.int32)
    label = tf.cast(features['video/class/label'], tf.int32)
    height = tf.cast(features['video/height'], tf.int32)
    width = tf.cast(features['video/width'], tf.int32)
    channel = tf.cast(features['video/channel'], tf.int32)

    encoded = tf.reshape(encoded, [frame_count, height, width, channel])

    return encoded, label, [frame_count, height, width, channel]


def build_data(dataset, is_training=True):
    filename_queue = tf.train.string_input_producer(dataset.filenames)

    # Read one video

    encoded, label, shape = read_and_decode(filename_queue)
    # c = lambda encoded, label, shape: tf.less(shape[0], 25)
    # b = lambda encoded, label, shape: read_and_decode(filename_queue)
    # encoded, label, shape = tf.while_loop(c, b, (encoded, label, shape))
    label = tf.reshape(label, [])
    frame_count = shape[0]
    delta = frame_count // _BETA
    # delta = tf.cond(tf.greater(delta, 0), lambda: delta, lambda: 1)
    p_max = frame_count - 1 - (_ALPHA - 1) * delta

    if is_training:
        p = tf.cast(tf.random_uniform([], minval=0, maxval=tf.cast(p_max, tf.float32)), tf.int32)
        index = tf.range(p, p + (_ALPHA - 1) * delta + 1, delta)
        exampled_frames = tf.gather(encoded, index)
        exampled_frames.set_shape([_ALPHA, 240, 240, 3])
    else:
        p = 0
        index = tf.range(p, p + (_BETA - 1) * delta + 1, delta)
        exampled_frames = tf.gather(encoded, index)
        exampled_frames.set_shape([_BETA, 240, 240, 3])

    return exampled_frames, label
    # Make queue
    # if mode == 'train':
    #     example_queue = tf.RandomShuffleQueue(
    #         capacity=16 * batch_size,
    #         min_after_dequeue=8 * batch_size,
    #         dtypes=[tf.float32, tf.int32],
    #         shapes=[out_shapes, [1]]
    #     )
    #     num_threads = 16
    # elif mode == 'validate':
    #     example_queue = tf.FIFOQueue(
    #         3 * batch_size,
    #         dtypes=[tf.float32, tf.int64],
    #         shapes=[out_shapes, [1]]
    #     )
    #     num_threads = 1
    # else:
    #     example_queue = tf.FIFOQueue(
    #         3 * batch_size,
    #         dtypes=[tf.float32, tf.int64],
    #         shapes=[out_shapes, [1]]
    #     )
    #     num_threads = 1
    #
    # example_queue_op = example_queue.enqueue([extract_frames, label])
    # tf.train.add_queue_runner(tf.train.queue_runner.QueueRunner(
    #     example_queue, enqueue_ops=[example_queue_op] * num_threads))
    #
    # # Read batch frames, label
    # inputs, labels = example_queue.dequeue_many(batch_size)
    # labels = tf.reshape(labels, [batch_size])
    # labels = tf.one_hot(labels, dataset.num_classes)
    #
    # assert len(inputs.shape) == 5
    # assert inputs.shape[0] == batch_size
    # assert inputs.shape[-1] == out_shapes[-1]
    # assert inputs.shape[1] == out_shapes[0]
    #
    # return inputs, labels
