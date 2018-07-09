# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""Downloads and converts Flowers data to TFRecords of TF-Example protos.

This module downloads the Flowers data, uncompresses it, reads the files
that make up the Flowers data and creates two TFRecord datasets: one for train
and one for test. Each TFRecord dataset is comprised of a set of TF-Example
protocol buffers, each of which contain a single image and label.

The script should take about a minute to run.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys

import tensorflow as tf
import cv2
import numpy as np

from datasets import dataset_utils

# The URL where the UCF101 data can be downloaded.
# _DATA_URL = ''

# The ratios of train set and validation set.
_RATIO_TRAIN = 0.8

# Seed for repeatability.
_RANDOM_SEED = 0

# The number of shards per dataset split.
_NUM_SHARDS = 10


class VideoReader(object):
    """Helper class that provides TensorFlow image coding utilities."""

    # def __init__(self):
    #   # Initializes function that decodes RGB JPEG data.
    #   self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
    #   self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

    @staticmethod
    def read_video_props(file_name):
        video = cv2.VideoCapture(file_name)
        frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
        width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
        return video, int(frame_count), int(height), int(width)

    def convert_video_to_numpy(self, file_name):
        video, frame_count, ori_height, ori_width = self.read_video_props(file_name)
        width = height = 240
        buf = np.empty((frame_count, height, width, 3), dtype=np.uint8)
        fc = 0
        ret = True

        while fc < frame_count and ret:
            ret, image = video.read()
            image = cv2.resize(image, (height, width))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            buf[fc] = image
            del(image)
            fc += 1

        video.release()
        assert len(buf.shape) == 4
        assert buf.shape[3] == 3
        return buf, frame_count, height, width, ori_height, ori_width


def _get_filenames_and_classes(dataset_dir):
    """Returns a list of filenames and inferred class names.

    Args:
      dataset_dir: A directory containing a set of subdirectories representing
        class names. Each subdirectory should contain PNG or JPG encoded images.

    Returns:
      A list of image file paths, relative to `dataset_dir` and the list of
      subdirectories, representing class names.
    """
    ucf_root = os.path.join(dataset_dir, 'UCF101')
    directories = []
    class_names = []
    for filename in os.listdir(ucf_root):
        path = os.path.join(ucf_root, filename)
        if os.path.isdir(path):
            directories.append(path)
            class_names.append(filename)

    video_filenames = []
    for directory in directories:
        for sub_directory in os.listdir(directory):
            if sub_directory != 'Annotation':
                for filename in os.listdir(os.path.join(directory, sub_directory)):
                    path = os.path.join(directory, sub_directory, filename)
                    video_filenames.append(path)

    return video_filenames, sorted(class_names)


def _get_class_in_filename(filename):
    return os.path.basename(os.path.dirname(os.path.dirname(filename)))


def _get_dataset_filename(dataset_dir, split_name, shard_id):
    output_filename = 'ucf11_%s_%05d-of-%05d.tfrecord' % (
        split_name, shard_id, _NUM_SHARDS)
    return os.path.join(dataset_dir, 'UCF101-tfrecord', output_filename)


def _convert_dataset(split_name, filenames, class_names_to_ids, dataset_dir):
    """Converts the given filenames to a TFRecord dataset.

    Args:
      split_name: The name of the dataset, either 'train' or 'validation'.
      filenames: A list of absolute paths to png or jpg images.
      class_names_to_ids: A dictionary from class names (strings) to ids
        (integers).
      dataset_dir: The directory where the converted datasets are stored.
    """
    assert split_name in ['train', 'validation', 'test']

    num_per_shard = int(math.ceil(len(filenames) / float(_NUM_SHARDS)))

    with tf.Graph().as_default():
        video_reader = VideoReader()

        with tf.Session('') as sess:

            for shard_id in range(_NUM_SHARDS):
                output_filename = _get_dataset_filename(
                    dataset_dir, split_name, shard_id)

                with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
                    start_ndx = shard_id * num_per_shard
                    end_ndx = min((shard_id + 1) * num_per_shard, len(filenames))
                    for i in range(start_ndx, end_ndx):
                        sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
                            i + 1, len(filenames), shard_id))
                        sys.stdout.flush()

                        # Read the filename:
                        video_data, fc, height, width, _, _ = video_reader.convert_video_to_numpy(filenames[i])
                        video_data = video_data.tostring()

                        class_name = _get_class_in_filename(filenames[i])
                        class_id = class_names_to_ids[class_name]

                        example = dataset_utils.video_to_tfexample(
                            video_data, b'mpg', fc, height, width, class_id)
                        tfrecord_writer.write(example.SerializeToString())

    sys.stdout.write('\n')
    sys.stdout.flush()


# def _clean_up_temporary_files(dataset_dir):
#     """Removes temporary files used to create the dataset.
#
#     Args:
#       dataset_dir: The directory where the temporary files are stored.
#     """
#     filename = _DATA_URL.split('/')[-1]
#     filepath = os.path.join(dataset_dir, filename)
#     tf.gfile.Remove(filepath)
#
#     tmp_dir = os.path.join(dataset_dir, 'flower_photos')
#     tf.gfile.DeleteRecursively(tmp_dir)


def _dataset_exists(dataset_dir):
    for split_name in ['train', 'validation']:
        for shard_id in range(_NUM_SHARDS):
            output_filename = _get_dataset_filename(
                dataset_dir, split_name, shard_id)
            if not tf.gfile.Exists(output_filename):
                return False
    return True


def run(dataset_dir):
    """Runs the download and conversion operation.

    Args:
      dataset_dir: The dataset directory where the dataset is stored.
    """
    if not tf.gfile.Exists(dataset_dir):
        tf.gfile.MakeDirs(dataset_dir)

    if _dataset_exists(dataset_dir):
        print('Dataset files already exist. Exiting without re-creating them.')
        return

    # dataset_utils.download_and_uncompress_tarball(_DATA_URL, dataset_dir)
    video_filenames, class_names = _get_filenames_and_classes(dataset_dir)
    class_names_to_ids = dict(zip(class_names, range(len(class_names))))

    # Divide into train and test:
    random.seed(_RANDOM_SEED)
    random.shuffle(video_filenames)

    num_train = int(len(video_filenames) * _RATIO_TRAIN)
    num_validation = len(video_filenames) - num_train

    training_filenames = video_filenames[:num_train]
    if num_validation != 0:
        validation_filenames = video_filenames[num_train:]
    test_filenames = video_filenames[num_validation:]

    log_name = os.path.join(dataset_dir, 'UCF101-tfrecord', 'log_ucf101.txt')
    with tf.gfile.Open(log_name, 'w') as log:
        log.write('_NUM_SHARDS: %d\n\n' % _NUM_SHARDS)
        log.write('Number training file names: %d\n' % len(training_filenames))
        if num_validation != num_train:
            log.write('Number validation file names: %d\n' % len(validation_filenames))
        log.write('Number test file names: %d\n' % len(test_filenames))

    # First, convert the training and validation sets.
    _convert_dataset('train', training_filenames, class_names_to_ids,
                     dataset_dir)
    if num_validation != num_train:
        _convert_dataset('validation', validation_filenames, class_names_to_ids,
                         dataset_dir)
    _convert_dataset('test', test_filenames, class_names_to_ids,
                     dataset_dir)

    # Finally, write the labels file:
    labels_to_class_names = dict(zip(range(len(class_names)), class_names))
    dataset_utils.write_label_file(labels_to_class_names, os.path.join(dataset_dir, 'UCF101-tfrecord'))

    # _clean_up_temporary_files(dataset_dir)
    print('\nFinished converting the UCF101 dataset!')


def test(dataset_dir):
    """Test the download and conversion operation.

    Args:
      dataset_dir: The dataset directory where the dataset is stored.
    """
    if not tf.gfile.Exists(dataset_dir):
        tf.gfile.MakeDirs(dataset_dir)

    if _dataset_exists(dataset_dir):
        print('Dataset files already exist. Exiting without re-creating them.')
        return

    # dataset_utils.download_and_uncompress_tarball(_DATA_URL, dataset_dir)
    video_filenames, class_names = _get_filenames_and_classes(dataset_dir)
    class_names_to_ids = dict(zip(class_names, range(len(class_names))))

    # Divide into train and test:
    random.seed(_RANDOM_SEED)
    random.shuffle(video_filenames)

    num_train = math.floor(len(video_filenames) * _RATIO_TRAIN)

    training_filenames = video_filenames[:num_train]
    validation_filenames = video_filenames[num_train:]
    test_filenames = video_filenames[:]

    print('Number training file names:', len(training_filenames))
    print('Number validation file names:', len(validation_filenames))
    print('Number test file names:', len(test_filenames))

    print('Test convert video')
    video_reader = VideoReader()
    filename_sample = training_filenames[0]
    video_data, frame_count, height, width, ori_height, ori_width = video_reader.convert_video_to_numpy(filename_sample)
    print('Class:', _get_class_in_filename(filename_sample))
    print('Video size: %dx%dx%d' % (frame_count, ori_height, ori_width))
    print('Show video')
    for i in range(frame_count):
        cv2.imshow(_get_class_in_filename(filename_sample), video_data[i])
        cv2.waitKey(0)
    cv2.destroyAllWindows()
