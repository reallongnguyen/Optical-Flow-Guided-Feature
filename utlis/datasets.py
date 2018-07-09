from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import tensorflow as tf
import numpy as np

LABELS_FILENAME = 'labels.txt'


def read_split_file(dataset_dir, filename, class_names_to_labels):
    file_path = os.path.join(dataset_dir, filename)
    with tf.gfile.Open(file_path, 'r') as f:
        lines = f.read()
    lines = lines.split('\n')
    lines = filter(None, lines)

    listfile = []
    labels = []
    for line in lines:
        listfile.append(os.path.join(dataset_dir, line))
        index = line.index('/')
        class_name = line[:index]
        labels.append(class_names_to_labels[class_name])

    return listfile, labels


def read_label_file(dataset_dir, filename=LABELS_FILENAME):
    """Reads the labels file and returns a mapping from ID to class name.

    Args:
      dataset_dir: The directory in which the labels file is found.
      filename: The filename where the class names are written.

    Returns:
      A map from a label (integer) to class name.
    """
    labels_filename = os.path.join(dataset_dir, filename)
    with tf.gfile.Open(labels_filename, 'r') as f:
        lines = f.read()  # .decode()
    lines = lines.split('\n')
    lines = filter(None, lines)

    labels_to_class_names = {}
    class_names_to_labels = {}
    for line in lines:
        index = line.index(':')
        labels_to_class_names[int(line[:index])] = line[index + 1:]
        class_names_to_labels[line[index + 1:]] = int(line[:index])
    return labels_to_class_names, class_names_to_labels


class Dataset(object):
    def __init__(self, dataset_dir, listfile, class_names_to_labels, mode='train'):
        self._dataset_dir = dataset_dir
        self._listfile = listfile
        self._class_names_to_lables = class_names_to_labels
        self._mode = mode
        self._pos = 0
        self._read_listfile()

    @property
    def files(self):
        return self._files

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    def _read_listfile(self):
        self._files, self._labels = read_split_file(self._dataset_dir, self._listfile, self._class_names_to_lables)
        self._num_examples = len(self._files)

    def next_batch(self, batch_size):
        if self._pos + batch_size <= self._num_examples:
            prev_pos = self._pos
            self._pos += batch_size
            return self._files[prev_pos:self._pos], self._labels[prev_pos:self._pos]
        else:
            # if (self._mode == 'train'):
            #     # shuffle

            self._pos = batch_size
            return self._files[0:self._pos], self._labels[0:self._pos]


def build_datasets(dataset_dir, trainlist, testlist, labels_filename=LABELS_FILENAME):
    class Datasets(object):
        pass

    datasets = Datasets()

    labels_to_class_names, class_names_to_labels = read_label_file(dataset_dir, labels_filename)
    datasets.labels_to_class_names = labels_to_class_names

    datasets.test = Dataset(dataset_dir, testlist, class_names_to_labels, mode='test')

    return datasets
