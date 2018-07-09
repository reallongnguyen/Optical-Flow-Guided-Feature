from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os
import math
import random

import cv2

_TRAIN = 0.7


def _get_video_directories_and_classes(dataset_dir):
    """Returns a list of filenames and inferred class names.

    Args:
      dataset_dir: A directory containing a set of subdirectories representing
        class names. Each subdirectory should contain PNG or JPG encoded images.

    Returns:
      A list of image file paths, relative to `dataset_dir` and the list of
      subdirectories, representing class names.
    """
    ucf_root = os.path.join(dataset_dir, 'UCF11')
    action_dir = []
    class_names = []
    for filename in os.listdir(ucf_root):
        path = os.path.join(ucf_root, filename)
        if os.path.isdir(path):
            action_dir.append(filename)
            class_names.append(filename)

    train_dir = []
    test_dir = []
    for directory in action_dir:
        path = os.path.join(ucf_root, directory)
        # video_dir = []
        for dir in os.listdir(path):
            if dir != 'Annotation':
                sufix = int(dir.split('_')[-1])
                if sufix <= 20:
                    train_dir.append(os.path.join(directory, dir))
                else:
                    test_dir.append(os.path.join(directory, dir))
        #         video_dir.append(os.path.join(directory, dir))
        # random.shuffle(video_dir)
        #
        # number_train_dir = math.ceil(_TRAIN * len(video_dir))
        #
        # train_dir += video_dir[:number_train_dir]
        # test_dir += video_dir[number_train_dir:]

    return train_dir, test_dir, sorted(class_names)


def _get_list_file(dataset_dir, list_dir):
    ucf_root = os.path.join(dataset_dir, 'UCF11')

    list_file = []

    for video_dir in list_dir:
        path = os.path.join(ucf_root, video_dir)
        for file in os.listdir(path):
            video = cv2.VideoCapture(os.path.join(path, file))
            frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
            if frame_count > 25:
                list_file.append(os.path.join(video_dir, file))
            else:
                print(os.path.join(video_dir, file))
            video.release()

    return list_file


def run(dataset_dir='.', mode=0):
    random.seed(0)
    train_dir, test_dir, class_name = _get_video_directories_and_classes(dataset_dir)

    train_files = _get_list_file(dataset_dir, train_dir)
    test_files = _get_list_file(dataset_dir, test_dir)

    if mode != 0:
        files = train_files + test_files
        random.shuffle(files)
        num_train_files = math.ceil(_TRAIN * len(files))
        train_files = files[:num_train_files]
        test_files = files[num_train_files:]

    with open(os.path.join(dataset_dir, 'UCF11', 'trainlist.txt'), 'w') as file:
        for file_name in sorted(train_files):
            file.write(file_name + '\n')

    with open(os.path.join(dataset_dir, 'UCF11', 'testlist.txt'), 'w') as file:
        for file_name in sorted(test_files):
            file.write(file_name + '\n')


run(mode=0)
