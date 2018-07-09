from __future__ import print_function
from utlis.datasets import build_datasets


def test():
    datasets = build_datasets('data/UCF11', 'trainlist.txt', 'testlist.txt', 'labels.txt')

    if datasets.test.num_examples == 478:
        print('datasets_test: OK')
