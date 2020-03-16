# -*- coding: utf-8 -*-
import numpy as np

Hashbit = 48

data_list_path = {'imagenet': {'database': "../data/imagenet/database.txt", \
                               'test': "../data/imagenet/test.txt", \
                               'train_set1': "../data/imagenet/train.txt", \
                               'train_set2': "../data/imagenet/train.txt", \
                               'train': "../data/imagenet/train.txt"}, \
                  'places365': {'database': "../data/places365_standard/database.txt", \
                                'test': "../data/places365_standard/val.txt", \
                                'train_set1': "../data/places365_standard/train.txt", \
                                'train_set2': "../data/places365_standard/train.txt", \
                                'train': "../data/places365_standard/train.txt"} \
                  }

th_h = {'fashion_mnist': 16, 'mnist': 16, 'cifar10': 15, 'imagenet': 12, \
        'nus_wide': 12, 'places365': 10}
th_l = {'fashion_mnist': 10, 'mnist': 8, 'cifar10': 7, 'imagenet': 8, \
        'nus_wide': 8, 'places365': 8}

iter_lists = {'ResNet152': {'imagenet': 94000, 'nus_wide': 88000, 'places365': 96000},
              'ResNet101': {'imagenet': 47000, 'places365': 100000},
              'ResNet50': {'mnist': 46000, 'coco': 20000, 'cifar10': 10000, 'imagenet': 10000, \
                           'imagenet64': 26000, 'fashion_mnist': 8000, 'places365': 100000},
              'ResNet34': {'imagenet': 10000},
              'ResNet18': {'imagenet': 33000},
              'VGG16BN': {'imagenet': 10000},
              'ResNext101_32x4d': {'imagenet': 94000, 'places365': 82000}
              }

multi_label_dataset_list = ['nus_wide']



