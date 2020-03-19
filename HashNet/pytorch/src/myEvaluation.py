# -*- coding: utf-8 -*-

import os
import numpy as np
from publicFunctions import make_one_hot


def mean_average_precision(params, R):
    database_code = params['database_code']
    validation_code = params['test_code']
    database_labels = params['database_labels']
    validation_labels = params['test_labels']
    query_num = validation_code.shape[0]

    sim = np.dot(database_code, validation_code.T)
    ids = np.argsort(-sim, axis=0)
    APx = []

    for i in range(query_num):
        label = validation_labels[i, :]
        label[label == 0] = -1
        idx = ids[:, i]
        imatch = np.sum(database_labels[idx[0:R], :] == label, axis=1) > 0
        relevant_num = np.sum(imatch)
        Lx = np.cumsum(imatch)
        Px = Lx.astype(float) / np.arange(1, R + 1, 1)
        if relevant_num != 0:
            APx.append(np.sum(Px * imatch) / relevant_num)

    return np.mean(np.array(APx))


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    # Possible values of job_dataset and net. (Easy to copy them)
    job_values = ['mnist', 'cifar10', 'fashion_mnist', 'imagenet', 'places365']
    net_values = ['ResNet18', 'ResNet34', 'AlexNet', 'ResNet50', 'ResNet101', 'ResNet152', 'ResNext101_32x4d']
    job_dataset = 'places365'
    net = 'ResNet152'

    from publicVariables import iter_lists

    snapshot_path = '../snapshot/' + job_dataset + '_48bit_' + net + '_hashnet/'
    model_path = snapshot_path + 'iter_%05d_model.pth.tar' % (iter_lists[net][job_dataset])
    query_path = './save_for_load/blackbox/' + net + '/' + job_dataset + '_test_output_code_label.npz'
    database_path = './save_for_load/blackbox/' + net + '/' + job_dataset + '_database_output_code_label.npz'
    R = 500

    C = 36

    # load and set the databaset code
    tmp = np.load(database_path)
    output, code, multi_label = tmp['arr_0'], tmp['arr_1'], tmp['arr_2']
    if 'mnist' in job_dataset or job_dataset == 'cifar10':
        multi_label = make_one_hot(multi_label)
    if 'imagenet' in job_dataset:
        multi_label = make_one_hot(multi_label, C=100)
    if 'places365' in job_dataset:
        multi_label = make_one_hot(multi_label, C=C)
        # load and set the test code
    tmp = np.load(query_path)
    query_output, query_code, query_multi_label = tmp['arr_0'], tmp['arr_1'], tmp['arr_2']
    if 'mnist' in job_dataset or job_dataset == 'cifar10':
        query_multi_label = make_one_hot(query_multi_label)
    if 'imagenet' in job_dataset:
        query_multi_label = make_one_hot(query_multi_label, C=100)
    if 'places365' in job_dataset:
        query_multi_label = make_one_hot(query_multi_label, C=C)
    import random

    random.seed(1)
    query_size = 500
    random_query_index = random.sample(range(query_multi_label.shape[0]), 1000)[:query_size]


    isTest = False

    if isTest:
        code_and_label = {"database_code": code, "database_labels": multi_label, \
                          "test_code": query_code[random_query_index],
                          "test_labels": query_multi_label[random_query_index]}
    else:
        code_and_label = {"database_code": code, "database_labels": multi_label, \
                          "test_code": code[random_query_index], \
                          "test_labels": multi_label[random_query_index]}

    mAP = mean_average_precision(code_and_label, R)

    print ("MAP: " + str(mAP))

