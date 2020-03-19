# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable


import matplotlib.pyplot as plt

from myLoss import clusterCircleLoss


from myRetrieval import  get_img_num_by_class_from_img
from myClustering import SavedCluster, SavedDBscanCluster

from publicFunctions import get_dsets, make_one_hot

import time


def get_adv_untarget(model, img_t, targetCode, eps=1.0 / 255, threshold=40, loss=nn.L1Loss()):
    """

    Args:
        model - whitebox model
        img_t (torch.Tensor) - input image
        targetCode (numpy.ndarray) - hash code of input
        eps (float) - the step size of the iFGSM algorithm
        threshold (int) - the targeted distance between the adversarial examples and the original input
        loss_fun (callable) - loss function

    Returns:

    """
    start = time.time()
    if not isinstance(targetCode, torch.cuda.FloatTensor):
        targetCode = Variable(torch.Tensor(targetCode).cuda())
    # BE CAUTIOUS when you do the reshape!
    X = np.array(img_t.unsqueeze(0))
    inputs = Variable(torch.Tensor(X).cuda(), requires_grad=True)
    output = model(inputs)
    targetCode = targetCode.detach()

    l1loss = loss(output, targetCode.detach())
    # l1loss.backward(retain_graph=True)
    l1loss.backward(retain_graph=True)
    tmp = inputs.grad

    tCodeValue = targetCode.cpu().data.numpy()
    oCodeValue = torch.sign(output).cpu().data.numpy()
    # Everything with 'code' in it is the one signed before
    # xxxCodeValue was signed
    #
    print '...non-targeted iterative FGSM begin....'
    print 'initial distance', np.sum(np.abs(tCodeValue - oCodeValue)) / 2
    i = 0
    while np.sum(np.abs(tCodeValue - oCodeValue)) / 2 <= threshold:
        print 'epoch ', i, ' loss: ', l1loss.cpu().data.numpy()
        print 'Hamming distance: ', np.sum(np.abs(tCodeValue - oCodeValue)) / 2

        adv = inputs + eps * torch.sign(inputs.grad)
        tmp = adv.cpu().data.numpy()
        tmp[tmp < 0] = 0
        tmp[tmp > 1] = 1
        inputs = Variable(torch.Tensor(tmp).cuda(), requires_grad=True)
        output_adv = model(inputs)
        l1loss = loss(output_adv, targetCode.detach())
        l1loss.backward(retain_graph=True)
        oCodeValue = torch.sign(output_adv).cpu().data.numpy()

        i = i + 1
        if i >= 20:
            print 'Adv generation failed'
            end = time.time()
            print end - start
            return inputs

    print 'Final Hamming distance : ', np.sum(np.abs(tCodeValue - oCodeValue)) / 2
    return inputs


def get_adv_cluster_circle_loss(model, img_t, lr, code_test, loss_fun, return_params=False, job_dataset='', net = '', K_value=0,
                                max_iters=100):
    """

    Args:
        model - whitebox model
        img_t (torch.Tensor) - input image
        lr (float) - learning rate
        code_test (numpy.ndarray) - hashcode for creating the cluster
        loss_fun (callable) - loss function
        return_params (bool) - returns params or not
        job_dataset (str) - name of dataset
        net (str) - name of the net
        K_value (int) - K value of K-means algorithm
        max_iters (int) - max iteration of optimization algorithm

    Returns:
        if return_params is True, it returns:
            inputs (torch.Variable) - the adversarial example
            cluster centers (numpy.ndarray) - the center of clusters
            distance to cluster centers (numpy.ndarray) - the distance from the adversarial
                example's hash code to the cluster centers after each iteration
        else it returns:
            inputs (torch.Variable) - the adversarial example
    """
    start = time.time()
    K = 0
    if K_value == 0:
        if 'imagenet' in job_dataset:
            K = 200
        elif 'mnist' in job_dataset or 'cifar10' in job_dataset:
            K = 20
        else:
            print 'Not an available dataset, exit'
            exit()
    else:
        K = K_value

    savedCluster = SavedCluster(job_dataset, K, code_test, net)
    circles_index, clusterCircleNums, clusterCenters = savedCluster.get_circle_vars()

    end1 = time.time()
    if not isinstance(clusterCenters, torch.cuda.FloatTensor):
        clusterCenters = Variable(torch.Tensor(clusterCenters).cuda(), requires_grad=True)

    disToCenter_i = np.zeros([max_iters, K]) - 1

    X = np.array(img_t.unsqueeze(0))
    inputs = Variable(torch.Tensor(X).cuda(), requires_grad=True)
    output = model(inputs)

    oCodeValue = torch.sign(output).cpu().data.numpy()
    tCodeValue = oCodeValue

    optimizer = optim.SGD([inputs], lr=lr, momentum=0.9)

    deg = 2
    poly_params = savedCluster.poly_fit(deg)

    print '...non-targeted SGD begin....'
    print 'initial distance', np.sum(np.abs(tCodeValue - oCodeValue)) / 2
    i = 0

    for i in range(max_iters):
        optimizer.zero_grad()
        output_adv = model(inputs)
        loss = loss_fun(output_adv, clusterCenters.detach(), poly_params, threshold_retrieval=0)
        disToCenters = torch.sum(torch.abs(output_adv - clusterCenters), dim=-1) / 2
        disToCenter_i[i] = disToCenters.cpu().data.numpy()
        print 'step', i, ': loss=', loss.cpu().data.numpy()

        if loss.cpu().data.numpy() == 0:
            inputs = torch.clamp(inputs, 0, 1)
            out = model(inputs)
            print np.sum(np.abs(torch.sign(out).cpu().data.numpy() - tCodeValue)) / 2
            if return_params:
                return inputs, clusterCenters.cpu().data.numpy()
            else:
                return inputs

        loss.backward()

        optimizer.step()
        print np.sum(np.abs(torch.sign(output_adv).cpu().data.numpy() - tCodeValue)) / 2

        inputs.data[inputs.data < 0] = 0
        inputs.data[inputs.data > 1] = 1
    end = time.time()
    print 'time counting:', end1 - start, end - start

    if return_params:
        return inputs, clusterCenters.cpu().data.numpy(), disToCenter_i
    else:
        return inputs


def get_adv_cluster_circle_loss_dbscan(model, img_t, lr, code_test, loss_fun, return_params=False, job_dataset='', net = '',
                                       eps=0.5,
                                       max_iters=100):
    """

    Args:
        model - whitebox model
        img_t (torch.Tensor) - input image
        lr (float) - learning rate
        code_test (numpy.ndarray) - hashcode for creating the cluster
        loss_fun (callable) - loss function
        return_params (bool) - returns or not
        job_dataset (str) - name of dataset
        net (str) - name of the net
        eps (float) - eps of the DBSCAN algorithm
        max_iters (int) - max iteration of optimization algorithm

    Returns:
        if return_params is True, it returns:
            inputs (torch.Variable) - the adversarial example
            cluster centers (numpy.ndarray) - the center of clusters
            distance to cluster centers (numpy.ndarray) - the distance from the adversarial
                example's hash code to the cluster centers after each iteration
        else it returns:
            inputs (torch.Variable) - the adversarial example
    """
    savedCluster = SavedDBscanCluster(job_dataset, eps, code_test, net)
    circles_index, clusterCircleNums, clusterCenters = savedCluster.get_circle_vars()

    if not isinstance(clusterCenters, torch.cuda.FloatTensor):
        clusterCenters = Variable(torch.Tensor(clusterCenters).cuda(), requires_grad=True)
    K = clusterCircleNums.shape[0]
    print K
    disToCenter_i = np.zeros([max_iters, K]) - 1

    X = np.array(img_t.unsqueeze(0))
    inputs = Variable(torch.Tensor(X).cuda(), requires_grad=True)
    output = model(inputs)

    oCodeValue = torch.sign(output).cpu().data.numpy()
    tCodeValue = oCodeValue

    optimizer = optim.SGD([inputs], lr=lr, momentum=0.9)

    deg = 2
    poly_params = np.zeros([K, deg + 1])
    X_index = circles_index[0]

    for i in range(K):
        print K
        Y_dps = clusterCircleNums[i]
        poly_params[i] = np.polyfit(X_index, Y_dps, deg)

    print '...non-targeted SGD begin....'
    print 'initial distance', np.sum(np.abs(tCodeValue - oCodeValue)) / 2
    i = 0

    for i in range(max_iters):
        optimizer.zero_grad()
        output_adv = model(inputs)
        loss = loss_fun(output_adv, clusterCenters.detach(), poly_params, threshold_retrieval=0)
        disToCenters = torch.sum(torch.abs(output_adv - clusterCenters), dim=-1) / 2
        disToCenter_i[i] = disToCenters.cpu().data.numpy()
        print 'step', i, ': loss=', loss.cpu().data.numpy()

        if loss.cpu().data.numpy() == 0:
            inputs = torch.clamp(inputs, 0, 1)
            out = model(inputs)
            print np.sum(np.abs(torch.sign(out).cpu().data.numpy() - tCodeValue)) / 2
            if return_params:
                return inputs, clusterCenters.cpu().data.numpy()
            else:
                return inputs

        loss.backward()
        optimizer.step()
        print np.sum(np.abs(torch.sign(output_adv).cpu().data.numpy() - tCodeValue)) / 2

        inputs.data[inputs.data < 0] = 0
        inputs.data[inputs.data > 1] = 1

    out = model(inputs)
    if return_params:
        return inputs, clusterCenters.cpu().data.numpy(), disToCenter_i
    else:
        return inputs


def get_trans_img(img, job_dataset):
    img = np.asarray(img)
    if img.shape[0] == 3:
        img = np.moveaxis(img, 0, -1)

    if job_dataset == 'mnist':
        return img
    if job_dataset == 'cifar10':
        return img
    if 'imagenet' in job_dataset:
        return img
    if job_dataset == 'fashion_mnist':
        return img
    if 'places365' in job_dataset:
        return img


def get_round_adv(adv__):
    # round the adv__
    img_rounded = torch.round(adv__ * 255) / 255
    return img_rounded


if __name__ == "__main__":
    job_dataset = 'cifar10'
    job_values = ['mnist', 'cifar10', 'fashion_mnist', 'imagenet', 'places365']
    net_values = ['ResNet50', 'ResNet152']
    net = 'ResNet50'

    # index of the image in test set
    index = 6
    # cluster-based method paramters
    K_value = 25
    eps = 1.5
    lr = 0.1
    # hamming maximum method paramters
    hdm_eps = 1.0
    threshold = 32

    adv_method = 'hdm'
    adv_method_value = ['hdm', 'cbwm', 'ori']

    from publicVariables import th_h, th_l

    # threshold = 47
    from publicVariables import iter_lists

    snapshot_path = '../snapshot/' + job_dataset + '_48bit_' + net + '_hashnet/'
    model_path = snapshot_path + 'iter_%05d_model.pth.tar' % (iter_lists[net][job_dataset])
    query_path = './save_for_load/blackbox/' + net + '/' + job_dataset + '_test_output_code_label.npz'
    database_path = './save_for_load/blackbox/' + net + '/' + job_dataset + '_database_output_code_label.npz'

    from publicFunctions import load_model_class

    model_dict_path = snapshot_path + 'iter_%05d_model_dict.pth.tar' % (iter_lists[net][job_dataset])
    model = load_model_class(net)
    model.load_state_dict(torch.load(model_dict_path))
    model = model.cuda().eval()

    dsets = get_dsets(job_dataset)
    dset_test = dsets['test']
    dset_database = dsets['database']

    tmp = np.load(database_path)
    output, code, multi_label = tmp['arr_0'], tmp['arr_1'], tmp['arr_2']

    tmp = np.load(query_path)
    output_test, code_test, multi_label_test = tmp['arr_0'], tmp['arr_1'], tmp['arr_2']

    img = dset_test[index][0]
    img_t = img
    targetCode = code_test[index]

    isKmeans = True

    if adv_method == 'hdm':
        adv__ = get_adv_untarget(model, img_t, targetCode, eps=hdm_eps / 255, threshold=threshold)
    elif adv_method == 'cbwm':
        if isKmeans:
            adv__ = get_adv_cluster_circle_loss(model, img_t, lr=lr / 255, code_test=code, \
                                                loss_fun=clusterCircleLoss, job_dataset=job_dataset, net = net, \
                                                K_value=K_value)
        else:
            adv__ = get_adv_cluster_circle_loss_dbscan(model, img_t, lr=lr / 255, code_test=code,
                                                       loss_fun=clusterCircleLoss, eps=eps, job_dataset=job_dataset , net = net)

    img__ = Variable(img_t).cuda().unsqueeze(0)
    img_num_by_class = get_img_num_by_class_from_img(img__, model, code, multi_label,
                                                     threshold=th_h[job_dataset]).astype(int)
    print 'Original Image:\n', img_num_by_class, np.sum(img_num_by_class[:], axis=1)
    ori_code = np.sign(model(img__).cpu().data.numpy())
    print 'Original Code Difference:\n', ori_code - targetCode

    img_num_by_class = get_img_num_by_class_from_img(adv__, model, code, multi_label,
                                                     threshold=th_h[job_dataset]).astype(int)
    print 'Adversarial Image:\n', img_num_by_class, np.sum(img_num_by_class[:], axis=1)

    plotResult = False
    if plotResult:
        X = np.array(img_t.unsqueeze(0))
        img_ori = get_trans_img(X[0], job_dataset)
        plt.figure(666)
        plt.subplot(2, 2, 1)
        plt.title('Original Image')
        plt.imshow(img_ori)
        plt.subplot(2, 2, 2)
        plt.title('Original Image')
        plt.imshow(img_ori.mean(axis=2), cmap='gray')

        img_adv = adv__.cpu().data.numpy()
        img_adv = get_trans_img(img_adv[0], job_dataset)
        plt.subplot(2, 2, 3)
        plt.title('Adversarial Image')
        plt.imshow(img_adv)
        plt.subplot(2, 2, 4)
        plt.title('Adversarial Image')
        plt.imshow(img_adv.mean(axis=2), cmap='gray')
