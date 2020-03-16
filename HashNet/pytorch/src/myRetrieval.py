# -*- coding: utf-8 -*-


import numpy as np
import torch

from data_list import ImageList
from torch.autograd import Variable

from myEvaluation import make_one_hot

import matplotlib.pyplot as plt


def show_img(path):
    img = plt.imread(path)
    plt.imshow(img)


def get_multi_labels_from_vector(multi_label):
    img_num = multi_label.shape[0]
    multi_labels = []
    for i in range(img_num):
        multi_labels.append(np.argwhere(multi_label[i, :] == 1))
    return multi_labels


def plot_by_result(query_result_with_index, mnist_train):
    for i in range(min(query_result_with_index.size, 200)):
        img_index = query_result_with_index[i][0]
        plt.subplot(10, 20, i + 1)
        plt.imshow(np.asarray(mnist_train[img_index][0]))


def cal_intra_class_dis(code, multi_label_one_hot):
    num_class = multi_label_one_hot.shape[1]
    avg_dis_class = np.zeros([num_class, num_class])
    # hamming_dis_stat: 
    # mean, min, max, var 
    hamming_dis_stat = np.zeros([num_class, 4])
    hamming_dis_class = []
    class_index = []
    # optional: let all multi-label imgs invisible
    #multi_label[multi_label.sum(axis=1)>1] = 0 
    print num_class
    for i in range(num_class):
        class_i_index = np.argwhere(multi_label_one_hot[:, i]==1).reshape([-1])
        class_index.append(class_i_index)
        class_i_code = code[class_i_index, :]

        for j in range(num_class):
            class_j_index = np.argwhere(multi_label_one_hot[:, j]==1).reshape([-1])
            class_j_code = code[class_j_index, :]
            hamming_dis = np.matmul(class_i_code, class_j_code.transpose()) * (-0.5) + 24
            if i == j:
                hamming_dis_class.append(hamming_dis)

            avg_dis_class[i, j] = np.mean(hamming_dis)
    return avg_dis_class, hamming_dis_stat, class_index

def get_retrieval_result_by_query_code(query_code, database_code, threshold = 2):
    """

    Args:
        query_code (numpy.ndarray) - hash code of querying images
        database_code (numpy.ndarray) - hash code in the database
        threshold (int, optional) - querying threshold

    Returns:
        query_result (numpy.ndarray): the index of querying results of each query image
    """
    if len(query_code.shape) >1:
        query_size = query_code.shape[0]
    else:
        query_size = 1
        query_code = np.expand_dims(query_code)
    query_result = []
    hamming_dis = np.matmul(query_code, database_code.transpose()) * -0.5 + 24
    for i in range(query_size):
        matched_index = np.argwhere(hamming_dis[i]<=threshold)
        query_result.append(matched_index)
    return np.array(query_result)

def count_by_query_result(query_result, labels):
    """

    Args:
        query_result (numpy.ndarray) - the index of querying results of each query image
        labels (numpy.ndarray) - labels

    Returns:
        img_num_by_class (numpy.ndarray): number of querying results of each class
    """
    img_num = query_result.shape[0]
    img_num_by_class = np.zeros([img_num, int(labels.max() + 1)])
    if len(query_result.shape) == 3:
        if query_result.size == 0:
            return np.zeros([img_num, int(labels.max()) + 1])
        query_result = query_result.reshape([-1, query_result.shape[1]])

    for i in range(img_num):
        for j in range(int(labels.max() + 1)):
            img_num_by_class[i, j] = np.argwhere(labels[query_result[i]] == j).shape[0]
    return img_num_by_class

def get_query_result_num_by_class(query_code, database_code, labels, threshold = 10):
    """

    Args:
        query_code (numpy.ndarray) - hash code of querying images
        database_code (numpy.ndarray) - hash code in the database
        labels (numpy.ndarray) - labels
        threshold (int, optional) - querying threshold

    Returns:
        img_num_by_class (numpy.ndarray): number of querying results of each class
    """
    query_result = get_retrieval_result_by_query_code(query_code, database_code, threshold = threshold)
    img_num_by_class = count_by_query_result(query_result, labels)
    return img_num_by_class

def get_img_num_by_class_from_img(img, model, database_code, labels, threshold = 10):
    """

    Args:
        img(torch.Variable) - input image
        model() - a HashNet model
        database_code (numpy.ndarray) - hash code in the database
        labels (numpy.ndarray) - labels
        threshold (int, optional) - querying threshold

    Returns:
        img_num_by_class (numpy.ndarray): number of querying results of each class
    """
    oCodeValue = np.sign(model(img).cpu().data.numpy())
    return get_query_result_num_by_class(oCodeValue, database_code, labels, threshold)

def get_imgs_num_by_class_from_NPimgs(img_np, model, database_code, labels, threshold = 10):
    """

    Args:
        img_np(np.ndarray) - input image
        model() - a HashNet model
        database_code (numpy.ndarray) - hash code in the database
        labels (numpy.ndarray) - labels
        threshold (int, optional) - querying threshold

    Returns:
        img_num_by_class (numpy.ndarray): number of querying results of each class
    """
    query_size = img_np.shape[0]
    imgs_num_by_class = np.zeros([query_size, int(labels.max() + 1)])
    for i in range(query_size):
        img = Variable(torch.Tensor(img_np[i:i+1])).cuda()
        imgs_num_by_class[i] = get_img_num_by_class_from_img(img, model, database_code, labels, threshold = threshold)
    return imgs_num_by_class

    
def load_ori_dsets(job_dataset):
    # Load the original dsets that have no transforms
    dsets = {}

    if job_dataset == 'mnist':
        import torchvision.datasets as dset
        root = '../../../../data/mnist'
        dsets["test"] = dset.MNIST(root=root, train=False, download=True)
        dsets["database"] = dset.MNIST(root=root, train=True, download=True)
        dsets["train"] = dsets["database"]
    if job_dataset == 'cifar10':
        import torchvision.datasets as dset
        root = '../../../../data/cifar10/pytorch_path'
        dsets['test'] = dset.CIFAR10(root=root, train=False, download=True)
        dsets['database'] = dset.CIFAR10(root=root, train=True, download=True)
        dsets["train"] = dsets["database"]
    if job_dataset == 'fashion_mnist':
        import torchvision.datasets as dset
        root = '../../../../data/mnist'
        dsets["test"] = dset.FashionMNIST(root=root, train=False, download=True)
        dsets["database"] = dset.FashionMNIST(root=root, train=True, download=True)
        dsets["train"] = dsets["database"]
    if 'imagenet' in job_dataset:
        dsets["train"] = ImageList(open('../data/imagenet/train.txt').readlines())
        dsets["test"] = ImageList(open('../data/imagenet/test.txt').readlines())
        dsets["database"] = ImageList(open('../data/imagenet/database.txt').readlines())
    if job_dataset == 'places365':
        import torchvision.datasets as dset
        dsets["database"] = ImageList(open("../data/places365_standard/database.txt").readlines())
        dsets['test'] = ImageList(open("../data/places365_standard/val.txt").readlines())

    return dsets


if __name__ == "__main__": 
    job_dataset = 'places365'
    threshold = 10
    job_values = ['mnist', 'cifar10', 'fashion_mnist', 'places365', 'imagenet']
    net_values = ['ResNet18','ResNet34', 'AlexNet']#
    net = 'ResNet152'

    from publicVariables import iter_lists
    snapshot_path = '../snapshot/'+job_dataset+'_48bit_'+ net +'_hashnet/'
    model_path = snapshot_path + 'iter_%5d_model.pth.tar'%(iter_lists[net][job_dataset])
    query_path = './save_for_load/blackbox/'+net+'/'+job_dataset+'_test_output_code_label.npz'
    database_path = './save_for_load/blackbox/'+net+'/'+job_dataset+'_database_output_code_label.npz'


        

    from publicFunctions import load_model_class
    model_dict_path = snapshot_path + 'iter_%5d_model_dict.pth.tar'%(iter_lists[net][job_dataset])
    model = load_model_class(net)
    model.load_state_dict(torch.load(model_dict_path))
    model = model.cuda().eval()

    dsets = load_ori_dsets(job_dataset)
    dset_test = dsets['test']
    dset_database = dsets['database']
    tmp = np.load(database_path)
    output, code, multi_label = tmp['arr_0'], tmp['arr_1'], tmp['arr_2']


    tmp = np.load(query_path)
    query_output, query_code, query_multi_label = tmp['arr_0'], tmp['arr_1'], tmp['arr_2']

    # use C=100 for ImageNet and C=36 for places365
    multi_label_one_hot = make_one_hot(multi_label, C=int(multi_label.max()+1))
    avg_dis_class, hamming_dis_class, class_index = cal_intra_class_dis(code[:], multi_label_one_hot[:])    
    avg_intra_dis = np.array([avg_dis_class[ii][ii] for ii in range(avg_dis_class.shape[0])])

    query_result = get_retrieval_result_by_query_code(query_code[:], code, threshold)
    
    query_result_count = np.array([ multi_label[query_result[i]].shape[0] for i in range(len(query_result))])
    query_precisions = np.array([ np.sum(multi_label[query_result[i]]==query_multi_label[i]).astype(float)/multi_label[query_result[i]].size for i in range(len(query_result))])
    query_precisions = np.nan_to_num(query_precisions)
    print 'query_precision:', query_precisions.mean()
    a = np.array([multi_label[query_result[i]].size for i in range(len(query_result))])
    print 'Number of images without results:', (a == 0).sum()
    weighted_query_precision = np.sum(query_result_count.astype(float) * query_precisions) / query_result_count.sum()
    print 'weighted query precision:%f'%(weighted_query_precision)
    print 'avg query_result_count:%f'%(query_result_count.mean())
    '''
    # for debug
    abno_acc = np.argwhere(np.array(query_accuracys) < 0.9).reshape([-1])
    plt.figure(1)
    for i in range(min(abno_acc.size, 200)):
        img_index = abno_acc[i]
        plt.subplot(10,20,i+1)
        tmp = np.asarray(dset_test[img_index][0])
        tmp = get_trans_img(tmp, job_dataset)
        plt.imshow(tmp)
    '''
    
    # For ploting the queried results
    '''
    plt.figure(2)
    index = 0
    #for i in range(query_result[index].shape[0]):
    for i in range(min(query_result[index].size, 200)):
        img_index = query_result[index][i][0]
        plt.subplot(10,20,i+1)
        tmp = np.moveaxis(np.asarray(dset_database[img_index][0]), 0, -1)
        tmp = get_trans_img(tmp, job_dataset)
        plt.imshow(tmp)
    '''
    '''
    show_img(img_list[index])
    for i in range(query_result[index].size):
        img_index = query_result[index][i][0]
        plt.subplot(4,5,i+1)
        show_img(img_list[img_index])
        
    # For get the queried results' labels
    query_label = multi_label[query_result[index].reshape([-1])]
    query_labels = get_multi_labels_from_vector(query_label)
    '''
    
    
    
    
    