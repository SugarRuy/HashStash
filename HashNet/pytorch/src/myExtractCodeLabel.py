# -*- coding: utf-8 -*-


import os
import numpy as np
import torch
import torch.utils.data as util_data

from data_list import ImageList
from torch.autograd import Variable

from myTrain import trans_train
import torchvision.datasets as dset

# initial Hashbit
Hashbit = 0


def trans_train_resize_mnist(resize=224):
    # Deprecated. Using get_dsets_loader_by_dataset(job_dataset) instead
    # It is still kept because of the historical reason.
    return trans_train(resize, 0, 'mnist')


def trans_train_resize_cifar10(resize=224):
    # Deprecated. Using get_dsets_loader_by_dataset(job_dataset) instead
    # It is still kept because of the historical reason.
    return trans_train(resize, 0, 'cifar10')


def trans_train_resize_imagenet(resize=224):
    # Deprecated. Using get_dsets_loader_by_dataset(job_dataset) instead
    # It is still kept because of the historical reason.
    return trans_train(resize, 0, 'imagenet')


def get_dsets_loader_by_dataset(job_dataset):
    # Parameters:
    #   job_dataset (str) -  name of dataset
    # Returns:
    #   Tuple(dsets, dsets_loaders).
    #   dsets is a dict containing the train and the test datasets
    #   dsets_loader is a dict containing the train and the test dataset loaders
    # Return type:
    #   tuple
    prep_dict = {}
    dsets = {}
    dset_loaders = {}

    prep_dict["database"] = trans_train(job_dataset)
    prep_dict["train"] = trans_train(job_dataset)
    prep_dict["test"] = trans_train(job_dataset)

    root = '../../../../data/%s' % (job_dataset)

    #
    if job_dataset == 'mnist':
        dsets["test"] = dset.MNIST(root=root, train=False, transform=prep_dict["test"], download=True)
        dsets["database"] = dset.MNIST(root=root, train=True, transform=prep_dict["database"], download=True)
        dsets["train"] = dsets["database"]
    elif job_dataset == 'fashion_mnist':
        dsets["test"] = dset.FashionMNIST(root=root, train=False, transform=prep_dict["test"], download=True)
        dsets["database"] = dset.FashionMNIST(root=root, train=True, transform=prep_dict["database"], download=True)
        dsets["train"] = dsets["database"]
    elif job_dataset == 'cifar10':
        dsets["test"] = dset.CIFAR10(root=root, train=False, transform=prep_dict["test"], download=True)
        dsets["database"] = dset.CIFAR10(root=root, train=True, transform=prep_dict["database"], download=True)
        dsets["train"] = dsets["database"]

    else:
        from publicVariables import data_list_path

        dsets["test"] = ImageList(open(data_list_path[job_dataset]["test"]).readlines(), \
                                  transform=prep_dict["test"])
        dsets["database"] = ImageList(open(data_list_path[job_dataset]["database"]).readlines(), \
                                      transform=prep_dict["database"])

    dset_loaders["test"] = util_data.DataLoader(dsets["test"], \
                                                batch_size=1, \
                                                shuffle=False, num_workers=16)

    dset_loaders["database"] = util_data.DataLoader(dsets["database"], \
                                                    batch_size=1, \
                                                    shuffle=False, num_workers=16)
    return dsets, dset_loaders


def get_dsets_loader(mode='test'):
    # Deprecated. Using get_dsets_loader_by_dataset(job_dataset) instead
    # It is still kept because of the historical reason.  
    dsets, dset_loaders = get_dsets_loader_by_dataset('mnist')
    return dsets, dset_loaders


def get_dsets_loader_coco(mode='test'):
    # Deprecated. Using get_dsets_loader_by_dataset(job_dataset) instead
    # It is still kept because of the historical reason.
    dsets, dset_loaders = get_dsets_loader_by_dataset('coco')
    return dsets, dset_loaders


def get_dsets_loader_cifar10(mode='test'):
    # Deprecated. Using get_dsets_loader_by_dataset(job_dataset) instead
    # It is still kept because of the historical reason.
    dsets, dset_loaders = get_dsets_loader_by_dataset('cifar10')
    return dsets, dset_loaders


def get_dsets_loader_imagenet(mode='test'):
    # Deprecated. Using get_dsets_loader_by_dataset(job_dataset) instead
    # It is still kept because of the historical reason.
    dsets, dset_loaders = get_dsets_loader_by_dataset('imagenet')
    return dsets, dset_loaders


def get_dsets_loader_fashion_mnist(mode='test'):
    # Deprecated. Using get_dsets_loader_by_dataset(job_dataset) instead
    # It is still kept because of the historical reason.
    dsets, dset_loaders = get_dsets_loader_by_dataset('fashion_mnist')
    return dsets, dset_loaders


def get_dsets_loader_nus_wide(mode='test'):
    # Deprecated. Using get_dsets_loader_by_dataset(job_dataset) instead
    # It is still kept because of historical reason.
    dsets, dset_loaders = get_dsets_loader_by_dataset('nus_wide')
    return dsets, dset_loaders


def get_dsets_loader_places365(mode='test'):
    # Deprecated. Using get_dsets_loader_by_dataset(job_dataset) instead
    # It is still kept because of historical reason.
    dsets, dset_loaders = get_dsets_loader_by_dataset('places365')
    return dsets, dset_loaders


def get_output_code_label_list_by_dset(dsets, dset_loaders, model, mode='test'):
    """

    Args:
        dsets: dict containing datasets of test and database;
        dset_loaders: dict containing dataset loaders of test and database;
        model: HashNet model;
        mode: 'test' or 'database'.

    Returns:
        Tuple(all_output, all_label)
        all_output: original output of the dataset of the model - (numpy.ndarray)
        all_label: one-hot labels of the dataset - (numpy.ndarray)
    """

    global Hashbit
    HashBit = Hashbit
    len_loader = len(dset_loaders[mode])

    print 'Total number of %s images: %d' % (mode, len_loader)

    all_output = np.zeros([len_loader, HashBit])
    all_label = np.zeros([len_loader])
    for i in range(len_loader):
        print i
        data = dsets[mode][i]
        inputs = torch.unsqueeze(data[0], 0)
        labels = data[1]
        inputs = Variable(inputs.cuda())
        outputs = model(inputs)
        all_output[i] = outputs.cpu().data.float()
        if not isinstance(labels, int):
            all_label[i] = np.argmax(labels)
        else:
            all_label[i] = labels

    all_output = torch.Tensor(all_output)
    all_label = torch.Tensor(all_label)

    return all_output, all_label


if __name__ == "__main__":
    # Step 2: Extract Code and One-hot Label
    # Extract the Hashcode of different job_datasets by different hashnet models
    # Then save them into ./save_for_load/blackbox/

    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    job_dataset = 'places365'
    global Hashbit
    Hashbit = 48
    net = 'ResNet152'

    from publicVariables import iter_lists

    snapshot_iter = iter_lists[net][job_dataset]
    snapshot_path = '../snapshot/%s_48bit_%s_hashnet/' % (job_dataset, net)
    model_path = snapshot_path + 'iter_%05d_model.pth.tar' % (snapshot_iter)
    black_dir_path = './save_for_load/blackbox/%s/' % (net)
    test_save_path = '%s/%s_test_output_code_label' % (black_dir_path, job_dataset)
    database_save_path = '%s/%s_database_output_code_label' % (black_dir_path, job_dataset)

    if not os.path.exists(black_dir_path):
        os.makedirs(black_dir_path)

    from publicFunctions import load_model_class

    model_dict_path = snapshot_path + 'iter_%5d_model_dict.pth.tar' % (iter_lists[net][job_dataset])
    model = load_model_class(net)
    model.load_state_dict(torch.load(model_dict_path))
    model = model.cuda().eval()

    # get
    dsets, dset_loaders = get_dsets_loader_by_dataset(job_dataset)

    MODE = 'test'

    # only take the batchsize = 1
    all_output, all_label = get_output_code_label_list_by_dset(dsets, dset_loaders, model, mode=MODE)
    output = all_output.cpu().numpy()
    label = all_label.cpu().numpy()
    output_code = np.sign(output)
    np.savez(test_save_path, output, output_code, label)

    MODE = 'database'

    all_output, all_label = get_output_code_label_list_by_dset(dsets, dset_loaders, model, mode=MODE)
    output = all_output.cpu().numpy()
    label = all_label.cpu().numpy()
    output_code = np.sign(output)
    np.savez(database_save_path, output, output_code, label)
