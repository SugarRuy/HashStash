# -*- coding: utf-8 -*-


def get_dsets(job_dataset):
    import torchvision.datasets as dset
    dsets = {}
    if job_dataset == 'mnist':
        from myExtractCodeLabel import trans_train_resize_mnist
        root = '../../../../data/mnist'
        trans = trans_train_resize_mnist()
        dsets['test'] = dset.MNIST(root=root, train=False, transform=trans, download=True)
        dsets['database'] = dset.MNIST(root=root, train=True, transform=trans, download=True)
    if job_dataset == 'cifar10':
        from myExtractCodeLabel import trans_train_resize_cifar10
        root = '../../../../data/cifar10'
        trans = trans_train_resize_cifar10()
        dsets['test'] = dset.CIFAR10(root=root, train=False, transform=trans, download=True)
        dsets['database'] = dset.CIFAR10(root=root, train=True, transform=trans, download=True)
    if 'imagenet' in job_dataset:
        from myExtractCodeLabel import get_dsets_loader_imagenet
        dsets, dset_loaders = get_dsets_loader_imagenet()
    if job_dataset == 'fashion_mnist':
        from myExtractCodeLabel import trans_train_resize_mnist
        root = '../../../../data/fashion_mnist'
        trans = trans_train_resize_mnist()
        dsets['test'] = dset.FashionMNIST(root=root, train=False, transform=trans, download=True)
        dsets['database'] = dset.FashionMNIST(root=root, train=True, transform=trans, download=True)
    if 'nus_wide' in job_dataset:
        from myExtractCodeLabel import get_dsets_loader_nus_wide
        dsets, dset_loaders = get_dsets_loader_nus_wide()
    if 'places365' in job_dataset:
        from myExtractCodeLabel import get_dsets_loader_places365
        dsets, dset_loaders = get_dsets_loader_places365()

    return dsets


def load_model_class(net, hash_bit=48):
    import myNetwork
    config = {"hash_bit": hash_bit, "network": {}}

    if "ResNet" in net:
        config["network"]["type"] = myNetwork.ResNetFc
        config["network"]["params"] = {"name": net, "hash_bit": config["hash_bit"]}
    elif "VGG" in net:
        config["network"]["type"] = myNetwork.VGGFc
        config["network"]["params"] = {"name": net, "hash_bit": config["hash_bit"]}
    elif "AlexNet" in net:
        config["network"]["type"] = myNetwork.AlexNetFc
        config["network"]["params"] = {"hash_bit": config["hash_bit"]}
    elif "ResNext" in net:
        config["network"]["type"] = myNetwork.ResNext
        config["network"]["params"] = {"name": net, "hash_bit": config["hash_bit"]}
    elif "Inc_v3" in net:
        config["network"]["type"] = myNetwork.Inception
        config["network"]["params"] = {"name": net, "hash_bit": config["hash_bit"]}
    elif "Inc_v4" in net:
        config["network"]["type"] = myNetwork.Inc_v4
        config["network"]["params"] = {"name": net, "hash_bit": config["hash_bit"]}
    elif "IncRes_v2" in net:
        config["network"]["type"] = myNetwork.IncRes_v2
        config["network"]["params"] = {"name": net, "hash_bit": config["hash_bit"]}

    net_config = config["network"]
    base_network = net_config["type"](**net_config["params"])
    return base_network


def make_one_hot(labels, C=10):
    import numpy as np
    one_hot = np.zeros([labels.shape[0], C])
    for i in range(labels.shape[0]):
        one_hot[i, labels[i].astype("uint16")] = 1
    return one_hot
