# -*- coding: utf-8 -*-

import os
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim


def transform_adv_img(adv__, job_dataset):
    # transform the adv__ to the same size of img
    if job_dataset == 'mnist':
        img = F.upsample(adv__, size=(28, 28), mode='bilinear')
        img = img.mean(dim=1).unsqueeze(1)
    if job_dataset == 'cifar10':
        #img = F.upsample(adv__, size=(32, 32), mode='bilinear')
        img = adv__
    if job_dataset == 'imagenet':
        img = adv__
    return img

def adv_class_verify(adv__, label, job_dataset):
    
    if job_dataset == 'mnist':
        model_path = 'save_for_load/mnist_lenet_model.pt'
        
    if job_dataset == 'cifar10':
        model_path = 'save_for_load/cifar10_vgg11_model.pt'
    
    
    model_class = torch.load(model_path)
    model_class = model_class.eval()
    
    img = transform_adv_img(adv__, job_dataset)
    img_out = model_class(img)
    label_pred = img_out.cpu().data.numpy().argmax()
    if label_pred == label:
        return True
    else:
        return False
    # adv__ may not have the same size as what the model can take,
    # so it needs to be transformed to the same size
    
        
def adv_class_verify_mnist(img, label):
    return adv_class_verify(img, label, 'mnist')

def adv_class_verify_cifar10(img, label):
    return adv_class_verify(img, label, 'cifar10')
    
    