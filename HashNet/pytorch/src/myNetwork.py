# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import models
from torch.autograd import Variable
import math

import pretrainedmodels
import pretrainedmodels.utils as utils


class AlexNetFc(nn.Module):
  def __init__(self, hash_bit):
    super(AlexNetFc, self).__init__()
    model_alexnet = models.alexnet(pretrained=True)
    self.features = model_alexnet.features
    self.classifier = nn.Sequential()
    for i in range(6):
        self.classifier.add_module("classifier"+str(i), model_alexnet.classifier[i])
    self.feature_layers = nn.Sequential(self.features, self.classifier)

    self.hash_layer = nn.Linear(model_alexnet.classifier[6].in_features, hash_bit)
    self.hash_layer.weight.data.normal_(0, 0.01)
    self.hash_layer.bias.data.fill_(0.0)
    self.iter_num = 0
    self.__in_features = hash_bit
    self.step_size = 200
    self.gamma = 0.005
    self.power = 0.5
    self.init_scale = 1.0
    self.activation = nn.Tanh()
    self.scale = self.init_scale

  def forward(self, x):
    if self.training:
        self.iter_num += 1
    x = self.features(x)
    x = x.view(x.size(0), 256*6*6)
    x = self.classifier(x)
    y = self.hash_layer(x)
    if self.iter_num % self.step_size==0:
        self.scale = self.init_scale * (math.pow((1.+self.gamma*self.iter_num), self.power))
    y = self.activation(self.scale*y)
    return y

  def output_num(self):
    return self.__in_features

resnet_dict = {"ResNet18":models.resnet18, "ResNet34":models.resnet34, "ResNet50":models.resnet50, "ResNet101":models.resnet101, "ResNet152":models.resnet152} 
class ResNetFc(nn.Module):
  def __init__(self, name, hash_bit):
    super(ResNetFc, self).__init__()
    model_resnet = resnet_dict[name](pretrained=True)
    self.conv1 = model_resnet.conv1
    self.bn1 = model_resnet.bn1
    self.relu = model_resnet.relu
    self.maxpool = model_resnet.maxpool
    self.layer1 = model_resnet.layer1
    self.layer2 = model_resnet.layer2
    self.layer3 = model_resnet.layer3
    self.layer4 = model_resnet.layer4
    self.avgpool = model_resnet.avgpool
    self.feature_layers = nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool, \
                         self.layer1, self.layer2, self.layer3, self.layer4, self.avgpool)

    self.hash_layer = nn.Linear(model_resnet.fc.in_features, hash_bit)
    self.hash_layer.weight.data.normal_(0, 0.01)
    self.hash_layer.bias.data.fill_(0.0)
    self.iter_num = 0
    self.__in_features = hash_bit
    self.step_size = 200
    self.gamma = 0.005
    self.power = 0.5
    self.init_scale = 1.0
    self.activation = nn.Tanh()
    self.scale = self.init_scale

  def forward(self, x):
    if self.training:
        self.iter_num += 1
    x = self.feature_layers(x)
    x = x.view(x.size(0), -1)
    y = self.hash_layer(x)
    if self.iter_num % self.step_size==0:
        self.scale = self.init_scale * (math.pow((1.+self.gamma*self.iter_num), self.power))
    y = self.activation(self.scale*y)
    return y

  def output_num(self):
    return self.__in_features

vgg_dict = {"VGG11":models.vgg11, "VGG13":models.vgg13, "VGG16":models.vgg16, "VGG19":models.vgg19, "VGG11BN":models.vgg11_bn, "VGG13BN":models.vgg13_bn, "VGG16BN":models.vgg16_bn, "VGG19BN":models.vgg19_bn} 
class VGGFc(nn.Module):
  def __init__(self, name, hash_bit):
    super(VGGFc, self).__init__()
    model_vgg = vgg_dict[name](pretrained=True)
    self.features = model_vgg.features
    self.classifier = nn.Sequential()
    for i in range(6):
        self.classifier.add_module("classifier"+str(i), model_vgg.classifier[i])
    self.feature_layers = nn.Sequential(self.features, self.classifier)

    self.hash_layer = nn.Linear(model_vgg.classifier[6].in_features, hash_bit)
    self.hash_layer.weight.data.normal_(0, 0.01)
    self.hash_layer.bias.data.fill_(0.0)
    self.iter_num = 0
    self.__in_features = hash_bit
    self.step_size = 200
    self.gamma = 0.005
    self.power = 0.5
    self.init_scale = 1.0
    self.activation = nn.Tanh()
    self.scale = self.init_scale

  def forward(self, x):
    if self.training:
        self.iter_num += 1
    x = self.features(x)
    x = x.view(x.size(0), 25088)
    x = self.classifier(x)
    y = self.hash_layer(x)
    if self.iter_num % self.step_size==0:
        self.scale = self.init_scale * (math.pow((1.+self.gamma*self.iter_num), self.power))
    y = self.activation(self.scale*y)
    return y

  def output_num(self):
    return self.__in_features

#inceptionv3_dict = {"Inc_v3": pretrainedmodels.inceptionv3}
inceptionv3_dict = {"Inc_v3": models.inception_v3}


class InceptionV3(nn.Module):
  def __init__(self, name, hash_bit):
    super(InceptionV3, self).__init__()
    #model_inception = inceptionv3_dict[name](pretrained='imagenet')
    model_inception = inceptionv3_dict[name](pretrained=True)
    self.Conv2d_1a_3x3 = model_inception.Conv2d_1a_3x3
    self.Conv2d_2a_3x3 = model_inception.Conv2d_2a_3x3
    self.Conv2d_2b_3x3 = model_inception.Conv2d_2b_3x3
    self.Conv2d_3b_1x1 = model_inception.Conv2d_3b_1x1
    self.Conv2d_4a_3x3 = model_inception.Conv2d_4a_3x3
    self.Mixed_5b = model_inception.Mixed_5b
    self.Mixed_5c = model_inception.Mixed_5c
    self.Mixed_5d = model_inception.Mixed_5d
    self.Mixed_6a = model_inception.Mixed_6a
    self.Mixed_6b = model_inception.Mixed_6b
    self.Mixed_6c = model_inception.Mixed_6c
    self.Mixed_6d = model_inception.Mixed_6d
    self.Mixed_6e = model_inception.Mixed_6e
    self.Mixed_7a = model_inception.Mixed_7a
    self.Mixed_7b = model_inception.Mixed_7b
    self.Mixed_7c = model_inception.Mixed_7c
    
    
    #self.features = model_inception.features
    self.feature_layers = nn.Sequential(self.Conv2d_1a_3x3, self.Conv2d_2a_3x3, self.Conv2d_2b_3x3, self.Conv2d_3b_1x1, self.Conv2d_4a_3x3, self.Mixed_5b, self.Mixed_5c, self.Mixed_5d, self.Mixed_6a, self.Mixed_6b, self.Mixed_6c, self.Mixed_6d, self.Mixed_6e,  self.Mixed_7a, self.Mixed_7b, self.Mixed_7c)
    #self.feature_layers = nn.Sequential(self.features)

    #self.hash_layer = nn.Linear(model_inception.last_linear.in_features, hash_bit)
    self.hash_layer = nn.Linear(model_inception.fc.in_features, hash_bit)
    self.hash_layer.weight.data.normal_(0, 0.01)
    self.hash_layer.bias.data.fill_(0.0)
    self.iter_num = 0
    self.__in_features = hash_bit
    self.step_size = 200
    self.gamma = 0.005
    self.power = 0.5
    self.init_scale = 1.0
    self.activation = nn.Tanh()
    self.scale = self.init_scale

  def forward(self, x):
    if self.training:
        self.iter_num += 1
    x = self.feature_layers(x)
    x = x.view(x.size(0), -1)
    y = self.hash_layer(x)
    if self.iter_num % self.step_size==0:
        self.scale = self.init_scale * (math.pow((1.+self.gamma*self.iter_num), self.power))
    y = self.activation(self.scale*y)
    return y

  def output_num(self):
    return self.__in_features

inceptionv4_dict = { "Inc_v4": pretrainedmodels.inceptionv4}
class InceptionV4(nn.Module):
  def __init__(self, name, hash_bit):
    super(InceptionV4, self).__init__()
    model_inception = inceptionv4_dict[name](pretrained='imagenet')
    self.avg_pool = model_inception.avg_pool
    self.features = model_inception.features

    self.feature_layers = nn.Sequential(self.features, self.avg_pool)

    self.hash_layer = nn.Linear(model_inception.last_linear.in_features, hash_bit)
    self.hash_layer.weight.data.normal_(0, 0.01)
    self.hash_layer.bias.data.fill_(0.0)
    self.iter_num = 0
    self.__in_features = hash_bit
    self.step_size = 200
    self.gamma = 0.005
    self.power = 0.5
    self.init_scale = 1.0
    self.activation = nn.Tanh()
    self.scale = self.init_scale

  def forward(self, x):
    if self.training:
        self.iter_num += 1
    x = self.feature_layers(x)
    x = x.view(x.size(0), -1)
    y = self.hash_layer(x)
    if self.iter_num % self.step_size==0:
        self.scale = self.init_scale * (math.pow((1.+self.gamma*self.iter_num), self.power))
    y = self.activation(self.scale*y)
    return y

  def output_num(self):
    return self.__in_features


incResV2_dict = {"IncRes_v2": pretrainedmodels.inceptionresnetv2}
class IncResV2(nn.Module):
  def __init__(self, name, hash_bit):
    super(IncResV2, self).__init__()
    model_inception = inceptionv4_dict[name](pretrained='imagenet')
    self.avg_pool = model_inception.avg_pool
    self.features = model_inception.features

    self.feature_layers = nn.Sequential(self.features)

    self.hash_layer = nn.Linear(model_inception.last_linear.in_features, hash_bit)
    self.hash_layer.weight.data.normal_(0, 0.01)
    self.hash_layer.bias.data.fill_(0.0)
    self.iter_num = 0
    self.__in_features = hash_bit
    self.step_size = 200
    self.gamma = 0.005
    self.power = 0.5
    self.init_scale = 1.0
    self.activation = nn.Tanh()
    self.scale = self.init_scale

  def forward(self, x):
    if self.training:
        self.iter_num += 1
    x = self.feature_layers(x)
    x = x.view(x.size(0), -1)
    y = self.hash_layer(x)
    if self.iter_num % self.step_size==0:
        self.scale = self.init_scale * (math.pow((1.+self.gamma*self.iter_num), self.power))
    y = self.activation(self.scale*y)
    return y

  def output_num(self):
    return self.__in_features

resnext_dict = {"ResNext101_32x4d":pretrainedmodels.resnext101_32x4d, "ResNext101_64x4d":pretrainedmodels.resnext101_64x4d} 
class ResNext(nn.Module):
  def __init__(self, name, hash_bit):
    super(ResNext, self).__init__()
    model_resnext = resnext_dict[name](pretrained='imagenet')
    self.avg_pool = model_resnext.avg_pool
    self.features = model_resnext.features
    
    self.feature_layers = nn.Sequential(self.features, self.avg_pool)

    self.hash_layer = nn.Linear(model_resnext.last_linear.in_features, hash_bit)
    self.hash_layer.weight.data.normal_(0, 0.01)
    self.hash_layer.bias.data.fill_(0.0)
    self.iter_num = 0
    self.__in_features = hash_bit
    self.step_size = 200
    self.gamma = 0.005
    self.power = 0.5
    self.init_scale = 1.0
    self.activation = nn.Tanh()
    self.scale = self.init_scale

  def forward(self, x):
    if self.training:
        self.iter_num += 1
    x = self.feature_layers(x)
    x = x.view(x.size(0), -1)
    y = self.hash_layer(x)
    if self.iter_num % self.step_size==0:
        self.scale = self.init_scale * (math.pow((1.+self.gamma*self.iter_num), self.power))
    y = self.activation(self.scale*y)
    return y

  def output_num(self):
    return self.__in_features