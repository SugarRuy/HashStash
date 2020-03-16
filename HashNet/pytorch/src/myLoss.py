# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F



def pairwise_loss(outputs1, outputs2, label1, label2, sigmoid_param=1.0, l_threshold=15.0, class_num=1.0):
    similarity = Variable(torch.mm(label1.data.float(), label2.data.float().t()) > 0).float()
    dot_product = sigmoid_param * torch.mm(outputs1, outputs2.t())
    exp_product = torch.exp(dot_product)
    mask_dot = dot_product.data > l_threshold
    mask_exp = dot_product.data <= l_threshold
    mask_positive = similarity.data > 0
    mask_negative = similarity.data <= 0
    mask_dp = mask_dot & mask_positive
    mask_dn = mask_dot & mask_negative
    mask_ep = mask_exp & mask_positive
    mask_en = mask_exp & mask_negative

    dot_loss = dot_product * (1-similarity)
    exp_loss = (torch.log(1+exp_product) - similarity * dot_product)
    loss = (torch.sum(torch.masked_select(exp_loss, Variable(mask_ep))) + torch.sum(torch.masked_select(dot_loss, Variable(mask_dp)))) * class_num + torch.sum(torch.masked_select(exp_loss, Variable(mask_en))) + torch.sum(torch.masked_select(dot_loss, Variable(mask_dn)))

    return loss / (torch.sum(mask_positive.float()) * class_num + torch.sum(mask_negative.float()))

def masked_adv_loss(advOutput, imgCode, threshold = 0):
    mask = torch.abs(advOutput - imgCode) < (1+threshold)
    m = torch.sum(mask == 1).cpu().data.numpy().astype(float)[0]
    if m == 0:
        return Variable(torch.Tensor([0]))
    advOutput_masked = torch.masked_select(advOutput, mask)
    imgCode_masked = torch.masked_select(imgCode, mask)
    return torch.pow(torch.dot(advOutput_masked, imgCode_masked) / m + 1, 2)

def cal_class_code_center(code, multi_label_one_hot):
    num_class = multi_label_one_hot.shape[1]
    hashbit = code.shape[1]
    imgCenterCodeListByClass = np.zeros([num_class, hashbit])
    
    for i in range(num_class):
        class_i_index = np.argwhere(multi_label_one_hot[:, i]==1).reshape([-1])
        class_i_code = code[class_i_index, :]
        imgCenterCodeListByClass[i] = class_i_code.mean(axis=0)
    return imgCenterCodeListByClass

def cal_class_code_variance(code, multi_label_one_hot):
    num_class = multi_label_one_hot.shape[1]
    hashbit = code.shape[1]
    imgVarianceCodeListByClass = np.zeros([num_class, hashbit])
    
    for i in range(num_class):
        class_i_index = np.argwhere(multi_label_one_hot[:, i]==1).reshape([-1])
        class_i_code = code[class_i_index, :]
        imgVarianceCodeListByClass[i] = class_i_code.var(axis=0)
    return imgVarianceCodeListByClass


def emptySpaceLoss(advOutput, imgCenterCodeListByClass, threshold_retrieval = 10):
    # advCode: 
    # imgCenterCodeListByClass: [10 x hashbit]
    m = imgCenterCodeListByClass.shape[1]
    disByClass = torch.pow(torch.mm(advOutput, imgCenterCodeListByClass.t()) / m + 1, 2)
    dis_mean = torch.mean(disByClass)
    return dis_mean

def emptySpaceLoss_weighted(advOutput, imgCenterCodeListByClass, imgVarianceCodeListByClass, threshold_retrieval = 10):
    # imgCenterCodeListByClass: [10 x hashbit]
    m = imgCenterCodeListByClass.shape[1]
    imgVarianceCodeListByClass_weighted = imgVarianceCodeListByClass / imgVarianceCodeListByClass.sum()
    disByClass = torch.pow(torch.mm(advOutput, imgCenterCodeListByClass.t()) / m + 1, 2)
    dis_mean = torch.mean(torch.mm(disByClass, imgVarianceCodeListByClass_weighted)) * m
    return dis_mean

def emptySpaceLoss_away(advOutput, imgCenterCodeListByClass, imgVarianceCodeListByClass, threshold_retrieval = 16):
    # imgCenterCodeListByClass: [10 x hashbit]
    
    m = 48
    center = imgCenterCodeListByClass
    
    dis_center = torch.sum(torch.abs(advOutput - center), dim=1) / 2 
    index_center = np.argwhere(dis_center.cpu().data.numpy() < threshold_retrieval).reshape([-1])
    disByClass = torch.pow(torch.mm(advOutput, center.t()) / m + 1, 2)
    print 'index_center', index_center
    #print 'dis_center', dis_center.cpu().data.numpy()
    if index_center.size == 0:
        dis_mean = torch.mean(disByClass)
        return Variable(torch.Tensor([0]).cuda(), requires_grad=True)
    else:
        #dis_mean = torch.mean(disByClass)
        dis_mean = torch.mean(torch.index_select(disByClass, 1, Variable(torch.LongTensor(index_center)).cuda() ) )
    return dis_mean


def emptySpaceLoss_3sigma(advOutput, imgCenterCodeListByClass, imgVarianceCodeListByClass, threshold_retrieval = 16):
    # imgCenterCodeListByClass: [10 x hashbit]
    
    m = 48
    sigma = imgVarianceCodeListByClass
    mu = imgCenterCodeListByClass
    boarder_add = mu + sigma * 3
    boarder_minus = mu - sigma * 3 
    boarder = torch.cat((boarder_add, boarder_minus))
    
    dis_boarder = torch.sum(torch.abs(advOutput - boarder), dim=1) / 2 
    boarder_index = np.argwhere(dis_boarder.cpu().data.numpy() < threshold_retrieval).reshape([-1])
    disByClass = torch.pow(torch.mm(advOutput, boarder.t()) / m + 1, 2)
    print 'boarder_index', boarder_index
    print 'disByClass', disByClass.cpu().data.numpy()
    print 'dis_boarder', dis_boarder.cpu().data.numpy()
    if boarder_index.size == 0:
        print 'size0'
        dis_mean = torch.mean(disByClass)
        return Variable(torch.Tensor([0]).cuda(), requires_grad=True)
    else:
        #dis_mean = torch.mean(disByClass)
        dis_mean = torch.mean(torch.index_select(disByClass, 1, Variable(torch.LongTensor(boarder_index)).cuda() ) )
    return dis_mean

def clusterCentersLoss(advOutput, clusterCenters, var_lambda=0.5):
    # first edition of clusterCenterLoss
    m = 48

    lossToClusters = torch.pow(torch.mm(advOutput, clusterCenters.t()) / m + 1, 2)
    loss_mean = torch.mean(lossToClusters)
    # just for check    
    disToClusters = 0.5 * (m - torch.mm(advOutput, clusterCenters.t() ) ).cpu().data.numpy() 
    #print disToClusters
    if disToClusters.min() > 18.0:
        
        return Variable(torch.Tensor([0]).cuda())
        
    return loss_mean
    
def clusterCircleLoss(advOutput, clusterCenters, poly_params, threshold_retrieval):
    # This loss function is not doable. The reason is that the gradient can not be propagated to the inputs 
    m = 48
    disToCenters = torch.sum(torch.abs(advOutput - clusterCenters), dim = -1) / 2 - threshold_retrieval
    disParams = torch.stack([disToCenters ** i for i in range(poly_params.shape[1]-1,-1,-1)], dim=1)
    poly_params = Variable(torch.Tensor(poly_params.T)).cuda()
    mm_result = torch.mm(disParams, poly_params)
    out = torch.diag(mm_result)
    print disToCenters.cpu().data.numpy()
    #print out.cpu().data.numpy()
    # print out
    #print disToCenters.shape, disParams.shape, poly_params.shape
    #print torch.mm(disParams, poly_params).shape
    # out = torch.gather(clusterCircleNums, 1, torch.reshape(disToCentersIndex.long(), [-1, 1]))

    '''
    # Try a basic method
    var = []
    for i in range(clusterCenters.shape[0]):
        index = Variable(torch.LongTensor([i])).cuda()
        a_row = torch.index_select(clusterCircleNums, 0, index)
        index_row = torch.index_select(disToCentersIndex.long(), 0, index)
        print index, index_row
        print a_row.shape
        var[i] = torch.index_select(a_row, -1 ,index_row)
   
    
    # Try another basic method
    B = zip(clusterCircleNums, disToCentersIndex)
    for a, i in B:
        print a.shape, i.shape
    out = torch.cat([torch.index_select(a, 0, i) for a, i in zip(clusterCircleNums, disToCentersIndex.long())])
    print out
    '''
    '''
    # Try another basic method again
    C = Variable(torch.arange(0, 19999, 1000).cuda(), requires_grad=True).long() + disToCentersIndex.long()
    out = torch.index_select(clusterCircleNums.view(-1), 0, C)
    print clusterCircleNums.requires_grad, clusterCircleNums.view(-1).requires_grad,C.requires_grad, out.requires_grad
    '''
    return torch.sum(out)/1

def tanhfunc_torch(x, tanh_param):
    a, b, c, d = tanh_param[0], tanh_param[1], tanh_param[2], tanh_param[3]
    return d*F.tanh(a*x+b) + c

def clusterCircleLoss_tanh(advOutput, clusterCenters, tanh_params, threshold_retrieval):
    m = 48
    tanh_params = Variable(torch.Tensor(tanh_params).cuda())
    disToCenters = torch.sum(torch.abs(advOutput - clusterCenters), dim = -1) / 2 - threshold_retrieval
    disResult = torch.stack([tanhfunc_torch(disToCenters[i], tanh_params[i]) for i in range(tanh_params.shape[0])], dim=1)
    out = torch.sum(disResult)
    print disToCenters.cpu().data.numpy()
    print disResult.cpu().data.numpy()
    return out

def expfunc_torch(x, exp_param):
    a, b, c= exp_param[0], exp_param[1], exp_param[2]
    if x.cpu().data.numpy()>0:
        return torch.exp(a*x+b) + c
    else:
        return torch.exp(b)+c
    
def clusterCircleLoss_exp(advOutput, clusterCenters, exp_params, threshold_retrieval):
    m = 48
    exp_params = Variable(torch.Tensor(exp_params).cuda())
    disToCenters = torch.sum(torch.abs(advOutput - clusterCenters), dim = -1) / 2 - threshold_retrieval
    disResult = torch.stack([expfunc_torch(disToCenters[i], exp_params[i]) for i in range(exp_params.shape[0])], dim=1)
    out = torch.sum(disResult)
    print disToCenters.cpu().data.numpy()
    print disResult.cpu().data.numpy()
    return out

def clusterCircleLoss_lambda(advOutput, clusterCenters, tCode, poly_params, threshold_retrieval, var_lambda):
    # This loss function is not doable. The reason is that the gradient can not be propagated to the inputs 
    m = 48
    disToCenters = torch.sum(torch.abs(advOutput - clusterCenters), dim = -1) / 2 - threshold_retrieval
    disParams = torch.stack([disToCenters ** i for i in range(poly_params.shape[1]-1,-1,-1)], dim=1)
    poly_params = Variable(torch.Tensor(poly_params.T)).cuda()
    mm_result = torch.mm(disParams, poly_params)
    out = torch.diag(mm_result)
    print disToCenters.cpu().data.numpy()
    print out.cpu().data.numpy()
    
    #t_dis = torch.max(out)
    t_dis = torch.sum(torch.abs(advOutput - clusterCenters)) / 2
    return torch.sum(out) * (1-var_lambda) + t_dis * var_lambda