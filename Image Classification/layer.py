#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2024/3/20 20:12
# @Author : 刘亚东
# @Site : 
# @File : layer.py
# @Software: PyCharm
import numpy as np
# class layer():
def fully_forward(w,x,b):
    """
    :param w: 全连接层的参数(D,M),D为输入的特征，M为输出
    :param x: 输入输入(N,D) N为batch_size，D为数据特征
    :param b: 偏置项
    :return:返回tuple，由w,x,b组成以及输出
    """
    output = x.dot(w) + b
    cache = (w,x,b)
    return output , cache

def fully_backward(dout,cache):
    """
    :param dout: 上游导数，shape(N,M)
    :param cache: Tuple(w,x,b）
           : x 输入输入(N,D) N为batch_size，D为数据特征
           : w 全连接层的参数(D,M),D为输入的特征，M为输出
           : b 偏置项
    :return:返回tuple，由dw,dx,db组成
    根据链式准则，这一层的导数等于上一层的dout乘这一层求导
    """
    w,x,b = cache
    dx,dw,db = None,None,None
    dx = dout.dot(w.T)
    dw = x.T.dot(dout)
    db = np.sum(dout,axis = 0)
    return dx , dw , db

def relu_forward(x):
    output = np.maximum(0,x)
    return output,x

def relu_backward(dout,cache):
    """
    :param dout: 上游导数，shape(N,M)
    :param cache: x 任意形状
    :return:
    """
    x = cache
    dout[x<0] = 0
    return dout
def svm_loss(x,y):
    """
    :param x: 输入输入(N,D) 在计算loss，表示得分
    :param y: x对应的label
    :return:返回loss 和 loss对于x的求导
    """
    correct_score = x[np.arange(x.shape[0]),y]
    x = x - correct_score + 1
    np.maximum(0 , x)
    x [np.arange(x.shape[0]),y] = 0
    loss = np.sum(x)
    loss /= x.shape[0]
    margin = x[x > 0]
    dx = np.zeros(x.shape)
    dx[margin] = 1
    mid = np.sum(dx , axis = 1)
    dx[np.arange(x.shape[0]),y] -= mid
    dx /= x.shape[0]
    return loss , dx

def softmax_loss(x,y):
    """
    :param x: 输入输入(N,D) 在计算loss，表示得分
    :param y: x对应的label
    :return:返回loss 和 loss对于x的求导
    """
    # 首先计算概率
    x -= np.argmax(x,axis = 1).reshape(1,-1).T
    x = np.exp(x)
    x = x/np.sum(x,axis = 1).reshape(1,-1).T
    loss = np.sum(-np.log(x[np.arange(x.shape[0]),y]))
    loss /= x.shape[0]

    # 计算梯度
    x[np.arange(x.shape[0]),y] -= 1
    dx = x
    dx /= x.shape[0]
    return loss,dx
