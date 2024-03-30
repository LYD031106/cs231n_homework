#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2024/3/28 15:58
# @Author : 刘亚东
# @Site : 
# @File : CNN.py
# @Software: PyCharm
import numpy as np
from layer import *
from layer_util import *
class CNN():
    def __init__(self,
                 input_dim=(3,32, 32),
                 num_filters=32,
                 filter_size=7,
                 hidden_dim=100,
                 num_classes=10,
                 weight_scale=1e-3,
                 reg=0.0,
                 dtype=np.float32,
                 ):
        self.params = {}
        self.reg = reg
        self.dtype = dtype
        C, H, W = input_dim  # 获取输入数据的通道数，高度，宽度

        # 初始化参数
        # 卷积层参数
        self.params["W1"] = np.random.normal(0, weight_scale, (num_filters, C, filter_size, filter_size))
        self.params["b1"] = np.zeros(num_filters)

        # 全连接层参数
        self.params["W2"] = np.random.normal(0, weight_scale, (num_filters * H * W // 4, hidden_dim))
        self.params["b2"] = np.zeros(hidden_dim)

        # 最后一个全连接层
        self.params["W3"] = np.random.normal(0, weight_scale, (hidden_dim, num_classes))
        self.params["b3"] = np.zeros(num_classes)

        # 转换numpy的数据类型便于后期计算
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)
    def loss(self,X,y = None):
        W1, b1 = self.params["W1"], self.params["b1"]
        W2, b2 = self.params["W2"], self.params["b2"]
        W3, b3 = self.params["W3"], self.params["b3"]

        filter_size = W1.shape[2]
        conv_param = {"stride": 1, "pad": (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {"pool_height": 2, "pool_width": 2, "stride": 2}

        scores = None
        # conv - relu - 2x2 max pool - affine - relu - affine - softmax
        out1, cache1 = conv_relu_pool_fast_forward(X, W1, b1, conv_param, pool_param)  # 卷积层
        out2, cache2 = affine_relu_forward(out1, W2, b2)  # 全连接层
        scores, cache3 = fully_forward(out2, W3, b3)  # 全连接层

        if y is None:
            return scores

        loss = 0
        grads = {}

        # 计算损失
        loss, dout = softmax_loss(scores, y)
        loss += 0.5 * self.reg * (np.sum(W1 ** 2) + np.sum(W2 ** 2) + np.sum(W3 ** 2))  # L2正则化

        # 计算梯度
        dout1, grads["W3"], grads["b3"] = fully_backward(dout, cache3)  # 全连接层
        dout2, grads["W2"], grads["b2"] = affine_relu_backward(dout1, cache2)  # 全连接层
        dout3, grads["W1"], grads["b1"] = conv_relu_pool_fast_backward(dout2, cache1)  # 卷积层

        # 加上正则化项的梯度
        grads["W3"] += self.reg * W3
        grads["W2"] += self.reg * W2
        grads["W1"] += self.reg * W1

        return loss, grads