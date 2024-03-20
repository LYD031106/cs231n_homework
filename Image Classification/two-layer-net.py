#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2024/3/18 21:15
# @Author : 刘亚东
# @Site : 
# @File : two-layer-net.py
# @Software: PyCharm

import numpy as np

class two_layer_net():
    def __init__(self,
        input_dim = 3 * 32 * 32,
        hidden_dim = 100,
        num_classes = 10,
        weight_scale = 1e-3,
        reg = 0.0,
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.weight_scale = weight_scale
        self.reg = reg
        self.params = {}
        self.params["W1"] = np.random.normal(loc=0.0,scale=weight_scale,size=(input_dim , hidden_dim))
        self.params["b1"] = np.zeros(hidden_dim)
        self.params["W2"] = np.random.normal(loc=0.0,scale=weight_scale , size = (hidden_dim , num_classes))
        self.params["b2"] = np.zeros(num_classes)
        self.reg = reg

    def loss(self,X,y = None):
        """
        :param X: 输入(
        :param y:
        :return: 返回本轮的loss以及梯度
        """