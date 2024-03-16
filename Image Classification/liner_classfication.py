#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2024/3/15 19:03
# @Author : 刘亚东
# @Site : 
# @File : liner_classfication.py
# @Software: PyCharm
from assignment1_colab.assignment1.cs231n.data_utils import load_CIFAR10
import numpy as np

class Liner_classification:
    def __init__(self,input_size,output_size):
        self.W = np.random.randn(input_size,output_size)
        self.b = np.random.randn(output_size)

    def predict(self,X_test,Y_test,reg):
        """
        :param X_test: 包含一个所有数据的形状为 (N, D) 的 NumPy 数组，其中 N 是测试集的数量，D 是特征的维度
        :param Y_test: 测试集中的数据
        :param reg: 正则化的值，使用L1正则化
        :return:
        """
        total_loss = 0
        for i in range(X_test.shape[0]):
            output = self.forward(X_test[i])
            loss = self.SVMLoss(output,Y_test[i])
            total_loss += loss
        total_loss += reg*np.sum(self.W * self.W)
        #对于total_loss进行均值
        total_loss /= X_test.shape[0]

    def minbatchpredict(self,X_test,Y_test):
        """
        :param X_test:  包含一个小批量数据的形状为 (N, D) 的 NumPy 数组，其中 N 是批量大小，D 是特征的维度
        :param Y_test:  维度是（N）,存储了小批量数据的label值
        :return: 返回loss值
        """
        output = self.forward(X_test)
        loss = self.SVMLoss(output,Y_test)

    def forward(self,input):
        output = np.dot(input , self.W)
        output = output + self.b
        return output

    def SVMLoss(self,output,label):
            mid = output - output[label] + 1
            mid[label] = 0
            loss = np.sum(np.maximum(mid,0))
            return loss
    def SVMbatchloss(self,output,Y_test):
        loss = output - Y_test

X_train,Y_train,X_test,Y_test = load_CIFAR10("../data/cifar-10-python")
X_train = X_train.reshape(X_train.shape[0],X_train.shape[1]*X_train.shape[2]*X_train.shape[3])
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1]*X_test.shape[2]*X_test.shape[3])

model = Liner_classification(X_train.shape[1],output_size = 10)
model.predict(X_test,Y_test,1e-9)

