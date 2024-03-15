#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2024/3/14 19:24
# @Author : 刘亚东
# @Site : 
# @File : KNN.py
# @Software: PyCharm
from assignment1_colab.assignment1.cs231n.data_utils import load_CIFAR10
import numpy as np

class imageclassification_by_NearestNeighborClassifier:

    def __init__(self,X_train,Y_train):
        self.X_train = X_train
        self.Y_train = Y_train


    def predict_l1(self,X_test,k = 1):
        X_test_pred = np.zeros(self.X_train.shape[0])
        for i in range(X_test.shape[0]):
            # 取出待计算的训练集中的数据Xi,计算距离Xi最近的几个训练集数据
            distance = np.abs(np.sum(X_train - X_test[i],axis = 1))
            mindistance_index = np.argmin(distance)
            X_test_pred[i] = self.Y_train[mindistance_index]
        return X_test_pred
    def predict_l2(self,X_test,k = 1):
        X_test_pred = np.zeros(self.X_train.shape[0])
        for i in range(X_test.shape[0]):
            # 取出待计算的训练集中的数据Xi
            distance = np.sqrt(np.sum(np.square(X_train - X_test[i]),axis = 1))
            mindistance_index = np.argmin(distance)
            X_test_pred[i] = self.Y_train[mindistance_index]
        return X_test_pred

    def predict_l1_k(self,X_test,k):
        """
        :param X_test:
        :param k: 求出前k个最小的距离的label
        :return: 返回预测标签
        """
        X_test_pred = np.zeros(self.X_train.shape[0],dtype = X_train.dtype)
        for i in range(X_test.shape[0]):
            # 取出待计算的训练集中的数据Xi
            distance = np.abs(np.sum(X_train - X_test[i],axis = 1))
            mindistance_index = np.argsort(distance)[:k]
            lable_k = self.Y_train[mindistance_index]
            X_test_pred[i] = np.bincount(lable_k).argmax()
        return X_test_pred

    def predict_l2_k(self,X_test,k):
        X_test_pred = np.zeros(self.X_train.shape[0],dtype = X_train.dtype)
        for i in range(X_test.shape[0]):
            # 取出待计算的训练集中的数据Xi
            distance = np.sqrt(np.sum(np.square(X_train - X_test[i]),axis = 1))
            mindistance_index = np.argsort(distance)[:k]
            lable_k = self.Y_train[mindistance_index]
            X_test_pred[i] = np.bincount(lable_k).argmax()
        return X_test_pred

X_train,Y_train,X_test,Y_test = load_CIFAR10("../data/cifar-10-python")
X_train = X_train.reshape(X_train.shape[0],X_train.shape[1]*X_train.shape[2]*X_train.shape[3])
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1]*X_test.shape[2]*X_test.shape[3])

model = imageclassification_by_NearestNeighborClassifier(X_train,Y_train)
result = model.predict_l1_k(X_test,k = 4)
print(result==Y_test)