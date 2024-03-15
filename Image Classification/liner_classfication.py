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

    def predict(self,X_test,Y_test):
        total_loss = 0
        for i in range(X_test.shape[0]):
            output = self.forward(X_test[i])
            loss = self.SVMLoss(output,Y_test[i])
            total_loss += loss

    def forward(self,input):
        output = np.dot(input , self.W)
        output = output + self.b
        return output
    def SVMLoss(self,output,label):
            loss = np.sum(np.maximum(output-output[label],0))
            return loss

X_train,Y_train,X_test,Y_test = load_CIFAR10("../data/cifar-10-python")
X_train = X_train.reshape(X_train.shape[0],X_train.shape[1]*X_train.shape[2]*X_train.shape[3])
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1]*X_test.shape[2]*X_test.shape[3])

model = Liner_classification(X_train.shape[1],output_size = 10)
model.predict(X_test,Y_test)

