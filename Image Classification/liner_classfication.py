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
    def __init__(self,output_size,X_train,Y_train):
        self.W = np.random.randn(X_train.shape[1],output_size)
        self.X_train = X_train
        self.Y_train = Y_train

    def train_svm_classification(self,reg):
        """
        :param reg: 学习率
        :return:
        """
        total_loss = 0
        dw = np.zeros(self.W.shape)
        for i in range(self.X_train.shape[0]):
            output = self.forward(self.X_train[i])
            correct_score = output[self.Y_train[i]]
            for j in range(self.X_train.shape[1]):
                if j == self.Y_train[i]:
                    continue
                else:
                    loss = output[j] - correct_score + 1
                    if loss > 0:
                       dw[:,Y_train[j]] += -self.X_train[i]
                       dw[:,j] += self.X_train[i]
                       total_loss += loss
        total_loss /= self.X_train.shape[0]
        dw /= self.X_train.shape[0]
        total_loss += reg*np.sum(self.W * self.W)
        #对于total_loss进行均值
        return total_loss,dw
    def train_svm_classification_byvector(self,reg):
        """
        :param reg: 学习率
        :return:
        """
        score = self.X_train.dot(self.W)
        #计算loss
        correct_score = score[np.arange(score.shape[0]),self.Y_train].reshape(1, -1)
        score = score - correct_score + 1
        midloss = np.maximum(0,score)
        score[range(score.shape[0]), self.Y_train] = 0
        loss = np.sum(score,axis=1)
        total_loss = np.sum(loss)/self.X_train.shape[0] + reg * np.sum(self.W * self.W)
        return score



    def gradienct(self,loss):
        pass
    def forward(self,input):
        output = np.dot(input , self.W)
        output = output
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

mean_image = np.mean(X_train,axis = 1)


model = Liner_classification(output_size = 10,X_train = X_train , Y_train = Y_train)
model.train_svm_classification_byvector(reg = 1e-9)

