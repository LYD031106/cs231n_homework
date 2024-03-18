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
        self.W = np.random.randn(X_train.shape[1],output_size) * 0.001
        self.X_train = X_train
        self.Y_train = Y_train

    def train(self,batch_size , epoch,lr,reg):
        """
        本部分使用SGD,每一个epoch随机挑选一个batch_size来进行梯度下降，因此仅仅需要梯度下降一次
        :param batch_size: 设定一次batch_size
        :param epoch: 设定训练多少论
        :param lr : 学习率
        :param reg : 正则化参数
        :return: 返回一个最佳的self.W
        """

        for i in range(1,epoch+1):
            traind_index = np.random.choice(self.X_train.shape[0],size = batch_size,replace = False)
            X_train_batch = self.X_train[traind_index]
            Y_train_batch = self.Y_train[traind_index]
            loss, gradient = self.train_svm_classification_byvector(X_train_batch,Y_train_batch,reg = reg)
            self.W -= lr * gradient
            predict_label = self.predict(X_train_batch)
            accuracy = np.mean(predict_label == Y_train_batch)
            if i % 1000 == 0:
                print(f"{i} epoch--loss :{loss}--accuracy:{accuracy}")


    def train_svm_classification(self,reg,X,Y):
        """
        :param X : 输入数据（N*D）N表示了X中包含的图片数量，D表示输入照片维度
        :param Y : 表示X对应的标签
        :param reg: 正则化
        :return:
        """
        total_loss = 0
        dw = np.zeros(self.W.shape)
        for i in range(X.shape[0]):
            output = self.forward(X[i])
            correct_score = output[Y[i]]
            for j in range(X.shape[1]):
                if j == Y[i]:
                    continue
                else:
                    loss = output[j] - correct_score + 1
                    if loss > 0:
                       dw[:,Y[j]] += -X[i]
                       dw[:,j] += X[i]
                       total_loss += loss
        total_loss /= X.shape[0]
        dw /= X.shape[0]
        dw += 2 * reg * np.sum(self.W)
        total_loss += reg*np.sum(self.W * self.W)
        #对于total_loss进行均值
        return total_loss,dw
    def train_svm_classification_byvector(self,X,Y,reg):
        """
        :param reg: 正则化
        :return:
        """
        score = X.dot(self.W)
        dw = np.zeros(self.W.shape) #我们最后的dW需要利用损失函数对于得分的偏导 乘 得分对于对于w的偏导，通过链式法则求解
        #计算loss
        correct_score = score[np.arange(score.shape[0]),Y].reshape(1, -1).T
        score = score - correct_score + 1
        score = np.maximum(0,score)
        score[range(score.shape[0]), Y] = 0
        loss = np.sum(score,axis=1)
        total_loss = np.sum(loss)/X.shape[0] + reg * np.sum(self.W * self.W)
        dS = np.zeros(score.shape)
        dS[score>0] = 1
        dS[np.arange(dS.shape[0]),Y] -= np.sum(dS,axis = 1)
        dw += X.T.dot(dS)
        dw /= X.shape[0]
        dw += 2 * reg * self.W
        return total_loss ,dw

    def train_softmax_classifcation(self,X,Y,reg):
        """
        :param X : 输入数据（N*D）N表示了X中包含的图片数量，D表示输入照片维度
        :param Y : 表示X对应的标签
        :param reg: 正则化
        :return:
        """
        total_loss = 0
        dw = np.zeros(self.W.shape)
        total_loss = 0
        for i in range(X.shape[0]):
            output = self.forward(X[i])
            output -= output[np.argmax(output, axis=0)]
            exp_x = np.exp(output)
            softmax_output = exp_x / np.sum(exp_x)
            total_loss += -np.log(softmax_output[Y[i]])
            for j in range(X.shape[1]):
                target = softmax_output[j]
                if j == Y[i]:
                    dw[:,j] += (target - 1) * X[i]
                else:
                    dw[:,j] += target * X[i]
        return total_loss, dw

    def predict(self,input):
        output = self.forward(input)
        label = np.argmax(output,axis = 1)
        return label

    def SVMLoss(self,output,label):
        mid = output - output[label] + 1
        mid[label] = 0
        loss = np.sum(np.maximum(mid,0))
        return loss
    def softmaxLoss(self,input,label,reg):
        output = input.dot(self.W)
        output -= np.argmax(output,axis = 1).reshape(1,-1).T
        exp_x = np.exp(output)
        softmax_output = exp_x/np.sum(exp_x,axis = 1).reshape(1,-1).T
        softmax_output = softmax_output * softmax_output[label]
        loss = np.sum(-np.log(softmax_output[np.arange(input.shape[0]),label]))/output.shape[0] + np.sum(self.W**2)*reg
        softmax_output[np.arange(input.shape[0]), label] -= 1
        return loss


    def forward(self,input):
        output = np.dot(input , self.W)
        return output


X_train,Y_train,X_test,Y_test = load_CIFAR10("../data/cifar-10-python")
X_train = X_train.reshape(X_train.shape[0],X_train.shape[1]*X_train.shape[2]*X_train.shape[3])
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1]*X_test.shape[2]*X_test.shape[3])

mean_image = np.mean(X_train,axis = 1)


model = Liner_classification(output_size = 10,X_train = X_train , Y_train = Y_train)
model.softmaxLoss(X_train,Y_train,2e-2)

