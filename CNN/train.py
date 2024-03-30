#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2024/3/28 20:04
# @Author : 刘亚东
# @Site : 
# @File : train.py
# @Software: PyCharm
import numpy as np

from CNN  import *
import os
from layer import *
from  layer_util import *
from opitim import *
from dataload import *
import pickle
class Train():
    def __init__(self
                 ,model
                 ,train_x
                 ,train_y
                 ,val_x
                 ,val_y
                 ,test_x
                 ,test_y
                 ,val_b = 4
                 ,lr = 1e-5):
        self.train_x = train_x
        self.train_y = train_y
        self.val_x = val_x
        self.val_y = val_y
        self.test_x = test_x
        self.test_y = test_y
        self.model = model
        self.loss_histroy = []
        self.train_acc = []
        self.val_acc = []
        self.best_paras = {}
        self.val_b = val_b
        self.best_val_acc = 0
        self.lr = lr

    def get_acc(self,X,y,batch_size = 64):
        y_pred = []
        total_sample = X.shape[0]
        num_batches = total_sample // batch_size
        if total_sample % batch_size == 0:
            num_batches+=1
        for i in range(num_batches):
            start = i * batch_size
            end = (i + 1) * batch_size
            scores = self.model.loss(X[start:end])
            y_pred.append(np.argmax(scores, axis=1))
        y_pred = np.hstack(y_pred)
        acc = np.mean(y_pred == y)
        return acc

    def save_checkpoint(self,i):
        if os.path.exists(r"checkpoint") and os.path.isdir(r"checkpoint"):
            os.makedirs(r"checkpoint")
        with open(f"checkpoint/epoch{i}", 'wb') as f:
            pickle.dump(self.best_params, f)

    def update_w(self, grad, mode={"learning_rate": 1e-6, "mode": "sgd"}):
        if mode["mode"] == "sgd":
            for parameter_name in self.model.params.keys():
                gradient = grad[parameter_name]
                # 使用梯度和学习率更新参数
                self.model.params[parameter_name] = sgd(self.model.params[parameter_name],gradient , mode)[0]

    def train(self
              ,lr = 1e-6
              ,epoch = 30
              ,val_epoch = 5
              ,batch_size = 32):
        # 使用随机梯度下降更新梯度
        for i in range(1,epoch + 1):
            num_train = self.train_x.shape[0]
            batch_mask = np.random.choice(num_train, batch_size)
            batch_x = self.train_x[batch_mask]
            batch_y = self.train_y[batch_mask]
            loss,grad = self.model.loss(batch_x,batch_y)
            print(f"Epoch {i}/{epoch}  Loss:{loss} ")
            self.loss_histroy.append(loss)
            # 更新
            self.update_w(grad, {"learning_rate": self.lr, "mode": "sgd"})
            # 计算acc
            train_acc = self.get_acc(self.train_x,self.train_y)
            self.train_acc.append(train_acc)
            if i % self.val_b == 0:
                val_acc = self.get_acc(self.val_x,self.val_y)
                self.val_acc.append(val_acc)
                if val_acc > self.best_val_acc:
                    self.best_params = self.model.params.copy()
                    self.best_val_acc = val_acc
                self.save_checkpoint(i)








model = CNN()
X_train,Y_train,X_test,Y_test = load_CIFAR10("../data/cifar-10-python")
# 分割验证集和训练集
total_samples = X_train.shape[0]
split_ratio = 0.9  # 9:1的比例
split_index = int(total_samples * split_ratio)
train_x = X_train[:split_index]
train_y = Y_train[:split_index]
val_x = X_train[split_index : ]
val_y = Y_train[split_index : ]
train = Train(model,train_x,train_y,val_x,val_y,X_test,Y_test)
train.train()
