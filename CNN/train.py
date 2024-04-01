#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2024/3/28 20:04
# @Author : 刘亚东
# @Site : 
# @File : train.py
# @Software: PyCharm
import numpy as np

from CNN import *
import os
from layer import *
from  layer_util import *
from opitim import *
from dataload import *
import pickle
import matplotlib.pyplot as plt
def write_imgage(train):
    # 创建一个新的图表
    plt.figure()

    # 在图表中画出损失数据
    plt.plot(train.loss_histroy)

    # 添加标题和坐标轴标签
    plt.title('Loss Curve')
    plt.xlabel('step')
    plt.ylabel('Loss')
    # 显示图表
    plt.show()

    # 创建一个新的图表
    plt.figure()

    # 在图表中画出损失数据
    plt.plot(train.loss_histroy)

    # 添加标题和坐标轴标签
    plt.title('Loss Curve')
    plt.xlabel('step')
    plt.ylabel('Loss')
    # 显示图表
    plt.show()

    # 创建一个新的图表
    plt.figure()
    train_acc = model.train_acc
    val_acc = model.val_acc
    epoch_train = len(train_acc)
    epoch_val = len(val_acc)
    # 在图表中画出训练准确率和验证准确率曲线
    plt.plot(epoch_train, train_acc, label='Train Accuracy', color='blue', marker='o')
    plt.plot(epoch_val, val_acc, label='Validation Accuracy', color='green', marker='s')


    # 添加标题和坐标轴标签
    plt.title('Training Metrics')
    plt.xlabel('Epochs')
    plt.ylabel('Metrics')

    # 添加图例
    plt.legend()

    # 显示图表
    plt.show()
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
                 ,lr = 2e-3
                 ):
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
        self.best_val_acc = 0
        self.lr = lr

    def get_acc(self,X,y,batch_size = 32):
        y_pred = []
        total_sample = X.shape[0]
        num_batches = total_sample // batch_size
        if total_sample % batch_size == 0:
            num_batches+=1
        for i in range(num_batches + 1):
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
        os.makedirs(fr"checkpoint/epoch{i}")
        with open(f"checkpoint/epoch{i}", 'wb') as f:
            pickle.dump(f"checkpoint/epoch{i}", f)

    def update_w(self, grad, mode={"learning_rate": 1e-3, "mode": "sgd"}):
        if mode["mode"] == "sgd":
            for parameter_name in self.model.params.keys():
                gradient = grad[parameter_name]
                # 使用梯度和学习率更新参数
                dw,config = sgd(self.model.params[parameter_name],gradient , mode)
                self.model.params[parameter_name] = dw

    def train(self
              ,epoch = 200
              ,val_epoch = 10
              ,batch_size = 64):
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
            if i % val_epoch == 0:
                val_acc = self.get_acc(self.val_x,self.val_y)
                print(f"{i}/{epoch}:train_acc : {train_acc} : {i}/{epoch}:val_acc : {val_acc}")
                print(f"")
                self.val_acc.append(val_acc)
                if val_acc > self.best_val_acc:
                    self.best_params = self.model.params.copy()
                    self.best_val_acc = val_acc
                # self.save_checkpoint(i)








model = CNN(use_dropout = False)
X_train,Y_train,X_test,Y_test = load_CIFAR10("../data/cifar-10-python")
# 分割验证集和训练集
# 随机选取训练集中的样本
num_train_samples = X_train.shape[0]
num_train_selected = 500
num_test_samples = X_test.shape[0]
num_test_selected = 100
train_indices = np.random.choice(num_train_samples, num_train_selected, replace=False)
X_train = X_train[train_indices]
Y_train = Y_train[train_indices]

# 随机选取测试集中的样本
test_indices = np.random.choice(num_test_samples, num_test_selected, replace=False)
X_test = X_test[test_indices]
Y_test = Y_test[test_indices]
total_samples = X_train.shape[0]
split_ratio = 0.8  # 9:1的比例
split_index = int(total_samples * split_ratio)
train_x = X_train[:split_index]
train_y = Y_train[:split_index]
val_x = X_train[split_index : ]
val_y = Y_train[split_index : ]
train = Train(model,train_x,train_y,val_x,val_y,X_test,Y_test)
train.train()

