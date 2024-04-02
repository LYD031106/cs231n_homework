#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2024/3/28 20:04
# @Author : 刘亚东
# @Site : 
# @File : train.py
# @Software: PyCharm
import numpy as np
from data_utils import get_CIFAR10_data
from CNN import *
import os
from layer import *
from  layer_util import *
from opitim import *
from dataload import *
import pickle
import matplotlib.pyplot as plt
def write_imgage(loss_histroy,train_acc_history,val_acc_history):
    # 创建一个新的图表
    plt.figure()

    # 在图表中画出损失数据
    plt.plot(loss_histroy)

    # 添加标题和坐标轴标签
    plt.title('Loss Curve')
    plt.xlabel('step')
    plt.ylabel('Loss')
    # 显示图表
    plt.show()

    # 创建一个新的图表
    plt.figure()
    epoch_train = range(len(train_acc_history))
    epoch_val =   range(len(val_acc_history))
    # 在图表中画出训练准确率和验证准确率曲线
    plt.plot(epoch_train, train_acc_history, label='Train Accuracy', color='blue', marker='o')
    plt.plot(epoch_val, val_acc_history, label='Validation Accuracy', color='green', marker='s')


    # 添加标题和坐标轴标签
    plt.title('Training Metrics')
    plt.xlabel('Epochs')
    plt.ylabel('Metrics')
    # 添加图例
    plt.legend()
    # 显示图表
    plt.show()

def data_norm(X_train):
    mean = np.mean(X_train, axis=(0, 2, 3), keepdims=True)
    std = np.std(X_train, axis=(0, 2, 3), keepdims=True)
    # 应用归一化
    X_normalized = (X_train - mean) / std
    return X_normalized

class Train():
    def __init__(self
                 ,model
                 ,train_x
                 ,train_y
                 ,val_x
                 ,val_y
                 ,test_x
                 ,test_y
                 ,lr = 1e-3
                 ,optim = "sgd"
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
        self.optim = optim
        self.optim_config = {"learning_rate": self.lr, "mode": self.optim}


    def get_acc(self,X,y,batch_size = 32):
        y_pred = []
        total_sample = X.shape[0]
        num_batches = total_sample // batch_size
        if total_sample % batch_size != 0:
            num_batches += 1
        for i in range(num_batches):
            start = i * batch_size
            end = min((i + 1) * batch_size, total_sample)  # 避免数据超过范围
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

    def update_w(self, grad):
        if self.optim_config["mode"] == "sgd":
            for parameter_name in self.model.params.keys():
                gradient = grad[parameter_name]
                # 使用梯度和学习率更新参数
                dw,config = sgd(self.model.params[parameter_name],gradient , self.optim_config)
                self.model.params[parameter_name] = dw

    def train(self
              ,epoch = 200
              ,val_epoch = 10
              ,batch_size = 128):
        # 使用随机梯度下降更新梯度
        for i in range(1,epoch + 1):
            num_train = self.train_x.shape[0]
            batch_mask = np.random.choice(num_train, batch_size,replace=False)
            batch_x = self.train_x[batch_mask]
            batch_y = self.train_y[batch_mask]
            loss,grad = self.model.loss(batch_x,batch_y)
            self.loss_histroy.append(loss)
            # 更新
            self.update_w(grad)
            if i % 4 == 0:
                # 计算acc
                train_acc = self.get_acc(self.train_x, self.train_y)
                self.train_acc.append(train_acc)
                val_acc = self.get_acc(self.val_x,self.val_y)
                self.val_acc.append(val_acc)
            if i % val_epoch == 0:
                ave_loss = np.mean(self.loss_histroy[-10:])
                print(f"Epoch {i}/{epoch}  Loss:{ave_loss} ")
                print(f"{i}/{epoch}:train_acc : {train_acc} : {i}/{epoch}:val_acc : {val_acc}")
                if val_acc > self.best_val_acc:
                    self.best_params = self.model.params.copy()
                    self.best_val_acc = val_acc
                # self.save_checkpoint(i)
            if i % 100 == 0:
                self.lr = self.lr * 0.1
                self.optim_config["learning_rate"] = self.lr

        # 最后采用效果最佳的参数
        self.model.params = self.best_params


np.random.seed(231)
data = get_CIFAR10_data()
X_train = data["X_train"]
Y_train = data["y_train"]
X_test = data["X_train"]
Y_test = data["y_train"]
X_train = X_train[ : 500]
Y_train = Y_train[ : 500]
X_test = X_test[:100]
Y_test =Y_test[:100]
total_samples = X_train.shape[0]
split_ratio = 0.9  # 9:1的比例
split_index = int(total_samples * split_ratio)
train_x = X_train[:split_index]
train_y = Y_train[:split_index]
val_x = X_train[split_index:]
val_y = Y_train[split_index:]
best_model = None
best_acc = 0
best_lr = 0
best_reg = 0
best_loss_history = []
best_train_acc_history = []
best_val_acc_history = []
# learning_rates = np.array([4e-3,5e-3,1e-2,2e-2])
# learning_rates = np.array([1e-1])
# regularization_strengths = np.array([1e-1,1e-2]) # 生成3个正则化参数：1e-2, 1e-1, 1e0
learning_rates = [0.02]
regularization_strengths = [0.01]
for lr in learning_rates:
    for reg in regularization_strengths:
        model = CNN(use_dropout = True,dropout_keep_ratio = 0.5,reg=reg ,batch_norm = True)
        train = Train(model,train_x,train_y,val_x,val_y,X_test,Y_test,lr = lr)
        train.train()
        # 在测试集上面进行测试
        acc = train.get_acc(X_test,Y_test)
        if best_acc < acc:
            best_model = model
            best_lr = lr
            best_reg = reg
            best_acc = acc
            best_loss_history = train.loss_histroy
            best_train_acc_history = train.train_acc
            best_val_acc_history = train.val_acc
print(f"best_acc : {best_acc},best_lr = {best_lr},best_reg:{best_reg}")

write_imgage(best_loss_history,best_train_acc_history,best_val_acc_history)

