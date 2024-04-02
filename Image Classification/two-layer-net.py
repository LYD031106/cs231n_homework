#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2024/3/18 21:15
# @Author : 刘亚东
# @Site : 
# @File : two-layer-net.py
# @Software: PyCharm

import numpy as np
from assignment1_colab.assignment1.cs231n.data_utils import load_CIFAR10
from layer import *
from torch.utils.tensorboard import SummaryWriter

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
        :param X: 输入输入(N,D) N为batch_size，D为input_dim
        :param y: x对于lable，假如没有不进行梯度计算以及loss
        :return: 返回本轮的loss以及梯度
        """
        ## 设计两层全连接层，中间一个激活函数用于增加线性
        w1 = self.params["W1"]
        b1 = self.params["b1"]
        w2 = self.params["W2"]
        b2 = self.params["b2"]
        output1,cache1 = fully_forward(w1,X,b1)
        output2,cache2 = relu_forward(output1)
        output3,cache3 = fully_forward(w2,output2,b2)

        if y is None:
            return output3
        # 计算loss
        loss,dx = softmax_loss(output3,y)
        # 反向传播更新梯度
        dx,dw2,db2 = fully_backward(dx,cache3)
        dx = relu_backward(dx,cache2)
        dx,dw1,db1 = fully_backward(dx,cache1)
        grads = {}
        grads["W1"] =dw1
        grads["b1"] =db1
        grads["W2"] =dw2
        grads["b2"] =db2
        return loss , grads

    def predcit_label(self,x):
         x -= np.argmax(x, axis=1).reshape(1, -1).T
         x = np.exp(x)
         x = x / np.sum(x, axis=1).reshape(1, -1).T
         label = np.argmax(x , axis = 1)
         return label


    def updata_para(self,grad,lr):
        for parameter_name in self.params.keys():
            # 获取参数对应的梯度
            gradient = grad[parameter_name]  # 假设梯度以字典形式存储，参数名为键
            # 使用梯度和学习率更新参数
            self.params[parameter_name] -= lr * gradient

    def get_acc(self,x,y):
        pred_label = self.predcit_label(x)
        acc = np.mean(pred_label == y)
        return acc


    def train(self,train_x,train_y,val_x,val_y,lr,reg,epoch, batch_size = 32):
        """
        用于更新模型参数
        :param x:输入的x，（num_examples,input_dim）
        :param y:x对于的label
        :param lr:学习率
        :param reg:正则化
        :param epoch: 训练轮数
        :return:
        """
        writer = SummaryWriter('logs')  # logs 是存储 TensorBoard 日志的目录
        num_examples = train_x.shape[0]
        step = 0
        total_loss = 0
        for i in range(1,epoch):
            # 使用小批量梯度下降算法
            shuffled_indices = np.random.permutation(num_examples)
            for batch_start in range(0, num_examples, batch_size):
                batch_indices = shuffled_indices[batch_start:batch_start + batch_size]
                batch_x = train_x[batch_indices]
                batch_y = train_y[batch_indices]
                loss , grad = self.loss(batch_x,batch_y)
                # 绘制损失函数
                total_loss += loss
                writer.add_scalar('Loss/train', loss, step)
                step += 1
                # 更新梯度
                self.updata_para(grad,lr)
                if step % 100 == 0:
                    print(f"num_epoch:{i}:steo:{step}:loss :{total_loss}")
                    total_loss = 0
                # if i >= 10:
                #     lr = 1e-8

            if i % 2 == 0:
                score = self.loss(val_x)
                val_loss,_= softmax_loss(score , val_y)
                val_acc = self.get_acc(score,val_y)
                print(val_acc)
                # 记录损失和准确率到 TensorBoard
                writer.add_scalar('Loss/val', val_loss, i)
                writer.add_scalar('Accuracy/val', val_acc, i)


X_train,Y_train,X_test,Y_test = load_CIFAR10("../data/cifar-10-python")
X_train = X_train.reshape(X_train.shape[0],X_train.shape[1]*X_train.shape[2]*X_train.shape[3])
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1]*X_test.shape[2]*X_test.shape[3])
# 分割验证集和训练集
total_samples = X_train.shape[0]
split_ratio = 0.9  # 9:1的比例
split_index = int(total_samples * split_ratio)
train_x = X_train[:split_index]
train_y = Y_train[:split_index]
val_x = X_train[split_index : ]
val_y = Y_train[split_index : ]
model = two_layer_net()
model.train(train_x,train_y,val_x,val_y,lr = 1e-4, reg= 0.2 , epoch = 1000)

