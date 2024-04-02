#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2024/3/28 15:58
# @Author : 刘亚东
# @Site : 
# @File : layer.py
# @Software: PyCharm
import time
import numpy as np
from im2col_python import *
def fully_forward(x,w,b):
    """
    :param w: 全连接层的参数(D,M),D为输入的特征，M为输出
    :param x: 输入输入(N, d_1, ..., d_k) N为batch_size，d_1, ..., d_k为数据特征,同时d_1相乘是D
    :param b: 偏置项
    :return:返回tuple，由w,x,b组成以及输出
    """
    output = x.reshape((x.shape[0],-1)).dot(w) + b
    cache = (x,w,b)
    return output , cache

def fully_backward(dout,cache):
    """
    :param dout: 上游导数，shape(N,M)
    :param cache: Tuple(w,x,b）
           : x 输入输入(N,D) N为batch_size，D为数据特征
           : w 全连接层的参数(D,M),D为输入的特征，M为输出
           : b 偏置项
    :return:返回tuple，由dw,dx,db组成
    根据链式准则，这一层的导数等于上一层的dout乘这一层求导
    """
    x,w,b = cache
    dx,dw,db = None,None,None
    dx = dout.dot(w.T)
    shape = x.shape
    dx = dx.reshape(shape)
    dw = x.reshape(x.shape[0],-1).T.dot(dout)
    db = np.sum(dout,axis = 0)
    return dx , dw , db

def relu_forward(x):

    output = np.maximum(0,x)
    return output,x

def relu_backward(dout,cache):
    """
    :param dout: 上游导数，shape(N,M)
    :param cache: x 任意形状
    :return:
    """
    x = cache
    dout[x<0] = 0
    return dout
def svm_loss(x,y):
    """
    :param x: 输入输入(N,D) 在计算loss，表示得分
    :param y: x对应的label
    :return:返回loss 和 loss对于x的求导
    """
    correct_score = x[np.arange(x.shape[0]),y]
    x = x - correct_score + 1
    np.maximum(0 , x)
    x [np.arange(x.shape[0]),y] = 0
    loss = np.sum(x)
    loss /= x.shape[0]
    margin = x[x > 0]
    dx = np.zeros(x.shape)
    dx[margin] = 1
    mid = np.sum(dx , axis = 1)
    dx[np.arange(x.shape[0]),y] -= mid
    dx /= x.shape[0]
    return loss , dx

def softmax_loss(x,y):
    """
    :param x: 输入输入(N,D) 在计算loss，表示得分
    :param y: x对应的label
    :return:返回loss 和 loss对于x的求导
    """
    loss, dx = None, None
    output = x
    output -= np.max(output, axis=1).reshape(1, -1).T
    exp_x = np.exp(output)
    softmax_output = exp_x / np.sum(exp_x, axis=1).reshape(1, -1).T
    loss = np.sum(-np.log(softmax_output[np.arange(x.shape[0]), y])) / output.shape[0]

    softmax_output[np.arange(x.shape[0]), y] -= 1
    dx = softmax_output
    dx /= x.shape[0]

    return loss, dx

def conv_forward_naive(x, w, b, conv_param):
    """A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    start = time.time()
    out = None
    # 先获取一些需要用到的数据
    N, C, H_input, W_input = x.shape  # N个样本，C个通道，H_input高，W_input宽
    F, C_w_, HH, WW = w.shape  # F个卷积核, C_w_个通道，HH高，WW宽
    stride = conv_param["stride"]  # 步长
    pad = conv_param["pad"]  # 填充数量

    # 计算卷积后的高和宽
    out_H = int(1 + (H_input + 2 * pad - HH) / stride)
    out_W = int(1 + (W_input + 2 * pad - WW) / stride)

    # 给x的上下左右填充上pad个0
    x_pad = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), "constant", constant_values=0)
    # 将卷积核w转换成F * (C * HH * WW)的矩阵 (便于使用矩阵乘法)
    w_row = w.reshape(F, -1)
    # 生成空白输出便于后续循环填充
    out = np.zeros((N, F, out_H, out_W))

    # 开始卷积
    for n in range(N):  # 遍历样本
        for f in range(F):  # 遍历卷积核
            for i in range(out_H):  # 遍历高
                for j in range(out_W):  # 遍历宽
                    # 获取当前卷积窗口
                    window = x_pad[n, :, i * stride:i * stride + HH, j * stride:j * stride + WW]
                    # 将卷积窗口拉成一行
                    window_row = window.reshape(1, -1)
                    # 计算当前卷积窗口和卷积核的卷积结果
                    out[n, f, i, j] = np.sum(window_row * w_row[f, :]) + b[f]

        # 将pad后的x存入cache (省的反向传播的时候在计算一次)
    x = x_pad
    cache = (x, w, b, conv_param)
    end = time.time()
    return out, cache

def conv_backward_naive(dout, cache):
    """A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    # 获取一些需要用到的数据
    x, w, b, conv_param = cache
    N, C, H_input, W_input = x.shape  # N个样本，C个通道，H_input高，W_input宽
    F, C_w_, HH, WW = w.shape  # F个卷积核, C_w_个通道，HH高，WW宽
    stride = conv_param["stride"]  # 步长
    pad = conv_param["pad"]  # 填充数量

    # 计算卷积后的高和宽
    out_H = int(1 + (H_input - HH) / stride)
    out_W = int(1 + (W_input - WW) / stride)

    # 给dx,dw,db分配空间
    dx = np.zeros_like(x)
    dw = np.zeros_like(w)
    db = np.zeros_like(b)

    for n in range(N):
        for f in range(F):
            for i in range(out_H):
                for j in range(out_W):
                    # 获取当前卷积窗口
                    window = x[n, :, i * stride:i * stride + HH, j * stride:j * stride + WW]
                    # 计算db
                    db[f] += dout[n, f, i, j]
                    # 计算dw
                    dw[f] += window * dout[n, f, i, j]
                    # 计算dx
                    dx[n, :, i * stride:i * stride + HH, j * stride:j * stride + WW] += w[f] * dout[n, f, i, j]

    # 去掉dx的pad
    dx = dx[:, :, pad:H_input - pad, pad:W_input - pad]
    return dx, dw, db

def conv_forward_fast(x, w, b, conv_param):
    """
    Forward pass for the convolution operation using img2col.

    :param x: Input image or batch of images, shape (batch, channel, height, width)
    :param w: Convolution filters, shape (num_filters, channel, filter_height, filter_width)
    :param b: Bias term for the filters, shape (num_filters,)
    :param conv_param: Dictionary containing stride and padding information

    :return: Output of the convolution operation and cache for the backward pass
    """
    stride = conv_param["stride"]
    padding = conv_param["pad"]
    batch, channel, height, width = x.shape
    num_filters, _, filter_height, filter_width = w.shape

    # Apply padding to the input
    input_padded = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant')

    # Determine output dimensions
    h_out = (height + 2 * padding - filter_height) // stride + 1
    w_out = (width + 2 * padding - filter_width) // stride + 1


    # Img2Col
    col_input = img2col(input_padded, h_out, w_out, filter_height, filter_width, stride)
    # merge channel
    col_input = col_input.reshape(col_input.shape[0], -1, col_input.shape[3])

    # reshape kernel
    weights_flatten = w.reshape(w.shape[0], -1)

    # compute convolution
    output = weights_flatten @ col_input + b.reshape(-1, 1)

    # reshape convolution result
    output = output.reshape(output.shape[0], output.shape[1], h_out, w_out)
    cache = (x, w, b, conv_param)
    #####################################################################################
    return output,cache

def conv_backward_fast(dout, cache):
    """
    # Arguments
        out_grad: gradient to the forward output of conv layer, with shape (batch, out_channel, out_height, out_width)
        input: numpy array with shape (batch, in_channel, in_height, in_width)
        weights: numpy array with shape (out_channel, in_channel, kernel_h, kernel_w)
        bias: numpy array with shape (out_channel)

    # Returns
        in_grad: gradient to the forward input of conv layer, with same shape as input
        w_grad: gradient to weights, with same shape as weights
        b_bias: gradient to bias, with same shape as bias
    """

    x, w, b, conv_param = cache
    padding = conv_param['pad']
    stride = conv_param['stride']
    batch, in_channel, in_height, in_width = x.shape
    batch, out_channel, out_height, out_width = dout.shape
    num_filters, _, kernel_h, kernel_w = w.shape
    #################################################################################
    batch, out_channel, out_height, out_width = dout.shape
    """
       compute b_grad
    """
    b_grad = np.sum(dout, axis=(0, 2, 3))
    b_grad = b_grad.reshape(out_channel)

    # pad zero to input
    input_padded = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant')
    # Img2Col
    col_input = img2col(input_padded, out_height, out_width, kernel_h, kernel_w, stride)
    # merge channel
    col_input = col_input.reshape(col_input.shape[0], -1, col_input.shape[3])

    # transpose and reshape col_input to 2D matrix
    X_hat = col_input.transpose(1, 2, 0).reshape(in_channel * kernel_h * kernel_w, -1)
    # transpose and reshape out_grad
    out_grad_reshape = dout.transpose(1, 2, 3, 0).reshape(out_channel, -1)

    """
        compute w_grad
    """
    w_grad = out_grad_reshape @ X_hat.T
    w_grad = w_grad.reshape(w.shape)

    """
        compute in_grad
    """
    # reshape kernel
    W = w.reshape(out_channel, -1)
    in_grad_column = W.T @ out_grad_reshape

    # Split batch dimension and transpose batch to first dimension
    in_grad_column = in_grad_column.reshape(in_grad_column.shape[0], -1, batch).transpose(2, 0, 1)

    in_grad = col2img(in_grad_column, in_height + padding*2, in_width + padding*2, kernel_h, kernel_w, in_channel, padding, stride)
    #################################################################################
    return in_grad, w_grad, b_grad


def max_pool_forward_naive(x, pool_param):
    """A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here, eg you can assume:
      - (H - pool_height) % stride == 0
      - (W - pool_width) % stride == 0

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    # 获取一些需要用到的数据
    N, C, H, W = x.shape  # N个样本，C个通道，H高，W宽
    pool_height = pool_param["pool_height"]  # 池化核高
    pool_width = pool_param["pool_width"]  # 池化核宽
    stride = pool_param["stride"]  # 步长

    # 计算池化后的高和宽
    out_H = int(1 + (H - pool_height) / stride)
    out_W = int(1 + (W - pool_width) / stride)

    # 给out分配空间
    out = np.zeros((N, C, out_H, out_W))

    for n in range(N):
        for c in range(C):
            for i in range(out_H):
                for j in range(out_W):
                    # 获取当前池化窗口
                    window = x[n, c, i * stride:i * stride + pool_height, j * stride:j * stride + pool_width]
                    # 计算当前池化窗口的最大值
                    out[n, c, i, j] = np.max(window)
    cache = (x, pool_param)
    return out, cache

def max_pool_backward_naive(dout, cache):
    """A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    # 获取一些需要用到的数据
    x, pool_param = cache
    N, C, H, W = x.shape  # N个样本，C个通道，H高，W宽
    pool_height = pool_param["pool_height"]  # 池化核高
    pool_width = pool_param["pool_width"]  # 池化核宽
    stride = pool_param["stride"]  # 步长

    # 计算池化后的高和宽
    out_H = int(1 + (H - pool_height) / stride)
    out_W = int(1 + (W - pool_width) / stride)

    # 给dx分配空间
    dx = np.zeros_like(x)

    for n in range(N):
        for c in range(C):
            for i in range(out_H):
                for j in range(out_W):
                    # 获取当前池化窗口
                    window = x[n, c, i * stride:i * stride + pool_height, j * stride:j * stride + pool_width]
                    # 计算当前池化窗口的最大值
                    max_index = np.argmax(window)
                    # 计算dx
                    dx[n, c, i * stride + max_index // pool_width, j * stride + max_index % pool_width] += dout[n, c, i, j]
    return dx

def max_pool_forward_fast(x, pool_param):
    N, C, H, W = x.shape
    pool_height, pool_width, stride = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']
    H_out = 1 + (H - pool_height) // stride
    W_out = 1 + (W - pool_width) // stride
    out = np.zeros((N, C, H_out, W_out))

    for i in range(H_out):
        for j in range(W_out):
            window = x[:, :, i * stride:i * stride + pool_height, j * stride:j * stride + pool_width]
            out[:, :, i, j] = np.max(window, axis=(2, 3))

    cache = (x, pool_param)
    return out, cache

def max_pool_backward_fast(dout, cache):
    x, pool_param = cache
    N, C, H, W = x.shape
    pool_height, pool_width, stride = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']
    H_out = 1 + (H - pool_height) // stride
    W_out = 1 + (W - pool_width) // stride
    dx = np.zeros_like(x)

    for i in range(H_out):
        for j in range(W_out):
            window = x[:, :, i * stride:i * stride + pool_height, j * stride:j * stride + pool_width]
            max_values = np.max(window, axis=(2, 3), keepdims=True)
            mask = (window == max_values)
            dx[:, :, i * stride:i * stride + pool_height, j * stride:j * stride + pool_width] += mask * dout[:, :, i, j, np.newaxis, np.newaxis]

    return dx

def dropout_forward(x, dropout_param):
    """Forward pass for inverted dropout.

    Note that this is different from the vanilla version of dropout.
    Here, p is the probability of keeping a neuron output, as opposed to
    the probability of dropping a neuron output.
    See http://cs231n.github.io/neural-networks-2/#reg for more details.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We keep each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.
    """
    p, mode = dropout_param["p"], dropout_param["mode"]
    if "seed" in dropout_param:
        np.random.seed(dropout_param["seed"])
    mask = None
    out = None
    if mode == "train":
        mask = np.random.rand(*x.shape) < p
        out = x * mask
    elif mode == "test":
        out = x
    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache

def dropout_backward(dout, cache):
    """Backward pass for inverted dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param["mode"]

    dx = None
    if mode == "train":
        dx = dout*mask
    elif mode == "test":
        dx = dout
    return dx

def batchnorm_forward(x, gamma, beta, bn_param):
    """Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param["mode"]
    eps = bn_param.get("eps", 1e-5)
    momentum = bn_param.get("momentum", 0.9)

    N, D = x.shape
    running_mean = bn_param.get("running_mean", np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get("running_var", np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == "train":
        x_mean = np.mean(x , axis = 0)
        x_var = np.var(x, axis=0)
        x_std = np.sqrt(x_var + eps)
        x_norm = (x - x_mean) / x_std
        out = gamma*x_norm + beta

        cache = (x, x_mean, x_var, x_std, x_norm, out, gamma, beta , eps)
        running_mean = momentum * running_mean + (1 - momentum) * x_mean
        running_var = momentum * running_var + (1 - momentum) * x_var
    elif mode == "test":
        running_std = np.sqrt(running_var + eps)
        x_norm = (x - running_mean)/running_std
        out = gamma*x_norm + beta
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param["running_mean"] = running_mean
    bn_param["running_var"] = running_var

    return out, cache

def batchnorm_backward_alt(dout, cache):
    """Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass.
    See the jupyter notebook for more hints.

    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    x, x_mean, x_var, x_std, x_norm, out, gamma, beta, eps = cache
    dgamma = np.sum(dout * x_norm, axis=0)  # 计算dgamma
    dbeta = np.sum(dout, axis=0)  # 计算dbeta
    dx_norm = dout * gamma  # 计算dx_norm
    dx_var = np.sum(dx_norm * (x - x_mean) * (-0.5) * np.power(x_var + eps, -1.5), axis=0)  # 计算dx_var
    dx_mean = np.sum(dx_norm * (-1) / x_std, axis=0) + dx_var * np.sum(-2 * (x - x_mean), axis=0) / x.shape[0]  # 计算dx_mean
    dx = dx_norm / x_std + dx_var * 2 * (x - x_mean) / x.shape[0] + dx_mean / x.shape[0]  # 计算dx

    return dx, dgamma, dbeta


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    N, C, H, W = x.shape
    x_flat = x.transpose(0, 2, 3, 1).reshape(-1, C)  # 将 x 变形为 (N*H*W, C)
    # 调用 batchnorm_forward 函数进行批量归一化
    out_flat, cache = batchnorm_forward(x_flat, gamma, beta, bn_param)

    # 将输出数据重新变形为 (N, H, W, C)
    out = out_flat.reshape(N, H, W, C).transpose(0, 3, 1, 2)


    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N, C, H, W = dout.shape
    dout_flat = dout.transpose(0, 2, 3, 1).reshape(-1, C)  # 将 dout 变形为 (N*H*W, C)

    # 调用 batchnorm_backward_alt 函数进行反向传播
    dx_flat, dgamma, dbeta = batchnorm_backward_alt(dout_flat, cache)

    # 将 dx 重新变形为 (N, H, W, C)
    dx = dx_flat.reshape(N, H, W, C).transpose(0, 3, 1, 2)

    return dx, dgamma, dbeta

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta