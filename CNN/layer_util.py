#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2024/3/28 15:58
# @Author : 刘亚东
# @Site : 
# @File : layer_util.py
# @Software: PyCharm
from im2col_python import *
from layer import *

def affine_relu_forward(x, w, b):
    affine_out,cache_affine = fully_forward(x,w,b)
    relu_out,cache_relu = relu_forward(affine_out)
    cache = (cache_affine , cache_relu)
    return relu_out , cache

def affine_relu_backward(dout,cache):
    cache_affine , cache_relu = cache
    dx = relu_backward(dout,cache_relu)
    dx , dw ,db = fully_backward(dx , cache_affine)
    return dx , dw ,db

def conv_relu_forward(x, w, b, conv_param):
    conv_out , cache_conv = conv_forward_naive(x,w,b,conv_param)
    relu_out , cache_relu = relu_forward(conv_out)
    cache = (cache_conv , cache_relu)
    return relu_out , cache

def conv_relu_backward(dout , cache):
    cache_conv, cache_relu = cache
    dx = relu_backward(dout , cache_relu)
    dx , dw , db = conv_backward_naive(dx , cache_conv)
    return dx,dw,db

def conv_relu_fast_forward(x, w, b, conv_param):
    conv_out , cache_conv = conv_forward_fast(x,w,b,conv_param)
    relu_out , cache_relu = relu_forward(conv_out)
    cache = (cache_conv , cache_relu)
    return relu_out , cache

def conv_relu_fast_backward(dout , cache):
    cache_conv, cache_relu = cache
    dx = relu_backward(dout , cache_relu)
    dx , dw , db = conv_backward_fast(dx , cache_conv)
    return dx,dw,db

def conv_bn_relu_forward(x, w, b, gamma, beta, conv_param, bn_param):
    """Convenience layer that performs a convolution, a batch normalization, and a ReLU.

    Inputs:
    - x: Input to the convolutional layer
    - w, b, conv_param: Weights and parameters for the convolutional layer
    - pool_param: Parameters for the pooling layer
    - gamma, beta: Arrays of shape (D2,) and (D2,) giving scale and shift
      parameters for batch normalization.
    - bn_param: Dictionary of parameters for batch normalization.

    Returns a tuple of:
    - out: Output from the pooling layer
    - cache: Object to give to the backward pass
    """
    a, conv_cache = conv_forward_naive(x, w, b, conv_param)
    an, bn_cache = spatial_batchnorm_forward(a, gamma, beta, bn_param)
    out, relu_cache = relu_forward(an)
    cache = (conv_cache, bn_cache, relu_cache)
    return out, cache


def conv_bn_relu_backward(dout, cache):
    """Backward pass for the conv-bn-relu convenience layer.
    """
    conv_cache, bn_cache, relu_cache = cache
    dan = relu_backward(dout, relu_cache)
    da, dgamma, dbeta = spatial_batchnorm_backward(dan, bn_cache)
    dx, dw, db = conv_backward_fast(da, conv_cache)
    return dx, dw, db, dgamma, dbeta

def conv_relu_pool_forward(x, w, b, conv_param, pool_param):
    conv_out , conv_cache = conv_forward_naive(x,w,b,conv_param)
    relu_out , relu_cache = relu_forward(conv_out)
    max_pool_out , max_pool_cache = max_pool_forward_naive(relu_out,pool_param)
    cache = (conv_cache,relu_cache,max_pool_cache)
    return max_pool_out,cache

def conv_relu_pool_backward(dout , cache):
    conv_cache, relu_cache, max_pool_cache = cache
    dx = max_pool_backward_naive(dout,max_pool_cache)
    dx = relu_backward(dx , relu_cache)
    dx , dw,db = conv_backward_naive(dx,conv_cache)
    return dx,dw,db


def conv_relu_pool_fast_forward(x, w, b, conv_param, pool_param):
    conv_out , conv_cache = conv_forward_fast(x,w,b,conv_param)
    relu_out , relu_cache = relu_forward(conv_out)
    max_pool_out , max_pool_cache = max_pool_forward_fast(relu_out,pool_param)
    cache = (conv_cache,relu_cache,max_pool_cache)
    return max_pool_out,cache

def conv_relu_pool_fast_backward(dout , cache):
    conv_cache, relu_cache, max_pool_cache = cache
    dx = max_pool_backward_fast(dout,max_pool_cache)
    dx = relu_backward(dx , relu_cache)
    dx , dw,db = conv_backward_fast(dx,conv_cache)
    return dx,dw,db
