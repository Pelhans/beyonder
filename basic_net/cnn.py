#!/usr/bin/env python
# coding=utf-8
import tensorflow as tf
from .basic import pooling, dropout, max_pooling


def __cnn1d_cell__(x, hidden_size=230, kernel_size=3, stride_size=1):
    # tf.layers.conv1d 的输入 (batch_size, seq_length, embedding_dim)
    # 输出为 (batch_size, out_length, filter_num)
    # hidden_size filters 卷积核的数目
    # kernel_size 卷积核的大小(宽度， 深度为 embedding 的大小)
    x = tf.layers.conv1d(inputs=x, 
                         filters=hidden_size, 
                         kernel_size=kernel_size, 
                         strides=stride_size, 
                         padding='same', 
                         kernel_initializer=tf.contrib.layers.xavier_initializer())
    return x

def cnn_1d(x, hidden_size=230, kernel_size=3, stride_size=1, activation=tf.nn.relu,
           is_pooling=True, var_scope=None, keep_prob=1.0):
    with tf.variable_scope(var_scope or "cnn", reuse=tf.AUTO_REUSE):
        x = __cnn1d_cell__(x, hidden_size, kernel_size, stride_size)
        if is_pooling:
            x = pooling(x)
        x = activation(x)
        x = dropout(x, keep_prob)
        return x

# tf.nn.conv2d (input, filter, strides, padding, use_cudnn_on_gpu=None, data_format=None, name=None)
"""
input : 输入的要做卷积的图片，要求为一个张量，shape为 [ batch, in_height, in_weight, in_channel  ]，其中batch为图片的数量，in_height 为图片高度，in_weight 为图片宽度，in_channel 为图片的通道数，灰度图该值为1，彩色图为3。(也可以用其它值，但是具体含义不是很理解)
filter： 卷积核，要求也是一个张量，shape为 [ filter_height, filter_weight, in_channel, out_channels  ]，其中 filter_height 为卷积核高度，filter_weight 为卷积核宽度，in_channel 是图像通道数 ，和 input 的 in_channel 要保持一致，out_channel 是卷积核数量。
strides： 卷积时在图像每一维的步长，这是一个一维的向量，[ 1, strides, strides, 1 ]，第一位和最后一位固定必须是1
padding： string类型，值为“SAME” 和 “VALID”，表示的是卷积的形式，是否考虑边界。'SAME'是考虑边界，不足的时候用0去填充周围，'VALID'则不考虑
use_cudnn_on_gpu： bool类型，是否使用cudnn加速，默认为True
"""

def cnn_2d(x, in_channel=1, hidden_size=200, kernel_size=3,
           pooling_size=2, stride_size=1, activation=tf.nn.relu,
           is_pooling=True, var_scope=None, keep_prob=1.0):
    with tf.variable_scope(var_scope or "cnn", reuse=tf.AUTO_REUSE):
        filter = filters(x, kernel_size, in_channel, hidden_size)
        x = tf.nn.conv2d(x, 
                    filter,
                    strides=[1, stride_size, stride_size, 1],
                    padding='SAME', )
        if is_pooling:
            x = max_pooling(x, ksize=pooling_size, stride=pooling_size)
        x = activation(x)
        x = dropout(x, keep_prob)
        return x

def filters(x, kernel_size, in_channel, hidden_size):
    filter = tf.get_variable(
                "filters",
                shape=[kernel_size,  kernel_size, in_channel, hidden_size],
#                regularizer=tf.contrib.layers.l2_regularizer(0.0001),
                initializer=tf.contrib.layers.xavier_initializer(),
                dtype=tf.float32)
    return filter
