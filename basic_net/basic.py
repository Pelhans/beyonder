#!/usr/bin/env python
# coding=utf-8
import tensorflow as tf


def dropout(x, keep_prob=1.0):
    return tf.contrib.layers.dropout(x, keep_prob=keep_prob)

def dropout_active(x, keep_prob=1.0):
    uni_mask = tf.random_uniform(x.shape)
    mask = uni_mask > x
    mask = tf.cast(x, dtype=tf.float32)
    return mask * x

def pooling(x):
    return tf.reduce_max(x, axis=-2)

def max_pooling(x, ksize=3, stride=1):
    """
    tf.nn.max_pool 参数是四个
    第一个参数value：需要池化的输入，一般池化层接在卷积层后面，所以输入通常是feature map，依然是[batch, height, width, channels]这样的shape
    第二个参数ksize：池化窗口的大小，取一个四维向量，一般是[1, height, width, 1]，因为我们不想在batch和channels上做池化，所以这两个维度设为了1
    第三个参数strides：和卷积类似，窗口在每一个维度上滑动的步长，一般也是[1, stride,stride, 1]
    第四个参数padding：和卷积类似，可以取'VALID' 或者'SAME'
    返回一个Tensor，类型不变，shape仍然是[batch, height, width, channels]这种形式
    """
    pool_out = tf.nn.max_pool(x, 
                              ksize=[1, ksize, ksize, 1],
                              strides=[1, stride, stride, 1],
                              padding='SAME')
    return pool_out

def linear_layer_act(in_tensor, out_shape, var_scope, initializer_range):
    with tf.variable_scope(var_scope or "linear_layer_act", reuse=tf.AUTO_REUSE):
        output = tf.layers.dense(
            in_tensor,
            out_shape,
            activation=tf.tanh,
            kernel_initializer=create_initializer(initializer_range))
    return output

def linear_layer(in_tensor, out_shape, var_scope, initializer_range):
    with tf.variable_scope(var_scope or "linear_layer_act", reuse=tf.AUTO_REUSE):
        output = tf.layers.dense(
            in_tensor,
            out_shape,
            kernel_initializer=create_initializer(initializer_range))
    return output

def create_initializer(initializer_range=0.02):
  """Creates a `truncated_normal_initializer` with the given range."""
  return tf.truncated_normal_initializer(stddev=initializer_range)

def weight_variable(shape, name):
    with tf.variable_scope(name):
        initial = tf.get_variable("weight",
                                  shape, 
                                  initializer=tf.contrib.layers.xavier_initializer())
    return initial

def bias_variable(shape, name):
    with tf.variable_scope(name):
        initial = tf.get_variable("bias",
                                  shape,
                                  initializer=tf.zeros_initializer())
    return initial

def embedding(pretrain_word_vec, embedding_size):
    temp_word_embedding = tf.get_variable(initializer=pretrain_word_vec,
                                          name = 'temp_word_embedding',
                                          dtype=tf.float32)
    unk_word_embedding = tf.get_variable('unk_embedding',
                                         [embedding_size],
                                         dtype=tf.float32,
                                         initializer=tf.contrib.layers.xavier_initializer())
    word_vec = tf.concat([temp_word_embedding,
                          tf.reshape(unk_word_embedding, [1, embedding_size])], 0)
    return word_vec

def cosine(q,a):
    pooled_len_1 = tf.sqrt(tf.reduce_sum(q * q, 1))
    pooled_len_2 = tf.sqrt(tf.reduce_sum(a * a, 1))
    pooled_mul_12 = tf.reduce_sum(q * a, 1)
    score = tf.div(pooled_mul_12, pooled_len_1 * pooled_len_2 +1e-8, name="scores")
    return score 
