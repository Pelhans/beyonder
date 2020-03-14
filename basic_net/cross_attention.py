#!/usr/bin/env python
# coding=utf-8
import tensorflow as tf

def local_inference(x1, x1_mask, x2, x2_mask, var_scope):
    """Local inference collected over sequences
    Args:
        x1: float32 Tensor of shape [batch_size, seq_length1, dim].
        x1_mask: float32 Tensor of shape [batch_size, seq_length1].
        x2: float32 Tensor of shape [batch_size, seq_length2, dim].
        x2_mask: float32 Tensor of shape [batch_size, seq_length2].
    Return:
        x1_dual: float32 Tensor of shape [batch_size, seq_length1, dim]
        x2_dual: float32 Tensor of shape [batch_size, seq_length2, dim]
    """

    with tf.variable_scope(var_scope or "local_inference", reuse=tf.AUTO_REUSE):
    
        # x1: [batch_size, seq_length1, dim].
        # x1_mask: [batch_size, seq_length1].
        # x2: [batch_size, seq_length2, dim].
        # x2_mask: [batch_size, seq_length2].
    #    x1 = tf.transpose(x1, [1, 0, 2])
    #    x1_mask = tf.transpose(x1_mask, [1, 0])
    #    x2 = tf.transpose(x2, [1, 0, 2])
    #    x2_mask = tf.transpose(x2_mask, [1, 0])
    
        # attention_weight: [batch_size, seq_length1, seq_length2]
        attention_weight = tf.matmul(x1, tf.transpose(x2, [0, 2, 1]))
    
        # calculate normalized attention weight x1 and x2
        # attention_weight_2: [batch_size, seq_length1, seq_length2]
        attention_weight_2 = tf.exp(
            attention_weight - tf.reduce_max(attention_weight, axis=2, keepdims=True))
        attention_weight_2 = attention_weight_2 * tf.expand_dims(x2_mask, 1)
        # alpha: [batch_size, seq_length1, seq_length2]
        alpha = attention_weight_2 / (tf.reduce_sum(attention_weight_2, -1, keepdims=True) + 1e-8)
        # x1_dual: [batch_size, seq_length1, dim]
        x1_dual = tf.reduce_sum(tf.expand_dims(x2, 1) * tf.expand_dims(alpha, -1), 2)
        # x1_dual: [seq_length1, batch_size, dim]
    #    x1_dual = tf.transpose(x1_dual, [1, 0, 2])
    
        # attention_weight_1: [batch_size, seq_length2, seq_length1]
        attention_weight_1 = attention_weight - tf.reduce_max(attention_weight, axis=1, keepdims=True)
        attention_weight_1 = tf.exp(tf.transpose(attention_weight_1, [0, 2, 1]))
        attention_weight_1 = attention_weight_1 * tf.expand_dims(x1_mask, 1)
    
        # beta: [batch_size, seq_length2, seq_length1]
        beta = attention_weight_1 / \
            (tf.reduce_sum(attention_weight_1, -1, keepdims=True) + 1e-8)
        # x2_dual: [batch_size, seq_length2, dim]
        x2_dual = tf.reduce_sum(tf.expand_dims(x1, 1) * tf.expand_dims(beta, -1), 2)
        # x2_dual: [seq_length2, batch_size, dim]
     #   x2_dual = tf.transpose(x2_dual, [1, 0, 2])
    
        return x1_dual, x2_dual
