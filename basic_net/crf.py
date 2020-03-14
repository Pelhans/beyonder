#!/usr/bin/env python
# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import crf

def crf_layer(y_pred, label_ids, batch_size, sequence_length,
              num_labels, max_seq_length, name):
    with tf.variable_scope(name):
        sequence_length = tf.squeeze(sequence_length)
        logits = tf.reshape(y_pred, [-1, max_seq_length, num_labels])
        loss, trans = _crf_layer(logits, num_labels, label_ids,
                                length=sequence_length, name="crf_bio")
        
        sequence_length = tf.reshape(sequence_length, [batch_size])
        pred_ids, _ = crf.crf_decode(potentials=logits, 
                                     transition_params=trans,
                                     sequence_length=sequence_length)
    
        return loss, loss, logits, pred_ids

def _crf_layer(logits, num_labels, label_ids, length, name):
    """Calculate the likelihood loss function with CRF layer"""
    with tf.variable_scope(name):
        log_likelihood, trans = tf.contrib.crf.crf_log_likelihood(
            inputs=logits,
            tag_indices=label_ids,
            sequence_lengths=length)
        return tf.reduce_mean(-log_likelihood), trans
