#!/usr/bin/env python
# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, "../../basic_net/"))

import tensorflow as tf
import modeling
from basic import linear_layer_act, linear_layer
from lstm import blstm
from crf import crf_layer

FLAGS = tf.app.flags.FLAGS

def blstm_crf(bert_config, is_training, input_ids, segment_ids, input_mask,
               label_ids, sequence_length, num_labels,  use_one_hot_embeddings):
    """combine bert + crf_layer
    """

    batch_size = tf.shape(input_ids)[0]
    embedding = tf.get_variable("embedding", 
                                [FLAGS.vocab_size, FLAGS.embedding_size],
                                dtype=tf.float32)
    inputs = tf.nn.embedding_lookup(embedding, input_ids)
    if is_training:
        inputs = modeling.layer_norm_and_dropout(inputs, 0.2)
    else:
        inputs = modeling.layer_norm(inputs)
    y_pred = blstm(is_training, inputs, FLAGS.hidden_size_blstm,
                  FLAGS.layer_num, FLAGS.max_seq_length, var_scope="bilstm")
    if is_training:
        linear_out = modeling.layer_norm_and_dropout(y_pred, 0.2)
    else:
        linear_out = modeling.layer_norm(y_pred)
    hidden_size = FLAGS.hidden_size_blstm
    linear_out = tf.reshape(linear_out, [-1, hidden_size*2])
    linear_out = linear_layer_act(linear_out, hidden_size*2, "linear_act", 0.02)
    linear_out = linear_layer(linear_out, num_labels, "linear_out", 0.02)
    crf_out = crf_layer(linear_out, label_ids, batch_size, 
                        sequence_length, num_labels, FLAGS.max_seq_length, "crf")
    return crf_out
