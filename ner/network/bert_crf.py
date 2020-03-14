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
from crf import crf_layer
from bert import bert

FLAGS = tf.app.flags.FLAGS

def bert_crf(bert_config, is_training, input_ids, segment_ids, input_mask,
            label_ids, sequence_length, num_labels,  use_one_hot_embeddings):
    batch_size = tf.shape(input_ids)[0]
    bert_out = bert(bert_config, is_training, input_ids,
                    input_mask, segment_ids, use_one_hot_embeddings)
    hidden_size = 768
    linear_out = tf.reshape(bert_out, [-1, hidden_size])
    if is_training:
        linear_out = modeling.layer_norm_and_dropout(linear_out, 0.2)
    else:
        linear_out = modeling.layer_norm(linear_out)
    linear_out = linear_layer_act(linear_out, hidden_size, "linear_act", 0.02)
    linear_out = linear_layer(linear_out, num_labels, "linear_out", 0.02)
    crf_out = crf_layer(linear_out, label_ids, batch_size, 
                          sequence_length, num_labels, FLAGS.max_seq_length, "crf")
    return crf_out
