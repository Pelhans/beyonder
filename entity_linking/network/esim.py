#!/usr/bin/env python3
# coding=utf-8

"""Model dec """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, "../../basic_net/"))

import tensorflow as tf
from basic import linear_layer_act, linear_layer
from lstm import blstm
import modeling
from cross_attention import local_inference

FLAGS = tf.app.flags.FLAGS

def esim(is_training, input_idsA, input_idsB, 
         input_maskA, input_maskB, label_ids, num_labels): 
    """esim
    """

    keep_prob = 0.5
    hidden_size_cnn = FLAGS.hidden_size_cnn
    embedding_size = FLAGS.embedding_size
    vocab_size = FLAGS.vocab_size
    # 用预训练模型
#    pretrain_word_vec = build_pretrain_vec(FLAGS.word_vec, embedding_size)
#    word_vec = embedding(pretrain_word_vec, embedding_size)

    # 采用随机初始化
    embedding = tf.get_variable("embedding",  
                                [vocab_size, embedding_size],
                                dtype=tf.float32)
    inputsA = tf.nn.embedding_lookup(embedding, input_idsA)
    inputsB = tf.nn.embedding_lookup(embedding, input_idsB)
    tf.logging.info("shape of inputsB: {}".format(inputsA))
    tf.logging.info("shape of inputsB: {}".format(inputsB))

    if is_training:
        inputsA = tf.nn.dropout(inputsA, 0.9)
        inputsB = tf.nn.dropout(inputsB, 0.9)

    bilstm_A1 = blstm(is_training, inputsA, hidden_size_blstm=hidden_size_cnn,
                 layer_num=1, max_seq_length=FLAGS.max_seq_length_A, var_scope="blstm_1")
    bilstm_B1 = blstm(is_training, inputsB, hidden_size_blstm=hidden_size_cnn,
                 layer_num=1, max_seq_length=FLAGS.max_seq_length_B, var_scope="blstm_1")

    dualA, dualB = local_inference(bilstm_A1, input_maskA, bilstm_B1, input_maskB, "local_inference1")
    x1_match = tf.concat([bilstm_A1, dualA, bilstm_A1 * dualA, bilstm_A1 - dualA], 2)
    x2_match = tf.concat([bilstm_B1, dualB, bilstm_B1 * dualB, bilstm_B1 - dualB], 2)

    x1_match_mapping = linear_layer_act(x1_match, hidden_size_cnn, "fnn", 0.02)
    x2_match_mapping = linear_layer_act(x2_match, hidden_size_cnn, "fnn", 0.02)

    if is_training:
        x1_match_mapping = modeling.layer_norm_and_dropout(x1_match_mapping, keep_prob)
        x2_match_mapping = modeling.layer_norm_and_dropout(x2_match_mapping, keep_prob)
    else:
        x1_match_mapping = modeling.layer_norm(x1_match_mapping)
        x2_match_mapping = modeling.layer_norm(x2_match_mapping)

    bilstm_A2 = blstm(is_training, x1_match_mapping, hidden_size_blstm=hidden_size_cnn,
                 layer_num=1, max_seq_length=FLAGS.max_seq_length_A, var_scope="blstm_2")
    bilstm_B2 = blstm(is_training, x2_match_mapping, hidden_size_blstm=hidden_size_cnn,
                 layer_num=1, max_seq_length=FLAGS.max_seq_length_B, var_scope="blstm_2")

    logit_x1_sum = tf.reduce_sum(bilstm_A2 * tf.expand_dims(input_maskA, -1), 1) / \
        tf.expand_dims(tf.reduce_sum(input_maskA, 1), 1)
    logit_x1_max = tf.reduce_max(bilstm_A2 * tf.expand_dims(input_maskA, -1), 1)
    logit_x2_sum = tf.reduce_sum(bilstm_B2 * tf.expand_dims(input_maskB, -1), 1) / \
        tf.expand_dims(tf.reduce_sum(input_maskB, 1), 1)
    logit_x2_max = tf.reduce_max(bilstm_B2 * tf.expand_dims(input_maskB, -1), 1)

    logits = tf.concat([logit_x1_sum, logit_x1_max, logit_x2_sum, logit_x2_max], 1)

    if is_training:
        logits = modeling.layer_norm_and_dropout(logits, keep_prob)
    else:
        logits = modeling.layer_norm(logits)

    logits = linear_layer_act(logits, hidden_size_cnn, "linear_act", 0.02)

    if is_training:
        logits = modeling.layer_norm_and_dropout(logits, keep_prob)
    else:
        logits = modeling.layer_norm(logits)

    logits = linear_layer(logits, num_labels, "linear", 0.02)

    log_probs = tf.nn.log_softmax(logits, axis=-1)
    probabilities = tf.nn.softmax(logits, axis=-1)
    one_hot_labels = tf.one_hot(label_ids, depth=num_labels, dtype=tf.float32)
    per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
    loss = tf.reduce_mean(per_example_loss)
    pred_ids = tf.argmax(logits, axis=-1, output_type=tf.int32)
    return (loss, loss, logits, probabilities, pred_ids)

