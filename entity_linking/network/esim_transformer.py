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
from modeling import transformer_model, gelu
from modeling import create_attention_mask_from_input_mask
from lstm import blstm
import modeling
from cross_attention import local_inference

FLAGS = tf.app.flags.FLAGS

def esim_transformer(is_training, input_idsA, input_idsB, 
                     input_maskA, input_maskB, label_ids, num_labels): 
    """esim which replace the first BiLSTM for the Transformer
    """
    hidden_keep_prob = 1
    attention_keep_prob = 1
    if is_training:
        hidden_keep_prob = 0.5
        attention_keep_prob = 0.9
    hidden_size_cnn = FLAGS.hidden_size_cnn
    embedding_size = FLAGS.embedding_size
    vocab_size = FLAGS.vocab_size

    # 采用随机初始化
    embedding = tf.get_variable("embedding",  
                                [vocab_size, embedding_size],
                                dtype=tf.float32)
    inputsA = tf.nn.embedding_lookup(embedding, input_idsA)
    inputsB = tf.nn.embedding_lookup(embedding, input_idsB)
    input_maskA_3d = create_attention_mask_from_input_mask(inputsA, input_maskA)
    input_maskB_3d = create_attention_mask_from_input_mask(inputsB, input_maskB)
    tf.logging.info("shape of inputsA: {}".format(inputsA))
    tf.logging.info("shape of inputsB: {}".format(inputsB))

    if is_training:
        inputsA = tf.nn.dropout(inputsA, attention_keep_prob)
        inputsB = tf.nn.dropout(inputsB, attention_keep_prob)

    with tf.variable_scope("transformer_1", reuse=tf.AUTO_REUSE):
        transformer_A1 = transformer_model(inputsA,
                                      attention_mask=input_maskA_3d,
                                      hidden_size=256,
                                      num_hidden_layers=1,
                                      num_attention_heads=1,
                                      intermediate_size=1024,
                                      intermediate_act_fn=gelu,
                                      hidden_dropout_prob=1-attention_keep_prob,
                                      attention_probs_dropout_prob=1-attention_keep_prob,
                                      initializer_range=0.02,
                                      do_return_all_layers=False)

    with tf.variable_scope("transformer_1", reuse=tf.AUTO_REUSE):
        transformer_B1 = transformer_model(inputsB,
                                      attention_mask=input_maskB_3d,
                                      hidden_size=256,
                                      num_hidden_layers=1,
                                      num_attention_heads=1,
                                      intermediate_size=1024,
                                      intermediate_act_fn=gelu,
                                      hidden_dropout_prob=1-attention_keep_prob,
                                      attention_probs_dropout_prob=1-attention_keep_prob,
                                      initializer_range=0.02,
                                      do_return_all_layers=False)
    print("transformer_A1: ", transformer_A1)
    print("transformer_A1: ", transformer_B1)

    dualA, dualB = local_inference(transformer_A1, input_maskA, transformer_B1, input_maskB, "local_inference1")
    x1_match = tf.concat([transformer_A1, dualA, transformer_A1 * dualA, transformer_A1 - dualA], 2)
    x2_match = tf.concat([transformer_B1, dualB, transformer_B1 * dualB, transformer_B1 - dualB], 2)

    x1_match_mapping = linear_layer_act(x1_match, hidden_size_cnn, "fnn", 0.02)
    x2_match_mapping = linear_layer_act(x2_match, hidden_size_cnn, "fnn", 0.02)

    if is_training:
        x1_match_mapping = modeling.layer_norm_and_dropout(x1_match_mapping, hidden_keep_prob)
        x2_match_mapping = modeling.layer_norm_and_dropout(x2_match_mapping, hidden_keep_prob)
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
        logits = modeling.layer_norm_and_dropout(logits, hidden_keep_prob)
    else:
        logits = modeling.layer_norm(logits)

    logits = linear_layer_act(logits, hidden_size_cnn, "linear_act", 0.02)

    if is_training:
        logits = modeling.layer_norm_and_dropout(logits, hidden_keep_prob)
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
