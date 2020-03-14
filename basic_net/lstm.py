#!/usr/bin/env python
# coding=utf-8
import tensorflow as tf
from tensorflow.contrib import rnn


def lstm_cell(hidden_size):
    cell = rnn.LSTMCell(hidden_size,
                        reuse=tf.get_variable_scope().reuse)
    return rnn.DropoutWrapper(cell, output_keep_prob=True)


def blstm(is_training, inputs, hidden_size_blstm,
         layer_num, max_seq_length, var_scope=None):
    """BLSTM model"""

    with tf.variable_scope(var_scope or "blstm", reuse=tf.AUTO_REUSE):
        batch_size = tf.shape(inputs)[0]
        hidden_size = hidden_size_blstm
    
        cell_fw = rnn.MultiRNNCell([lstm_cell(hidden_size) for _ in range(layer_num)],
                                   state_is_tuple=True)
        cell_bw = rnn.MultiRNNCell([lstm_cell(hidden_size) for _ in range(layer_num)],
                                   state_is_tuple=True)
    
        # shape c_state and h_state [batch_size, hidden_size],
        initial_state_fw = cell_fw.zero_state(batch_size, tf.float32)
        initial_state_bw = cell_bw.zero_state(batch_size, tf.float32)
    
        with tf.variable_scope('bidirectional_rnn'):
            outputs_fw = list()
            state_fw = initial_state_fw
            with tf.variable_scope('fw'):
                for timestep in range(max_seq_length):
                    if timestep > 0:
                        tf.get_variable_scope().reuse_variables()
                    # output_fw -> h_{t}, state_fw -> cell_{t}
                    (output_fw, state_fw) = cell_fw(inputs[:, timestep, :], state_fw)
                    outputs_fw.append(output_fw)
            outputs_bw = list()
            state_bw = initial_state_bw
            with tf.variable_scope('bw'):
                inputs = tf.reverse(inputs, [1])
                for timestep in range(max_seq_length):
                    if timestep > 0:
                        tf.get_variable_scope().reuse_variables()
                    (output_bw, state_bw) = cell_bw(inputs[:, timestep, :], state_bw)
                    outputs_bw.append(output_bw)
            outputs_bw = tf.reverse(outputs_bw, [0])
            # after concat, shape of output [timestep, batch_size, hidden_size*2]
            output = tf.concat([outputs_fw, outputs_bw], 2)
            output = tf.transpose(output, perm=[1,0,2])
    #        output = tf.reshape(output, [-1, hidden_size*2])

            return output

def bilstm_cudnn(input_data, num_layers, rnn_size, keep_prob=1.):
    """Multi-layer BiLSTM cudnn version, faster
    Args:
        input_data: float32 Tensor of shape [seq_length, batch_size, dim].
        num_layers: int64 scalar, number of layers.
        rnn_size: int64 scalar, hidden size for undirectional LSTM.
        keep_prob: float32 scalar, keep probability of dropout between BiLSTM layers 
    Return:
        output: float32 Tensor of shape [seq_length, batch_size, dim * 2]
    """
    with tf.variable_scope("bilstm", reuse=tf.AUTO_REUSE):
        lstm = tf.contrib.cudnn_rnn.CudnnLSTM(
            num_layers=num_layers,
            num_units=rnn_size,
            input_mode="linear_input",
            direction="bidirectional",
            dropout=1 - keep_prob)

        # to do, how to include input_mask
        outputs, output_states = lstm(inputs=input_data)

    return outputs
