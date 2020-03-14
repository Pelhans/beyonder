#!/usr/bin/env python
# coding=utf-8

import modeling

def bert(bert_config, is_training, input_ids, input_mask,
         segment_ids, use_one_hot_embeddings, task_type="sequence"):
    """Use bert model to get sequence output"""
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings,)

    # get sequence output
    # output_layer's shape [batch_size, sequence_length, hidden_size]
    # where hidden_size is 768
    if task_type == "sequence":
        output_layer = model.get_sequence_output()
    elif task_type == "binary":
        output_layer = model.get_pooled_output()
    return output_layer
