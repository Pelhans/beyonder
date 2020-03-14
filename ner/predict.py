#!/usr/bin/env python3
# coding=utf-8
 
""" Packaging the model and providing a unified predictive function interface  """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

import tensorflow as tf
import numpy as np
from collections import defaultdict

from evaluation import decode_ner
from tokenizer import Tokenizer
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
from grpc.beta import implementations
from run import convert_single_example, NERProcessor, word2id

max_seq_length = 128

def preprocess(querys, processor, label_list,  wordid_map, label_map, tokenizer):
    """Preprocess data for BERT,  The first run after startup will
        be slow due to tf. contrib. util. make_tensor_proto
    :return: dict contains input_ids, input_mask, label_ids, schema_label_ids,
            segment_ids, offset, sequence_length
    :rtype: tf.contrib.util.make_tensor_proto
    """
    if not isinstance(querys[0], str):
        raise ValueError("Input must be string")
    labels = [["O"] * len(text) for text in querys]
    length = [len(text) for text in querys]

    querys = ["\t".join([str(text), str(labels[idx])]) for idx, text in enumerate(querys)]
    print("querys: ", querys)

    # 默认输入是一句话, 因此只要 example
    examples = processor._create_examples(querys)
    features = defaultdict(list)
    for example in examples:
        feature, _, _ = convert_single_example(0, example, label_list, max_seq_length,
                                               wordid_map, label_map,  tokenizer)
        features["input_ids"].append(feature.input_ids)
        features["input_mask"].append(feature.input_mask)
        features["label_ids"].append(feature.label_ids)
        features["segment_ids"].append(feature.segment_ids)
        features["sequence_length"].append(feature.sequence_length)

    inputs = {
        'input_ids': tf.contrib.util.make_tensor_proto(features["input_ids"],
                dtype=tf.int64),
        'input_mask': tf.contrib.util.make_tensor_proto(features["input_mask"],
                dtype=tf.int64),
        'label_ids': tf.contrib.util.make_tensor_proto(features["label_ids"],
                dtype=tf.int64),
        'segment_ids': tf.contrib.util.make_tensor_proto(features["segment_ids"],
                dtype=tf.int64),
        'sequence_length': tf.contrib.util.make_tensor_proto(features["sequence_length"],
                dtype=tf.int64),}
    return inputs, length

def predict(querys, model_name, stub, processor, 
            label_list, wordid_map, label_map,  
            label_id2tag, tokenizer):
    """ Return the query parsing result with the return of BERT 
    """
    request = predict_pb2.PredictRequest()
    request.model_spec.name = model_name
    inputs, length = preprocess(querys, processor, label_list, wordid_map, label_map, tokenizer)
    for k, v in inputs.items():
        request.inputs[k].CopyFrom(v)
    result = stub.Predict(request, 60.0).outputs
    all_res = []
    pred_ids = result["pred_ids"].int_val
    pred_ids = np.reshape(pred_ids, [len(querys), -1])
    for idx, query in enumerate(querys):
        pred_id = pred_ids[idx][1:length[idx]+1]
        pred_id = [str(i) for i in pred_id]
        res = decode_ner(pred_id)
        print("res: ", res)
        if not res:
            all_res.append(query)
            continue
        all_res.append([query[res[i][0]: res[i][1]+1] for i in range(len(res))])
    return all_res

class Client:
    """Predict model Client
    """
    def __init__(self, model_name, server_ip="poi_bert_ner", server_port=6000):
        self.model_name = model_name
        self.server_ip = server_ip
        self.server_port = int(server_port)

        self.channel = implementations.insecure_channel(self.server_ip, self.server_port)
        self.stub = prediction_service_pb2.beta_create_PredictionService_stub(self.channel)
        self.processor = NERProcessor()
        self.label_list = self.processor.get_labels()
        self.label_map = {}
        for (i, label) in enumerate(self.label_list):
            self.label_map[label] = i
        self.wordid_map = word2id(os.path.join(BASE_DIR, "pb_model/vocab.txt"))
        self.tokenizer = Tokenizer(self.wordid_map)
        self.label_id2tag = {i: v for i,v in enumerate(self.label_list)}

    def query_parsing(self, querys):
        """ Functional interface provided to the outside
        :param query: query poi from get method
        :type query: string list
        :param city:  Query poi's city
        :type city: string
        :return: a dict with lon, lat and poi type
        :rtype: dict
        """
        if not querys:
            raise ValueError("Query text is empty !!")

        return predict(querys, self.model_name, self.stub, self.processor, self.label_list,
                       self.wordid_map, self.label_map, self.label_id2tag, self.tokenizer)

if __name__ == "__main__":
    client = Client("kg_ner_disambi",
                    server_ip="kg_ner_disambi",
                    server_port=6000,)

    res = client.query_parsing(["没人提尹正吗，《乌鸦嘴妙女郎》里他演的太好了，《飞驰人生》也很不错"])

    print("res: ", res)
