#!/usr/bin/env python3
# coding=utf-8
 
""" Packaging the model and providing a unified predictive function interface  """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
import numpy as np
import json
from collections import defaultdict

base_path = "/".join(os.path.realpath(__file__).split("/")[:-1])

from tokenizer import Tokenizer
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
from grpc.beta import implementations
from run import convert_single_example, DisambiProcessor, word2id
max_seq_length_a = 64
max_seq_length_b = 300

def preprocess(query, processor, label_list,  wordid_map, label_map, tokenizer):
    """Preprocess data for BERT,  The first run after startup will
        be slow due to tf. contrib. util. make_tensor_proto
    :return: dict contains input_ids, input_mask, label_ids, schema_label_ids,
            segment_ids, offset, sequence_length
    :rtype: tf.contrib.util.make_tensor_proto
    """

    examples = processor._create_examples(query)
    features = defaultdict(list)
    for example in examples:
        feature, _, _ = convert_single_example(0, example, label_list, max_seq_length_a,
                                               max_seq_length_b, wordid_map, label_map,
                                               tokenizer, 0, 0)
        features["input_idsA"].append(feature.input_idsA)
        features["input_idsB"].append(feature.input_idsB)
        features["input_maskA"].append(feature.input_maskA)
        features["input_maskB"].append(feature.input_maskB)
        features["label_ids"].append(feature.label_ids)

    inputs = {
        'input_idsA': tf.contrib.util.make_tensor_proto(features["input_idsA"],
                dtype=tf.int64),
        'input_idsB': tf.contrib.util.make_tensor_proto(features["input_idsB"],
                dtype=tf.int64),
        'input_maskA': tf.contrib.util.make_tensor_proto(features["input_maskA"],
                dtype=tf.float32),
        'input_maskB': tf.contrib.util.make_tensor_proto(features["input_maskB"],
                dtype=tf.float32),
        'label_ids': tf.contrib.util.make_tensor_proto(features["label_ids"],
                dtype=tf.int64),}
    return inputs

def predict(query, model_name, stub, processor, 
            label_list, wordid_map, label_map,  
            label_id2tag, tokenizer):
    """ Return the query parsing result with the return of BERT 
    """
    request = predict_pb2.PredictRequest()
    request.model_spec.name = model_name
    inputs = preprocess(query, processor, label_list, wordid_map, label_map, tokenizer)
    for k, v in inputs.items():
        request.inputs[k].CopyFrom(v)
    result = stub.Predict(request, 60.0).outputs
    pred_ids = result["pred_ids"].int_val
    prob = result["probabilities"].float_val
    prob = np.reshape(prob, [len(query), -1])
    prob = [list(p) for p in prob]
    res = {"pred_ids": list(pred_ids),
           "probabilities": list(prob)}
    res = json.dumps(res)
    return res

class Client:
    """Predict model Client
    """
    def __init__(self, model_name, server_ip="poi_bert_rerank", server_port=6000):
        self.model_name = model_name
        self.server_ip = server_ip
        self.server_port = int(server_port)

        self.channel = implementations.insecure_channel(self.server_ip, self.server_port)
        self.stub = prediction_service_pb2.beta_create_PredictionService_stub(self.channel)
        self.processor = DisambiProcessor()
        self.label_list = self.processor.get_labels()
        self.label_map = {}
        for (i, label) in enumerate(self.label_list):
            self.label_map[label] = i
        self.wordid_map = word2id("./pb_model/vocab.txt")
        self.tokenizer = Tokenizer(self.wordid_map)
        self.label_id2tag = {i: v for i,v in enumerate(self.label_list)}

    def query_parsing(self, query):
        """ Functional interface provided to the outside
        :param query: query poi from get method
        :type query: string
        :param city:  Query poi's city
        :type city: string
        :return: a dict with lon, lat and poi type
        :rtype: dict
        """
        if not query:
            raise ValueError("Query text is empty !!")

        return predict(query, self.model_name, self.stub, self.processor, self.label_list,
                       self.wordid_map, self.label_map, self.label_id2tag, self.tokenizer)

if __name__ == "__main__":
    client = Client("poi_rerank_disambi",
                    server_ip="poi_rerank_disambi",
                    server_port=6000,)

    query = "$一起走过的日子$刘德华《一起走过的日子》1999年演唱会—在线播放"
    candicate = "$一起走过的日子$[type]CreativeWork[摘要]《一起走过的日子》是刘德华1991年发行的专辑，该专辑共收录有11首歌曲，同名歌曲是一首风靡全球的流行歌曲，至今仍为人们传唱，它被视为刘德华的成名曲之一。四大天王之一的歌手刘德华更是深受大家喜欢。[曲目数量]11[作曲]胡伟立[音乐风格]流行[唱片公司]宝艺星唱片[专辑歌手]刘德华[发行时间]1991.01.01[中文名称]一起走过的日子[作词]梁小美(梁美微)[发行地区]香港[专辑语言]粤语[义项描述]刘德华个人专辑[标签]单曲[标签]专辑[标签]音乐作品"
    querys = []
    all_candicates = candicate.split("|")
    for candicate in all_candicates:
        querys.append(str({"query_text": query,
                           "query_entity": "一起走过的日子",
                           "candi_abstract": candicate,
                           "candi_entity": "一起走过的日子",
                           "tag": 0}))

    res = client.query_parsing(querys)
    print("res: ", res, type(res))
