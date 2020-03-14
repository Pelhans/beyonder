#!/usr/bin/env python3
# coding=utf-8
"""Import POI data into Elasticsearch"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import subprocess
import codecs
from time import sleep
import json
from tqdm import tqdm
import _pickle as pk
import threading
import traceback

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--POI_path", type=str, default="./data/baidu.txt",
                    help="path to yiwu POI")
parser.add_argument("--es_service", type=str, default="47.104.240.64:9919",
                    help="ES service ip:port")
parser.add_argument("--index_name", type=str, default="kg_name_dict",
                    help="ES index_name")
parser.add_argument("--re_import",  default=False, action='store_true',
                    help=" Whether to re-import data ")
parser.add_argument("--analyzer", type=str, default="not_analyzed",
                    choices=["ik_max_word", "standard"], help="")

def read_dataset(infile):
    total_line = int(subprocess.getoutput("wc -l {}".format(infile)).split()[0])
    reader = codecs.open(infile, "r", "utf-8")
    for idx in range(total_line):
        yield reader.readline()

class ESImport(object):
    def __init__(self, es_server="192.168.31.192:9319"):
        self.es = Elasticsearch(es_server)

    def build_index(self, index_name, analyzer="not_analyzed",
                    index_var=None, unindex_var=None):
        """ Create kg index
        """
        if not index_var and not unindex_var:
            return {u'acknowledged': False}

        mappings = {
            "mappings": {
                "_doc": {
                    "properties":{}
                }
            }
        }
        properties = mappings["mappings"]["_doc"]["properties"]
        if index_var:
            for var in index_var:
                properties[var] = {"type": "keyword","index": True, "store": True,}
#                                   "analyzer": analyzer, "search_analyzer": analyzer }
        if unindex_var:
            for var in unindex_var:
                properties[var] = { "type": "text"}
                
        return self.es.indices.create(index=index_name, body=mappings)

    def delete_index(self, index_name):
        """Delete index
        :param index_name:
        :type index_name: string
        :return: result of creating an index, {u'acknowledged': True}
        :rtype: dict
        """
        return self.es.indices.delete(index=index_name)

    def insert_one(self, data, index_name):
        """Insert data into ES
        :param data:  Dict data to be inserted into ES species 
            data = {"title_id": "1",
                    "title": u"上海",
                    "abstract": u"上海是个城市",
                    "infobox": u"别称沪",
                    "subject": u"城市,直辖市",
                    "disambi": u"上海(城市名)",
                    "redirect": u"沪",}
        :type data: dict
        :param index_name:
        :type index_name: string
        :return: result of inserting data
        :rtype: dict
        """
        return self.es.index(index=index_name, body = data, doc_type="_doc")

    def insert_many(self, data, index_name):
        """Insert multi data into ES
        :param data: type list, element with insert_one's data type
        :type data: list
        :param index_name:
        :type index_name: string
        :return: result of inserting data
        :rtype: dict
        """
        res,_ = bulk(self.es, data)
        return res

    def search(self, query, index_name):
        """Search name dict in ES with title or someother name
        :param query:  query title name
        :type query: basestring
        :param index_name:
        :type index_name: string
        :return:  Candidate title in ES
        :rtype: dict
        """

        doc = {
          "query": {
            "bool": {
              "should": [
                {"match": {
                    "title": {"query": query,
                             "boost": 1/10,
                }}},
#                {"match": {
#                    "disambi": {"query": query,
#                                "boost": 1/3,
#                }}},
              ],
            "must": [
                {"match_phrase":{"title": {"query": query}}},
            ]
        }}}
        res = self.es.search(index=index_name, body=doc, size=10)

        return res
    
    def search_titleid(self, query, index_name):
        """Search name dict in ES with title_id
        """

        doc = {
          "query": {
            "bool": {
              "should": [
              ],
            "must": [
                {"match_phrase":{"title_id": {"query": query}}},
            ]
        }}}
        res = self.es.search(index=index_name, body=doc, size=10)
        return res


def clean_pkl():
    """ Clean up POI data to meet ESImport.insert_one() function requirements 
    """
    
    data = {"name": "",}
    with open("./data/name_dict.pkl", "rb") as f:
        name_dict = pk.load(f)
        for name, id in name_dict.items():
            if len(id) > 1000:
                print("这是个什么玩意？？...", name, "...")
                continue
            data["name"] = name
            data["title_id"] = ",".join(id).replace('"', "")
            data["_index"] = args.index_name
            data["_type"]  = "_doc"
            yield data

if __name__ == "__main__":
    args = parser.parse_args()
    print("es_service: ", args.es_service)
    data = clean_pkl()
    es_service = "localhost:9919"
    index_name = "kg_name_dict"
    analyzer = args.analyzer
    es = ESImport(es_service)
    if args.re_import:
        try:
            es.delete_index(index_name)
            print("删除成功")
        except:
            traceback.print_exc()
        try: 
            es.build_index(index_name=index_name,
                           analyzer=args.analyzer,
                           index_var=["name",], 
                           unindex_var=["title_id"])
        except:
            traceback.print_exc()
        # 一条一条的导入，很慢，每秒只能导入几十条
        index_err = []
        try:
            print("Importing data into ES....")
            es.insert_many(data, index_name)
#            for i, d in enumerate(tqdm(data)):
#                try:
#                    es.insert_one(index_name=index_name, data=d)
#                except Exception as e:
#                    index_err.append(d)
#                    print(e)
#                    break
        except:
            traceback.print_exc()
    else:
        print("不重新导入数据，已开启ES服务...")
