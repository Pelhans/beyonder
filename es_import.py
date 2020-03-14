#!/usr/bin/env python3
# coding=utf-8
"""Import POI data into Elasticsearch"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import subprocess
import codecs
import json
import re
from tqdm import tqdm
import threading
import traceback

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--POI_path", type=str, default="./data/baidu.txt",
                    help="path to yiwu POI")
#parser.add_argument("--es_service", type=str, default="47.104.240.64:9919",
#                    help="ES service ip:port")
parser.add_argument("--es_service", type=str, default="kg_es:9200",
                    help="ES service ip:port")
parser.add_argument("--index_name", type=str, default="kg_baidu",
                    help="ES index_name")
parser.add_argument("--re_import",  default=False, action='store_true',
                    help=" Whether to re-import data ")
parser.add_argument("--analyzer", type=str, default="ik_max_word",
                    choices=["ik_max_word", "standard"], help="")

def read_dataset(infile):
    total_line = int(subprocess.getoutput("wc -l {}".format(infile)).split()[0])
    reader = codecs.open(infile, "r", "utf-8")
    for idx in range(total_line):
        yield reader.readline()

class ESImport(object):
    def __init__(self, es_server="192.168.31.192:9319"):
        self.es = Elasticsearch(es_server)

    def build_index(self, index_name, analyzer="ik_max_word",
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
                properties[var] = {"type": "text","index": True, "store": True,
                                   "analyzer": analyzer,  "store": True,
                                   "search_analyzer": analyzer }
        if unindex_var:
            for var in unindex_var:
                properties[var] = {"type": "keyword","index": True, "store": True}
                
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
            "must": [
                {"match_phrase":{"title": {"query": query}}},
            ]
        }}}
        res = self.es.search(index=index_name, body=doc, size=10)

        return res
    
    def search_name_dict(self, query, index_name):
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
                    "name": {"query": query,
                }}},
              ],
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


def clean_data(data_path):
    """ Clean up POI data to meet ESImport.insert_one() function requirements 
    :param data_path: path to POI data
    :type data_path: basestring
    :return: generator with  insert_one's element type
    :rtype: generator,element is dict
    """
    
    data = {"title_id": "",
            "title": u"",
            "abstract": u"",
            "infobox": u"",
            "subject": u"",
            "disambi": u"",
            "redirect": u"",}
    error_count = 0
    e = codecs.open("./err_line.txt", "w", "utf-8")
    with codecs.open(data_path, "r", "utf-8") as f:
        _ = subprocess.getoutput("wc -l {}".format(data_path))
        total_lines = int(_.split()[0])
        for line_num in range(total_lines):
            line = f.readline()
            all_poi = line.split("\t")
            if len(all_poi) != 8:
                error_count += 1
                e.write("\t".join([str(line_num), line]))
                continue

            data["title"], data["title_id"] = all_poi[0].strip("\""), all_poi[1].strip("\"")
            data["abstract"], data["redirect"] = all_poi[2].strip("\""), all_poi[6].strip("\"")
            data["subject"], data["disambi"] = all_poi[4].strip("\""), all_poi[5].strip("\"")
            data["subject"] = data["subject"].strip(",")
            out_info = ""
            try:
#                info = all_poi[3]
#                info = all_poi[3].strip().strip('"')
#                info = re.sub("\s*", "", info)
#                info = re.sub("\[\d*\]", "", info)
                infobox = eval(json.loads(all_poi[3]))
                for attr,value in infobox.items():
                    attr = re.sub("\s*", "", attr)
                    attr = re.sub("\[\d*\]", "", attr)
                    value = re.sub("\s*", "", value)
                    value = re.sub("\[\d*\]", "", value)
                    out_info += "[" + attr + "]" + value
            except:
                error_count += 1
                e.write("\t".join([str(line_num), line]))
                continue
#            res = {}
#            for k,v in data.items():
##                k = k.encode('UTF-8','ignore').decode("utf-8")
##                res[k] = v.encode('UTF-8','ignore').decode("utf-8")
#                k = k.encode("unicode_escape").decode("utf-8")
#                res[k] = v.encode("unicode_escape").decode("utf-8")
#            data = res
            data["infobox"] = out_info
#            data["_index"] = args.index_name
#            data["_type"]  = "_doc"
            
            if line_num < 5:
                print("data: ", data)
            yield data
        print(" Total number of rows that cannot be split: {}".format(error_count))

if __name__ == "__main__":
    args = parser.parse_args()
    print("es_service: ", args.es_service)
    data = clean_data(args.POI_path)
    es_service = args.es_service
    index_name = args.index_name
    analyzer = args.analyzer
    es = ESImport(es_service)
#    res = es.search("上海", "kg_name_dict")["hits"]["hits"]
#    print("ES: ",res)
    if args.re_import:
        try:
            es.delete_index(index_name)
            print("删除成功")
        except:
            traceback.print_exc()
        try: 
            es.build_index(index_name=index_name,
                           analyzer=args.analyzer,
                           index_var=["title", "abstract", 
                                      "infobox", "subject", "disambi",
                                      "redirect"], 
                           unindex_var=["title_id"])
        except:
            traceback.print_exc()
        # 一条一条的导入，很慢，每秒只能导入几十条
        index_err = []
        try:
            print("Importing data into ES....")
            for i, d in enumerate(tqdm(data)):
                try:
                    es.insert_one(index_name=args.index_name, data=d)
                except Exception as e:
                    index_err.append(d)
                    print(e)
#                    break
        except:
            traceback.print_exc()
        # 失败的再来一次
        print("找到的错误: ", len(index_err))
        final_err = 0
        log_import = codecs.open("./log_import.txt", "w", "utf-8")
        for i, d in enumerate(tqdm(index_err)):
            try:
                es.insert_one(index_name=args.index_name, data=d)
            except:
                log_import.write(str(d) + "\d")
                final_err += 1
        print("最终有 {} 导入失败的".format(final_err))
#        es.insert_many(index_name=index_name, data=data)

    else:
        print("不重新导入数据，已开启ES服务...")
