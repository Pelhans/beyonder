#!/usr/bin/env python3
# coding=utf-8
 
""" Packaging the model and providing a unified predictive function interface  """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from urllib.parse import quote
import os
import re
import _pickle as pk
base_path = "/".join(os.path.realpath(__file__).split("/")[:-1])

from es_import import ESImport
import requests
import string


def predict(hits):
    """ Return the final POI information with the return of BERT 
    :param hits:  Return of ES 
    :type hits: dict
    :return: dict with POI's lon, lat, class
    :rtype: dict
    """
    idx = 0
    return {"lon": hits[idx]["_source"]["lng"], 
            "lat": hits[idx]["_source"]["lat"],
            "name": hits[idx]["_source"]["name"],
            "address": hits[idx]["_source"]["address"],
           }

class Client:
    """Predict model Client
    :param model_name:  Model name set in tf_service 
    :type model_name: string
    :param server_ip: tf_service's ip
    :type server_ip: string
    :param server_port: tf_service's port
    :type server_port: string
    :param max_seq_length: max sequence length for BERT
    :type max_seq_length: int
    :param es_server: ES's ip:port
    """
    def __init__(self, es_server="47.104.240.64:9200",
                 name_dict = "kg_name_dict",
                index_name="kg_baidu",
                server_ner="poi_ner_tangshan",
                server_rerank="poi_rerank_tangshan"):

        self.es = ESImport(es_server)
        self.index_name = index_name
        self.server_ner = server_ner
        self.server_rerank = server_rerank
        self.name_dict = name_dict

#    @wrap_return
    def disambiguation(self, query):
        """ Functional interface provided to the outside
        :param query: query poi from get method
        :type query: string
        :return: a dict with lon, lat and poi type
        :rtype: dict
        """
        index_name = self.index_name
        # 网页请求有这些字符不行
        query = re.sub("[#&;]", "", query)
        ner_url = quote( "http://{}:6000/api/kg_ner?txt={}".format(self.server_ner, query), safe=string.printable )
        ner_entitys = requests.get(ner_url).text
        ner_entitys = eval(ner_entitys)["data"]
        print("ner_entitys: ", ner_entitys)
        res = []
        for entity in ner_entitys[0]:
            # 去 name_dict 里查询，返回对应的实体 id
            tmp_name_dict = self.es.search_name_dict(entity, self.name_dict)
            tmp_name_dict = tmp_name_dict["hits"]["hits"]
            hits = []
            if tmp_name_dict:
                title_id = tmp_name_dict[0]["_source"]["title_id"].split(",")
                for ti in title_id:
                    tmp_hits = self.es.search_titleid(ti, index_name)
                    tmp_hits = tmp_hits["hits"]["hits"]
                    if tmp_hits:
                        hits.append(tmp_hits[0])
#                print("命名实体字典返回的结果： ", hits)
            else:
                hits = self.poi_pair(entity, index_name)
            if len(hits) == 0:
                res.append(self._default_template(entity))
                continue
            candicate = ["[type]Thing[摘要]" + h["_source"]["abstract"] + \
                         h["_source"]["infobox"] for h in hits]
            candicate_entity = [h["_source"]["title"] for h in hits]
            len_candicate = len(candicate)
            candicate = "|".join(candicate)
            candicate_entity = "|".join(candicate_entity)
            candicate = re.sub("[#&;]", "", candicate)
            candicate_entity = re.sub("[#&;]", "", candicate_entity)
            rerank_res = requests.post(url="http://{}:6000/api/poi_rerank".format(self.server_rerank),
                                       json={"query": query,
                                             "candicate": candicate,
                                             "query_entity": entity,
                                             "candicate_entity": candicate_entity,
                                             "length": str(len_candicate)}).text
            all_tags = eval(rerank_res)["data"]["pred_ids"]
            all_probs = eval(rerank_res)["data"]["probabilities"]
            try:
                assert len(all_tags) == len(hits)
            except:
                print("查询实体: ", entity,
                      "\n候选实体", candicate_entity)
            tmp_res = []
            for idx, h in enumerate(hits):
                tmp_out = {"entity": h["_source"]["title"],
                           "mention": entity,
                           "desc": h["_source"]["abstract"],
#                           "attr": h["_source"]["infobox"],
                           "kg_id": h["_source"]["title_id"],
                           "confidence": all_probs[idx][1],
                           "tag": all_tags[idx],
                           "type": h["_source"]["subject"],
                          }
                tmp_res.append(tmp_out)
            tmp_res = self._resort(tmp_res)
            if tmp_res:
                desc = tmp_res[0]["desc"].split("。")
                if len(desc[0]) < 30 and len(desc) > 1:
                    tmp_res[0]["desc"] = "。".join(desc[:3])
                elif len(desc[0]) > 30:
                    tmp_res[0]["desc"] = desc[0]
                tmp_res[0]["desc"] = re.sub("\[\d*\]", "", tmp_res[0]["desc"]).strip()
                tmp_res[0]["desc"] = re.sub("\s*", "", tmp_res[0]["desc"])
                tmp_res[0].pop("tag")
                res.append(tmp_res[0])
            else:
                # EL 失败时，返回实体，并给出 NIL 标记
                res.append(self._default_template(entity))
        return res
    
    def _default_template(self, entity):
        tmp_res = {"entity": "NIL",
                   "mention": entity,
                   "desc": "",
                    "attr": "",
                   "kg_id": "",
                   "confidence": "",
                   "tag": "",
                   "type": "",
                  }
        return tmp_res

    
    def _resort(self, results):
        res = []
        res.extend( sorted([r for r in results if r["tag"] == 1], key=lambda x: x["confidence"]) )
        return res
    
    def poi_pair(self, query, index_name):
        """ Building candicate entity pairs from ES
        :param query: case poi
        :type query: string
        :param index_name: index name for ES,  POI data of different cities are
         stored in different index names 
        :type index_name: string
        :return: POI pair, <case_poi, candidate_poi> 
        :rtype: list
        """
        if not (isinstance(query, str) and isinstance(index_name, str)):
            query = str(query)
        res = self.es.search(query, index_name)
        hits = res["hits"]["hits"]
        return hits
    

if __name__ == "__main__":
    client = Client(es_server="47.104.240.64:9919",
                    name_dict="kg_name_dict",
                    index_name="kg_baidu",
                    server_ner="kg_ner_api",
                    server_rerank="kg_rerank_api")

    res = client.disambiguation("刘德华的老婆")

    print("res: ", res)
