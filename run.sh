#!/bin/bash 
# 设置NER 和 ReRank 模型与容器得名字
NET=kg_demo
SERVER_ES=kg_es
DOCKER_ES_IMPORT=es_docker
MODEL_NER=kg_ner_disambi
API_NER=kg_ner_api
MODEL_RERANK=kg_rerank_disambi
API_RERANK=kg_rerank_api
OUT_API=kg_el
INDEX_NAME=kg_baidu

# 在这设置主机 ip
HOST_IP='192.168.31.192'
# 总出口端口
OUT_PORT=9997

# ES 导入配置
POI_PATH=./data/baidu_430w/baidu.txt
# ES 端口
ES_PORT=9919
# 是否重新导入数据
RE_IMPORT='False'

# 创建组网
docker network create $NET;

## 开启 ES 服务并根据需求导入数据
#echo "开启 ES 服务并根据需求导入数据...."
#sudo ./import_es.sh $DOCKER_ES_IMPORT $SERVER_ES $ES_PORT $NET $RE_IMPORT $INDEX_NAME $POI_PATH;
#if [[ $RE_IMPORT == 'True' ]]; then
#    echo "ElasticSearch 导入需要大量时间，可以使用 docker logs -f $DOCKER_ES_IMPORT 命令查看进程";
#fi
#sleep 10s;

#echo "ES 服务开启完毕，准备开启 NER 模型"
## 在 ner 文件夹内开启服务;
#pushd ./ner/pb_model/;
## 在 model_server_config 配置模型名字
#sed -i "s/MODEL_NAME/$MODEL_NER/g" model_server_config;
#sudo ./docker_run.sh $MODEL_NER $NET;
## 检查一下配置
#cat model_server_config;
#cat docker_run.sh;
## 需要等一会才能改回来，否则会修改失败
#sleep 5s;
## 恢复修改，方便下次使用
#sed -i "s/$MODEL_NER/MODEL_NAME/g" model_server_config;
#cd ../;
#sudo ./docker_port.sh $MODEL_NER $API_NER $NET;
#popd;
#sleep 5s;
#
#echo "NER 模型开启完毕，准备开启 Rerank 模型"
#pushd ./entity_linking/pb_model/;
## 在 model_server_config 配置模型名字
#sed -i "s/MODEL_NAME/$MODEL_RERANK/g" model_server_config;
#sudo ./docker_run.sh $MODEL_RERANK $NET;
#cat model_server_config;
#cat docker_run.sh;
#sleep 5s;
## 恢复修改，方便下次使用
#sed -i "s/$MODEL_RERANK/MODEL_NAME/g" model_server_config;
#cd ../;
#sudo ./docker_port.sh $MODEL_RERANK $API_RERANK $NET;
#popd;
#sleep 5s;
#
## 开启总接口
#echo "开启总接口...."
##sudo ./docker_port.sh $SERVER_ES $API_NER $API_RERANK $OUT_PORT $OUT_API $NET $INDEX_NAME;
#
#echo "全部开启完毕"
