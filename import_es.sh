#!/bin/bash 
ES_CON_NAME=kg_es
ES_PORT=9919
NET=kg_demo

CUR_PATH=`pwd`;
DATA_PATH=$CUR_PATH/thirdpart/elasticsearch-7.2.0/data;
sudo mkdir -p $DATA_PATH/nodes;
sudo chmod -R 777 $DATA_PATH/nodes;

docker rm -f $ES_CON_NAME || true
sudo docker run \
    -d \
    -p $ES_PORT:9200 \
    --name ${ES_CON_NAME} \
    --net $NET \
    -e CUR_PATH=`pwd` \
    --restart always \
    -v `pwd`:`pwd` -v $DATA_PATH/nodes:/usr/share/elasticsearch/data/nodes \
    zhangpeng_es-ik:init &

