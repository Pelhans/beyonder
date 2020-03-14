#!/bin/bash
MODEL_NAME=kg_ner_disambi
NET=kg_demo
CON_NAME=$MODEL_NAME
docker rm -f $CON_NAME || true

docker run -d \
           -v `pwd`:/model_server \
           --net $NET \
           --name ${CON_NAME} \
           --restart always \
           pelhans_tf_serving_1.12.0_cpu
