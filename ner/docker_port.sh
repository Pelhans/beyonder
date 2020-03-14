work_dir=$(
    cd $(dirname $0)
        pwd
)
MODEL_NAME=kg_ner_disambi
API_NAME=kg_ner_api
NET=kg_demo
CMD="python3 api_ner.py"
CON_NAME=$API_NAME
docker rm -f $CON_NAME || true

docker run \
    --net $NET \
    --name ${CON_NAME} \
    -it \
    -p 9990:6000 \
    -v `pwd`:`pwd` \
    -w $work_dir \
    --restart always \
    zhangpeng/python3env \
    $CMD

docker logs -f $CON_NAME
