work_dir=$(
    cd $(dirname $0)
        pwd
)

MODEL_NAME=kg_rerank_disambi
API_NAME=kg_rerank_api
NET=kg_demo
CMD="python3 api_rerank.py -model_name $MODEL_NAME"
CON_NAME=$API_NAME
docker rm -f $CON_NAME || true

docker run \
    --net $NET \
    --name ${CON_NAME} \
    -itd \
    -v `pwd`:`pwd` \
    -w $work_dir \
    --restart always \
    pelhans_python3env \
    $CMD

docker logs -f $CON_NAME
