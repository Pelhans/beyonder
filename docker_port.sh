work_dir=$(
    cd $(dirname $0)
        pwd
)

CMD="python3 api.py"
CON_NAME=poi_total
NET=kg_demo

docker rm -f $CON_NAME || true

sudo docker run \
    --name ${CON_NAME} \
    -it \
    --net $NET \
    -p 9997:9997 \
    -v `pwd`:`pwd` \
    -w $work_dir \
    --restart always \
    zhangpeng/python3env \
    $CMD

docker logs -f ${CON_NAME}
