work_dir=$(
    cd $(dirname $0)
        pwd
)

NET=kg_demo
#CON_NAME=docker_es_import

#docker rm -f $CON_NAME || true
nvidia-smi
if [ $? -eq 0  ]; then
    docker run \
        -it \
        --runtime=nvidia \
        -e CUDA_VISIBLE_DEVICES=0 \
        --net $NET \
        -v `pwd`:`pwd` \
        -w $work_dir \
        pelhans_python3env
else
    echo "没有显卡，运行基本版"
    docker run \
        -it \
        --net $NET \
        -v `pwd`:`pwd` \
        -w $work_dir \
        pelhans_python3env
fi
