NUM=1

CURRENT=${NUM}
IMAGE_NAME=SRL_TO_TEMP
DOCKERFILE_NAME=Dockerfile

GIT_HASH=`git log --format="%h" -n 1`
IMAGE=$IMAGE_NAME_$USER-$GIT_HASH
IM_NAME=${IMAGE}_${NUM}

echo "Building $IMAGE"
docker build -f $DOCKERFILE_NAME -t $IMAGE .
beaker image create --name=${IM_NAME}_cuda111_${CURRENT} --description="SRL_TO_TEMP_V${CURRENT}_cuda111_${GIT_HASH}" $IMAGE
