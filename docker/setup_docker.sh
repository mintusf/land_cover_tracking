#!/bin/bash
# You need to change followings:
DOCKERFILE_PATH="./docker"
IMAGE_NAME="fmintus/land_cover_app"
CONTAINER_NAME="land_cover_app"
# GPU='"device=0,1,2,3"'
MEMORY="12G"

echo "Creating docker image"
docker build --network host -t $IMAGE_NAME $DOCKERFILE_PATH --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g)
echo "Finished !"

echo "Creating docker container"
docker run --net host \
           -v $REPOSITORY_PATH:/workspace \
           -v /raid/RTC_02/recieved_data:/received_data \
           -v /mnt/USB_HDD:/received_data2 \
           --shm-size=$MEMORY \
           --name $CONTAINER_NAME \
           -itd $IMAGE_NAME
echo "Finished !"
