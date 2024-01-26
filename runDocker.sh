xhost +local:docker
XSOCK=/tmp/.X11-unix
XAUTH=/tmp/.docker.xauth
docker run -m 8GB --runtime nvidia --gpus all -it --rm -e DISPLAY=$DISPLAY --device /dev/video2:/dev/video2 -v $XSOCK:$XSOCK -v $XAUTH:$XAUTH -e XAUTHORITY=$XAUTH -v ${PWD}:/src yolov8 ${1:-bash}
xhost -local:docker

