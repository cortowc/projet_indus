FROM ultralytics/ultralytics:latest-jetson

RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get update && \
    apt-get -y upgrade && \
    apt-get install -y --no-install-recommends \
      xcb \
      git \
      libglib2.0-0 \
      libgl1-mesa-glx && \
    apt-get -y clean && \
    rm -rf /var/lib/apt/lists/*

# Setting up working directory 
RUN mkdir /yolov8
WORKDIR /yolov8

RUN git clone https://github.com/roboflow/supervision.git /yolov8

# Remove opencv-python from requirements.txt as it conflicts with opencv-python installed in base image
#RUN grep -v '^opencv-python' setup.py > tmp.py && mv tmp.py setup.py
ENV DISPLAY=:0

COPY main_count.py .
ENV QT_X11_NO_MITSHM=1
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
CMD ["python3", "main_count.py"]

#To run the container:
# docker build -t yolov8_count Docker_application_comptage/
# chmod u+x runDocker.sh
# sh runDocker.sh
# python3 main_count.py
