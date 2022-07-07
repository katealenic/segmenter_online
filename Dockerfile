FROM nvidia/cuda:10.2-base
FROM nvidia/cuda:10.2-devel-ubuntu18.04
# Updating Ubuntu packages
RUN apt-get update && yes|apt-get upgrade
# Installation
RUN apt-get update && apt-get install --no-install-recommends --no-install-suggests -y curl
RUN apt-get -y update
RUN apt-get -y install python3
RUN apt-get -y install python3-pip

RUN apt-get update
RUN apt install -y libgl1-mesa-glx
RUN pip3 install --upgrade pip
RUN pip3 install opencv-python
RUN apt-get install -y wget zip unzip

RUN apt-get -y install git
RUN pip3 install torch torchvision torchaudio
RUN pip3 install click einops python-hostlist tqdm requests imageio
RUN pip3 install timm==0.4.12 mmcv==1.3.8 mmsegmentation==0.14.1
RUN pip3 install -U PyYAML
RUN echo "export LC_ALL=C.UTF-8" >> ~/.bashrc
RUN echo "export LANG=C.UTF-8" >> ~/.bashrc
RUN mkdir segmenter_online
COPY segm segmenter_online/segm
RUN cd segmenter_online && mkdir weights && mkdir results
WORKDIR segmenter_online