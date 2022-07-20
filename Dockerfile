FROM nvidia/cuda:11.1-runtime-ubuntu20.04

ENV TZ=Europe/Amsterdam
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# RUN : \
#     && apt-get update \
#     && apt-get install -y apt-transport-https \
#     && echo 'deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64 /' > /etc/apt/sources.list.d/cuda.list \
#     && :

# Install python3.8
RUN : \
    && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub \
    && apt-get update \
    && apt-get -y install libgl1-mesa-glx \
    && apt-get install -y --no-install-recommends software-properties-common \
    && add-apt-repository -y ppa:deadsnakes \
    && apt-get install -y --no-install-recommends python3.8-venv \
    && apt-get install libpython3.8-dev -y \
    && apt-get clean \
    && :

# Add env to PATH
RUN python3.8 -m venv /venv
ENV PATH=/venv/bin:$PATH

COPY ./libs/ASAP-2.0-py38-Ubuntu2004.deb /home/ASAP-2.0-py38-Ubuntu2004.deb

# Install ASAP
RUN : \
    && apt-get update \
    && apt-get -y install curl \
    && dpkg --install /home/ASAP-2.0-py38-Ubuntu2004.deb || true \
    && apt-get -f install --fix-missing --fix-broken --assume-yes \
    && ldconfig -v \
    && apt-get clean \
    && echo "/opt/ASAP/bin" > /venv/lib/python3.8/site-packages/asap.pth \
    && :

# # Install algorithm
COPY ./tigeralgorithm /home/user/pathology-tiger-algorithm/
COPY ./libs/mmsegmentation /home/user/mmsegmentation
COPY ./libs/mmdetection /home/user/mmdetection
COPY ./libs/yolov5 /home/user/yolov5
COPY ./libs/WEIGHT /home/user/WEIGHT

RUN : \
    && apt-get install -y git \
    && apt-get install -y g++ \
    && apt-get install -y gcc \
    && :

RUN : \
    && pip install wheel==0.37.0 \
    && pip install ensemble-boxes \
    && pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html \
    && pip install mmcv-full==1.4.7 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.10.0/index.html \
    && cd /home/user/mmsegmentation \
    && pip install -e . \
    && cd /home/user/mmdetection \
    && pip install -e . \
    && cd /home/user/yolov5 \
    && pip install -r requirements.txt \
    && pip install /home/user/pathology-tiger-algorithm \
    && :
    #&& rm -r /home/user/pathology-tiger-algorithm \
    

# Make user
RUN groupadd -r user && useradd -r -g user user
RUN chown user /home/user/
RUN mkdir /output/
RUN chown user /output/
USER user
WORKDIR /home/user

# Cmd and entrypoint
CMD ["-mtigeralgorithmexample"]
ENTRYPOINT ["python"]

# Compute requirements
LABEL processor.cpus="1"
LABEL processor.cpu.capabilities="null"
LABEL processor.memory="15G"
LABEL processor.gpu_count="1"
LABEL processor.gpu.compute_capability="null"
LABEL processor.gpu.memory="11G"
