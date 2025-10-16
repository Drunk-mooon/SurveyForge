FROM docker.v2.aispeech.com/sjtu/sjtu_chenlu-yangziyue-cuda_12.2.2-ubuntu_22.04-torch_2.6-cu_124_general:v1.0

WORKDIR /app

# 1. 复制整个Repo
COPY ./requirement.txt .

# 2. 安装依赖并随后删除Repo
RUN  pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirement.txt && \
    pip install --no-cache-dir -r h5py && \
    pip install --no-cache-dir -r jsonlines && \
    rm -rf /app/requirement.txt 

COPY . .
