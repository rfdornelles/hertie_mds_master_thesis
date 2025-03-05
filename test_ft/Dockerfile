FROM nvidia/cuda:12.2.0-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

# Instala dependências, adiciona o repositório deadsnakes e instala o Python 3.10 e dependências de desenvolvimento
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y \
       python3.10 \
       python3.10-dev \
       python3.10-distutils \
       build-essential \
       gcc \
       curl \
       && rm -rf /var/lib/apt/lists/*

# Configura o Python 3.10 como padrão para o comando python3
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Instala o pip para o Python 3.10 usando get-pip.py
RUN curl -sS https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3.10 get-pip.py && \
    rm get-pip.py

WORKDIR /app

COPY requirements.txt .
RUN python3.10 -m pip install --upgrade pip && python3.10 -m pip install -r requirements.txt

COPY . .

# COPY model_cache /app/model_cache
# ENV HF_HOME=/app/model_cache

# Comando padrão para executar seu script com Accelerate (ajuste o número de processos se necessário)
#CMD ["accelerate", "launch", "--num_processes=4", "test_ft.py"]
