FROM ubuntu:20.04
ENV LANG C.UTF-8
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    build-essential \
    ccache \ 
    && python3 -m pip install tensorflow \
    pickledb \
    CoolProp \
    cantera \
    tqdm \
    python-csv \
    matplotlib \
    scikit-learn \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3 10 \
    && /usr/sbin/update-ccache-symlinks \
    && echo 'export PATH="/usr/lib/ccache:$PATH"' | tee -a ~/.bashrc 

# Copies your code file from your action repository to the filesystem path `/` of the container
COPY getSU2DataMiner.sh /getSU2DataMiner.sh

ENTRYPOINT ["/getSU2DataMiner.sh"]
