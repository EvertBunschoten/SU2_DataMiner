
# Container image that runs your code
FROM alpine:3.10

# Copies your code file from your action repository to the filesystem path `/` of the container
COPY docker/build/entrypoint.sh /entrypoint.sh

# Code file to execute when the docker container starts up (`entrypoint.sh`)
ENTRYPOINT ["/entrypoint.sh"]

# FROM ubuntu:latest
# ENV LANG C.UTF-8
# RUN apt-get update && apt-get install -y \
#     python3 \
#     python3-pip \
#     git \
#     build-essential \
#     ccache \ 
#     && rm -rf /var/lib/apt/lists/* \
#     && update-alternatives --install /usr/bin/python python /usr/bin/python3 10 \
#     && /usr/sbin/update-ccache-symlinks \
#     && echo 'export PATH="/usr/lib/ccache:$PATH"' | tee -a ~/.bashrc \
#     && apt-get update \
#     && apt-get install -y python3-virtualenv \
#     && apt-get install -y python3.12-venv \
#     && python3 -m venv /home/unbuntu/pyenv \
#     && virtualenv -p /usr/bin/python3 /home/ubuntu/pyenv



# # Copies your code file from your action repository to the filesystem path `/` of the container
# COPY docker/build/getSU2DataMiner.sh /getSU2DataMiner.sh

# ENTRYPOINT ["/getSU2DataMiner.sh"]
