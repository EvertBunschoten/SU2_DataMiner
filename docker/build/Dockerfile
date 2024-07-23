FROM ubuntu:20.04
ENV LANG C.UTF-8
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    ccache \
    python3 \
 && rm -rf /var/lib/apt/lists/* \
 && update-alternatives --install /usr/bin/python python /usr/bin/python3 10 \
 && /usr/sbin/update-ccache-symlinks \
 && echo 'export PATH="/usr/lib/ccache:$PATH"' | tee -a ~/.bashrc

# pkg-config \
#     python3-pip \
#     git \
#     build-essential \
#     python3-numpy \
#     python3-scipy \
#     python3-mpi4py \
#     swig \
#     libopenmpi-dev \
#     openmpi-bin \
#     ccache \
    
# Copies your code file from your action repository to the filesystem path `/` of the container
#COPY compileSU2.sh /compileSU2.sh
COPY getPINNTraining.sh /getPINNTraining.sh

# Code file to execute when the docker container starts up (`entrypoint.sh`)
ENTRYPOINT ["/getPINNTraining.sh"]