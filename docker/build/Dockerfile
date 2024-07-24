FROM ubuntu:20.04
ENV LANG C.UTF-8
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    build-essential \
    ccache \ 
    && python3 -m pip install tensorflow==2.13.1 \
    pickledb \
    CoolProp==6.6.0 \
    tqdm==4.65.0 \
    python-csv \
    matplotlib==3.7.5 \
    scikit-learn==1.2.2
# RUN apt-get update && apt-get install -y \
#     python3 \
#     pkg-config \
#     python3-pip \
#     git \
#     build-essential \
#     python3-mpi4py \
#     swig \
#     libopenmpi-dev \
#     openmpi-bin \
#     ccache \
#  && rm -rf /var/lib/apt/lists/* \
#  && update-alternatives --install /usr/bin/python python /usr/bin/python3 10 \
#  && /usr/sbin/update-ccache-symlinks \
#  && echo 'export PATH="/usr/lib/ccache:$PATH"' | tee -a ~/.bashrc \

# Copies your code file from your action repository to the filesystem path `/` of the container
COPY ./getPINNTraining.sh /getPINNTraining.sh

ENTRYPOINT ["/getPINNTraining.sh"]
