


FROM ubuntu:latest
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    build-essential \
    ccache \ 
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3 10 \
    && /usr/sbin/update-ccache-symlinks \
    && echo 'export PATH="/usr/lib/ccache:$PATH"' | tee -a ~/.bashrc \
    && apt-get update \
    && apt-get install -y python3-virtualenv \
    && apt-get install -y python3.12-venv 


# Copies your code file from your action repository to the filesystem path `/` of the container
COPY runTests.sh /runTests.sh

ENTRYPOINT ["/runTests.sh"]
