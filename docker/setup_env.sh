# install requirements
apt-get update -y \
  && sudo apt-get install -y --no-install-recommends \
    sudo \
    bzip2 \
    git \
    python3.8\
    python3-dev \
    python3-pip \
    python3-wheel \
    python3-setuptools \
    unzip \
	vim \
    wget \
    build-essential \
  && sudo apt-get clean \
  && sudo rm -rf /var/lib/apt/lists/*

apt-get install ffmpeg libsm6 libxext6  -y

# Create pipenv environment
alias python=python3.8
python -m pip install -r ./docker/requirements.txt