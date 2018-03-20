FROM python:3.6-slim

RUN apt-get update \
  && apt-get install -y \
    python3-dev \
    libcurl4-openssl-dev \
    net-tools \
    curl \
    wget \
    mlocate \
    python3-tk \
    gcc \
    make \
    autoconf \
    build-essential \
    software-properties-common \
    git \
    vim \
    python3 \
    python3-dev \
    python-setuptools \
    python-virtualenv \
    python-pip \
    openssl \
    libssl-dev \
    cmake \
    autoconf \
    libffi6 \
    libffi-dev \
    unzip

RUN echo "installing pips" \
  && pip3.6 list \
  && /usr/local/bin/pip3.6 install --upgrade \
    requests \
    seaborn \
    pip \
    RISE \
    vega \
    jupyter \
    jupyterlab

RUN echo "installing pips" \
  && /usr/local/bin/pip3.6 list --format=columns \
  && which python

RUN echo "Installing JupyterLab" \
  && git clone https://github.com/jupyterlab/jupyterlab.git /opt/jupyterlab \
  && cd /opt/jupyterlab \
  && pip install -e .

RUN echo "Installing Vega" \
  && python /usr/local/bin/jupyter-nbextension install vega --py --sys-prefix

RUN echo "Enabling Vega" \
  && python /usr/local/bin/jupyter-nbextension enable vega --py --sys-prefix

RUN echo "Installing Rise" \
  && python /usr/local/bin/jupyter-nbextension install rise --py --sys-prefix

RUN echo "Enabling Rise" \
  && python /usr/local/bin/jupyter-nbextension enable rise --py --sys-prefix

RUN mkdir -p -m 777 /opt/antinex-core

COPY ai-core-latest.tgz /opt/antinex-core

RUN cd /opt/antinex-core \
  && tar xvf ai-core-latest.tgz \
  && rm -rf /opt/antinex-core/docker/data/* \
  && ls -l /opt/antinex-core \
  && ls -l /opt/antinex-core/docker

RUN cp -r /root/.jupyter /root/.bak_jupyter || true
RUN rm -rf /root/.jupyter || true
RUN cp -r /root/notebooks /root/.bak_notebooks || true
RUN rm -rf /root/notebooks || true

RUN ln -s /opt/antinex-core/docker/jupyter /root/.jupyter \
  && ln -s /opt/antinex-core/docker/notebooks /root/notebooks \
  && ls -l /opt/antinex-core \
  && ls -l /opt/antinex-core/docker \
  && ls -la /root/ \
  && cd /opt/antinex-core \
  && /usr/local/bin/pip3.6 install --upgrade -e . \
  && /usr/local/bin/pip3.6 list --format=columns

RUN ls /opt/antinex-core/ \
  && chmod 777 /opt/antinex-core/docker/jupyter/start-container.sh \
  && chmod 777 /opt/antinex-core/run-antinex-core.sh

ENV JUPYTER_START_SCRIPT /opt/antinex-core/docker/jupyter/start-container.sh
ENV WORKER_START_SCRIPT /opt/antinex-core/run-antinex-core.sh
ENV LOG_CFG /opt/antinex-core/antinex_core/log/colors-logging.json

WORKDIR /opt/antinex-core

ENTRYPOINT /opt/antinex-core/run-antinex-core.sh
