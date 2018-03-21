FROM python:3.6-slim

RUN apt-get update \
  && apt-get install -y \
    libcurl4-openssl-dev \
    net-tools \
    curl \
    wget \
    mlocate \
    gcc \
    make \
    autoconf \
    build-essential \
    software-properties-common \
    git \
    vim \
    python3 \
    python3-dev \
    python3-pip \
    python3-tk \
    python-setuptools \
    python-virtualenv \
    python-pip \
    openssl \
    libssl-dev \
    cmake \
    autoconf \
    libffi6 \
    libffi-dev \
    telnet \
    netcat \
    unzip

RUN echo "creating directories" \
  && mkdir -p -m 777 /opt \
  && mkdir -p -m 777 /data \
  && chmod 777 /opt \
  && mkdir -p -m 777 /opt/venvs \
  && mkdir -p -m 777 /opt/shared \
  && mkdir -p -m 777 /opt/data

RUN echo "preparing virtualenv" \
  && pip install --upgrade virtualenvwrapper pip

RUN echo "adding user" \
  && useradd -ms /bin/bash runner

RUN echo "setting user password" \
  && usermod --password admin runner \
  && echo "runner:admin" | chpasswd \
  && echo "root:admin" | chpasswd

# Change to the internal user
USER runner

COPY vimrc /home/runner/.vimrc

RUN echo "creating virtualenv" \
  && virtualenv -p python3 /opt/venvs/venvdrfpipeline

RUN echo "setting up virtualenv" \
  && ln -s /opt/venvs ~/.venvs \
  && ls -l ~/.venvs \
  && . ~/.venvs/venvdrfpipeline/bin/activate \
  && pip install --upgrade setuptools pip

RUN echo "" >> /home/runner/.bashrc \
  && echo "if [[ -e /home/runner/.venvs/venvdrfpipeline/bin/activate ]]; then" >> /home/runner/.bashrc \
  && echo "    source /home/runner/.venvs/venvdrfpipeline/bin/activate" >> /home/runner/.bashrc \
  && echo "fi" >> /home/runner/.bashrc \
  && echo "" >> /home/runner/.bashrc \
  && echo "alias api='cd /opt/antinex-api'" >> /home/runner/.bashrc \
  && echo "alias core='cd /opt/antinex-core'" >> /home/runner/.bashrc \
  && echo "alias ds='cd /opt/antinex-datasets'" >> /home/runner/.bashrc \
  && echo "alias ut='cd /opt/antinex-utils'" >> /home/runner/.bashrc \
  && echo "alias vi='/usr/bin/vim'" >> /home/runner/.bashrc \
  && echo "" >> /home/runner/.bashrc \
  && mkdir ~/.pip \
  && echo "[list]" >> ~/.pip/pip.conf \
  && echo "format=columns" >> ~/.pip/pip.conf

RUN echo "cloning repos" \
  && ls -l /opt \
  && ls -l / \
  && git clone https://github.com/jay-johnson/antinex-utils.git /opt/antinex-utils \
  && git clone https://github.com/jay-johnson/antinex-core.git /opt/antinex-core \
  && git clone https://github.com/jay-johnson/train-ai-with-django-swagger-jwt /opt/antinex-api \
  && git clone https://github.com/jay-johnson/antinex-datasets.git /opt/antinex-datasets \
  && chmod 775 /opt/antinex-api /opt/antinex-utils /opt/antinex-core /opt/antinex-datasets

RUN cd /opt/antinex-api \
  && ls -l /opt/antinex-api \
  && ls -l /opt/antinex-core \
  && ls -l /opt/antinex-utils \
  && ls -l /opt/antinex-datasets

RUN echo "installing utils" \
  && . ~/.venvs/venvdrfpipeline/bin/activate \
  && cd /opt/antinex-utils \
  && pip install --upgrade -e .

RUN echo "installing core" \
  && . ~/.venvs/venvdrfpipeline/bin/activate \
  && cd /opt/antinex-core \
  && pip install --upgrade -e .

RUN echo "installing api" \
  && . ~/.venvs/venvdrfpipeline/bin/activate \
  && cd /opt/antinex-api \
  && pip install --upgrade -r /opt/antinex-api/requirements.txt

RUN echo "installing jupyter pips" \
  && . ~/.venvs/venvdrfpipeline/bin/activate \
  && pip list --format=columns \
  && pip install --upgrade \
    requests \
    seaborn \
    RISE \
    vega3 \
    jupyter

RUN cp -r ~/.jupyter ~/.bak_jupyter || true
RUN rm -rf ~/.jupyter || true
RUN cp -r ~/notebooks ~/.bak_notebooks || true
RUN rm -rf ~/notebooks || true

RUN ln -s /opt/antinex-core/docker/jupyter ~/.jupyter \
  && ln -s /opt/antinex-core/docker/notebooks ~/notebooks

RUN echo "Installing JupyterLab" \
  && git clone https://github.com/jupyterlab/jupyterlab.git /opt/jupyterlab \
  && cd /opt/jupyterlab \
  && . ~/.venvs/venvdrfpipeline/bin/activate \
  && pip install -e .

RUN echo "Installing Vega" \
  && . ~/.venvs/venvdrfpipeline/bin/activate \
  && ~/.venvs/venvdrfpipeline/bin/jupyter-nbextension install vega3 --py --user

RUN echo "Enabling Vega" \
  && . ~/.venvs/venvdrfpipeline/bin/activate \
  && ~/.venvs/venvdrfpipeline/bin/jupyter-nbextension enable vega3 --py --user

RUN echo "Installing Rise" \
  && . ~/.venvs/venvdrfpipeline/bin/activate \
  && ~/.venvs/venvdrfpipeline/bin/jupyter-nbextension install rise --py --user

RUN echo "Enabling Rise" \
  && . ~/.venvs/venvdrfpipeline/bin/activate \
  && ~/.venvs/venvdrfpipeline/bin/jupyter-nbextension enable rise --py --user

RUN mkdir -p -m 777 /opt/antinex-core

RUN ls /opt/antinex-core/ \
  && chmod 777 /opt/antinex-core/docker/jupyter/start-container.sh \
  && chmod 777 /opt/antinex-core/run-antinex-core.sh

ENV JUPYTER_START_SCRIPT /opt/antinex-core/docker/jupyter/start-container.sh
ENV NO_DB_WORKER_START_SCRIPT /opt/antinex-core/run-antinex-core.sh
ENV LOG_CFG /opt/antinex-core/antinex_core/log/colors-logging.json
ENV ENVIRONMENT Development
ENV DJANGO_CONFIGURATION Development
ENV DJANGO_SECRET_KEY supersecret
ENV DJANGO_DEBUG yes
ENV DJANGO_TEMPLATE_DEBUG yes
ENV CELERY_ENABLED 0
ENV CACHEOPS_ENABLED 0
ENV ANTINEX_WORKER_ENABLED 1
ENV ANTINEX_WORKER_ONLY 0
ENV ANTINEX_DELIVERY_MODE persistent
ENV ANTINEX_AUTH_URL redis://0.0.0.0:6379/6
ENV ANTINEX_EXCHANGE_NAME webapp.predict.requests
ENV ANTINEX_EXCHANGE_TYPE topic
ENV ANTINEX_QUEUE_NAME webapp.predict.requests
ENV ANTINEX_WORKER_SSL_ENABLED 0
ENV USE_ENV drf-dev
ENV DJANGO_WORKER_START_SCRIPT /opt/antinex-api/run-worker.sh
ENV APP_START_SCRIPT /opt/antinex-api/run-django.sh
ENV MIGRATION_SCRIPT /opt/antinex-api/run-migrations.sh

WORKDIR /opt/antinex-core

ENTRYPOINT /opt/antinex-core/run-antinex-core.sh
