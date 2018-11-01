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
    pandoc \
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
  && mkdir -p -m 777 /opt/antinex \
  && mkdir -p -m 777 /data \
  && mkdir -p -m 777 /var/log/antinex/core \
  && mkdir -p -m 777 /var/log/antinex/api \
  && mkdir -p -m 777 /var/log/antinex/jupyter \
  && mkdir -p -m 777 /var/log/antinex/pipeline \
  && mkdir -p -m 777 /var/log/antinex/client \
  && mkdir -p -m 777 /opt/shared \
  && mkdir -p -m 777 /opt/data \
  && chmod 777 /opt

RUN echo "creating log files" \
  && touch /var/log/antinex/api/api.log && chmod 777 /var/log/antinex/api/api.log \
  && touch /var/log/antinex/api/worker.log && chmod 777 /var/log/antinex/api/worker.log \
  && touch /var/log/antinex/core/ai-core.log && chmod 777 /var/log/antinex/core/ai-core.log \
  && touch /var/log/antinex/client/client.log && chmod 777 /var/log/antinex/client/client.log \
  && touch /var/log/antinex/jupyter/jupyter.log && chmod 777 /var/log/antinex/jupyter/jupyter.log \
  && touch /var/log/antinex/pipeline/pipeline.log && chmod 777 /var/log/antinex/pipeline/pipeline.log \
  && touch /var/log/antinex/client/client.log && chmod 777 /var/log/antinex/client/client.log

RUN echo "preparing virtualenv" \
  && pip install --upgrade virtualenvwrapper pip

RUN echo "creating virtualenv" \
  && virtualenv -p python3 /opt/venv \
  && chmod 777 /opt/venv

RUN echo "setting up virtualenv" \
  && . /opt/venv/bin/activate \
  && pip install --upgrade setuptools pip

RUN echo "" >> /etc/bashrc \
  && echo "if [[ -e /opt/venv/bin/activate ]]; then" >> /etc/bashrc \
  && echo "    source /opt/venv/bin/activate" >> /etc/bashrc \
  && echo "fi" >> /etc/bashrc \
  && echo "" >> /etc/bashrc \
  && echo "alias api='cd /opt/antinex/api'" >> /etc/bashrc \
  && echo "alias core='cd /opt/antinex/core'" >> /etc/bashrc \
  && echo "alias client='cd /opt/antinex/client'" >> /etc/bashrc \
  && echo "alias pipe='cd /opt/antinex/pipeline'" >> /etc/bashrc \
  && echo "alias ut='cd /opt/antinex/utils'" >> /etc/bashrc \
  && echo "alias ad='cd /opt/antinex/antinex-datasets'" >> /etc/bashrc \
  && echo "alias ds='cd /opt/antinex/datasets'" >> /etc/bashrc \
  && echo "alias sp='cd /opt/spylunking'" >> /etc/bashrc \
  && echo "alias vi='/usr/bin/vim'" >> /etc/bashrc

RUN echo "setting up /etc/pip.conf" \
  && echo "" >> /etc/pip.conf \
  && echo "[list]" >> /etc/pip.conf \
  && echo "format=columns" >> /etc/pip.conf

RUN echo "cloning repos" \
  && ls -l /opt \
  && ls -l / \
  && git clone https://github.com/jay-johnson/train-ai-with-django-swagger-jwt /opt/antinex/api \
  && git clone https://github.com/jay-johnson/antinex-core.git /opt/antinex/core \
  && git clone https://github.com/jay-johnson/antinex-client.git /opt/antinex/client \
  && git clone https://github.com/jay-johnson/network-pipeline.git /opt/antinex/pipeline \
  && git clone https://github.com/jay-johnson/antinex-utils.git /opt/antinex/utils \
  && git clone https://github.com/jay-johnson/antinex-datasets.git /opt/antinex/antinex-datasets \
  && git clone https://github.com/jay-johnson/network-pipeline-datasets.git /opt/antinex/datasets \
  && git clone https://github.com/jay-johnson/deploy-to-kubernetes.git /opt/deploy-to-kubernetes \
  && git clone https://github.com/jay-johnson/spylunking.git /opt/spylunking \
  && chmod 775 \
    /opt/antinex/api \
    /opt/antinex/core \
    /opt/antinex/client \
    /opt/antinex/pipeline \
    /opt/antinex/utils \
    /opt/antinex/antinex-datasets \
    /opt/antinex/datasets \
    /opt/deploy-to-kubernetes \
    /opt/spylunking

RUN echo "checking repos in container" \
  && ls -l /opt/antinex/api \
  && ls -l /opt/antinex/core \
  && ls -l /opt/antinex/client \
  && ls -l /opt/antinex/pipeline \
  && ls -l /opt/antinex/utils \
  && ls -l /opt/antinex/antinex-datasets \
  && ls -l /opt/antinex/datasets \
  && ls -l /opt/deploy-to-kubernetes \
  && ls -l /opt/spylunking

RUN echo "installing python logger with splunk support" \
  && . /opt/venv/bin/activate \
  && cd /opt/spylunking \
  && pip install --upgrade -e . \
  && cd docs \
  && make html

RUN echo "installing utils" \
  && . /opt/venv/bin/activate \
  && cd /opt/antinex/utils \
  && pip install --upgrade -e . \
  && cd docs \
  && make html

RUN echo "installing pipeline" \
  && . /opt/venv/bin/activate \
  && cd /opt/antinex/pipeline \
  && pip install --upgrade -e . \
  && cd docs \
  && make html

RUN echo "installing core" \
  && . /opt/venv/bin/activate \
  && cd /opt/antinex/core \
  && pip install --upgrade -e . \
  && cd docs \
  && make html

RUN echo "installing api" \
  && . /opt/venv/bin/activate \
  && cd /opt/antinex/api \
  && pip install --upgrade -r /opt/antinex/api/requirements.txt

RUN echo "building docs" \
  && . /opt/venv/bin/activate \
  && . /opt/antinex/api/envs/drf-dev.env \
  && cd /opt/antinex/api/webapp \
  && ls -l \
  && ./build-docs.sh \
  && echo "collecting statics" \
  && ./collect-statics.sh

RUN echo "installing client" \
  && . /opt/venv/bin/activate \
  && cd /opt/antinex/client \
  && pip install --upgrade -e . \
  && cd docs \
  && make html

RUN echo "installing jupyter pips" \
  && . /opt/venv/bin/activate \
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

RUN echo "Installing JupyterLab" \
  && git clone https://github.com/jupyterlab/jupyterlab.git /opt/jupyterlab \
  && cd /opt/jupyterlab \
  && . /opt/venv/bin/activate \
  && pip install -e .

RUN echo "Installing Vega" \
  && . /opt/venv/bin/activate \
  && /opt/venv/bin/jupyter-nbextension install vega3 --py

RUN echo "Enabling Vega" \
  && . /opt/venv/bin/activate \
  && /opt/venv/bin/jupyter-nbextension enable vega3 --py

RUN echo "Installing Rise" \
  && . /opt/venv/bin/activate \
  && /opt/venv/bin/jupyter-nbextension install rise --py

RUN echo "Enabling Rise" \
  && . /opt/venv/bin/activate \
  && /opt/venv/bin/jupyter-nbextension enable rise --py

RUN ls /opt/antinex/core/ \
  && cp /opt/antinex/core/docker/update-all.sh /opt/antinex/update-all.sh \
  && chmod 777 /opt/antinex/core/docker/update-all.sh \
  && chmod 777 /opt/antinex/update-all.sh \
  && chmod 777 /opt/antinex/core/docker/jupyter/start-container.sh \
  && chmod 777 /opt/antinex/core/run-antinex-core.sh

RUN echo "Downgrading numpy and setuptools for tensorflow" \
  && . /opt/venv/bin/activate \
  && pip install --upgrade numpy==1.14.5 \
  && pip install --upgrade setuptools==39.1.0

ENV PIPELINE_PROCESSOR_SCRIPT /opt/antinex/pipeline/network_pipeline/scripts/packets_redis.py
ENV JUPYTER_START_SCRIPT /opt/antinex/core/docker/jupyter/start-container.sh
ENV NO_DB_WORKER_START_SCRIPT /opt/antinex/core/run-antinex-core.sh
ENV DJANGO_WORKER_START_SCRIPT /opt/antinex/api/run-worker.sh
ENV APP_START_SCRIPT /opt/antinex/api/run-django.sh
ENV MIGRATION_SCRIPT /opt/antinex/api/run-migrations.sh
ENV SHARED_LOG_CFG /opt/antinex/core/antinex_core/log/debug-openshift-logging.json
ENV API_LOG_CFG /opt/antinex/core/antinex_core/log/api-logging.json
ENV CLIENT_LOG_CFG /opt/antinex/core/antinex_core/log/client-logging.json
ENV CORE_LOG_CFG /opt/antinex/core/antinex_core/log/core-logging.json
ENV JUPYTER_LOG_CFG /opt/antinex/core/antinex_core/log/jupyter-logging.json
ENV PIPELINE_LOG_CFG /opt/antinex/core/antinex_core/log/pipeline-logging.json
ENV DEBUG_SHARED_LOG_CFG 0
ENV LOG_LEVEL DEBUG
ENV LOG_FILE /var/log/antinex/core/ai-core.log
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
ENV SKIP_BUILD_DOCS 1
ENV SKIP_COLLECT_STATICS 1
ENV JUPYTER_CONFIG /opt/antinex/core/docker/jupyter/jupyter_notebook_config.py
ENV NOTEBOOK_DIR /opt/antinex/core/docker/notebooks
ENV USE_ENV drf-dev
ENV USE_VENV /opt/venv

WORKDIR /opt/antinex/core

# set for anonymous user access in the container
RUN find /opt/antinex/api -type d -exec chmod 777 {} \;
RUN find /opt/antinex/core -type d -exec chmod 777 {} \;
RUN find /opt/antinex/client -type d -exec chmod 777 {} \;
RUN find /opt/antinex/pipeline -type d -exec chmod 777 {} \;
RUN find /opt/antinex/utils -type d -exec chmod 777 {} \;
RUN find /opt/antinex/antinex-datasets -type d -exec chmod 777 {} \;
RUN find /opt/antinex/datasets -type d -exec chmod 777 {} \;
RUN find /opt/deploy-to-kubernetes -type d -exec chmod 777 {} \;
RUN find /opt/spylunking -type d -exec chmod 777 {} \;
RUN find /opt/venv -type d -exec chmod 777 {} \;
RUN find /var/log -type d -exec chmod 777 {} \;

ENTRYPOINT /opt/antinex/core/run-antinex-core.sh
