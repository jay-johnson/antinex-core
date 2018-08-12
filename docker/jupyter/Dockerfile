FROM jayjohnson/ai-core:latest

RUN echo "creating project directories" \
  && mkdir -p -m 777 /var/log/antinex/jupyter \
  && mkdir -p -m 777 /opt/antinex \
  && chmod 777 //var/log/antinex/jupyter \
  && touch /var/log/antinex/jupyter/jupyter.log \
  && chmod 777 /var/log/antinex/jupyter/jupyter.log \
  && echo "updating repo" \
  && cd /opt/antinex/core \
  && git checkout master \
  && git pull \
  && echo "checking repos in container" \
  && ls -l /opt/antinex/core \
  && echo "activating venv" \
  && . /opt/venv/bin/activate \
  && cd /opt/antinex/core \
  && echo "installing pip upgrades" \
  && pip install --upgrade -e .

RUN echo "Downgrading numpy and setuptools for tensorflow" \
  && . /opt/venv/bin/activate \
  && pip install --upgrade numpy==1.14.5 \
  && pip install --upgrade setuptools==39.1.0

RUN echo "Making Sphinx docs" \
  && . /opt/venv/bin/activate \
  && cd /opt/antinex/core/docs \
  && ls -l \
  && make html

ENV PROJECT_NAME="jupyter" \
    SHARED_LOG_CFG="/opt/antinex/core/antinex_core/log/debug-openshift-logging.json" \
    DEBUG_SHARED_LOG_CFG="0" \
    LOG_LEVEL="DEBUG" \
    LOG_FILE="/var/log/antinex/jupyter/jupyter.log" \
    USE_ENV="drf-dev" \
    USE_VENV="/opt/venv" \
    API_USER="trex" \
    API_PASSWORD="123321" \
    API_EMAIL="bugs@antinex.com" \
    API_FIRSTNAME="Guest" \
    API_LASTNAME="Guest" \
    API_URL="http://api.antinex.com:8010" \
    API_VERBOSE="true" \
    API_DEBUG="false" \
    USE_FILE="false" \
    SILENT="-s" \
    RUN_JUPYTER="/opt/antinex/core/docker/jupyter/start-container.sh"

WORKDIR /opt/antinex/core/docker/jupyter

# set for anonymous user access in the container
RUN find /opt/antinex -type d -exec chmod 777 {} \;
RUN find /var/log/antinex -type d -exec chmod 777 {} \;

ENTRYPOINT . /opt/venv/bin/activate \
  && /opt/antinex/core/docker/jupyter \
  && ${RUN_JUPYTER}
