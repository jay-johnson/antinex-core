#!/bin/bash

venv=~/.venvs/venvdrfpipeline

# support for using venv in other locations
if [[ "${USE_VENV}" != "" ]]; then
    if [[ -e ${USE_VENV}/bin/activate ]]; then
        echo "Using custom virtualenv: ${USE_VENV}"
        venv=${USE_VENV}
    else
        echo "Did not find custom virtualenv: ${USE_VENV}"
        exit 1
    fi
fi

echo "Activating pips: ${venv}/bin/activate"
. ${venv}/bin/activate
echo ""

pip list

echo ""
echo "Loading Celery environment variables"
echo ""

num_workers=1
log_level=DEBUG
log_file=/tmp/core.log
worker_module=antinex_core.antinex_worker
worker_name="antinexcore@%h"

if [[ "${NUM_WORKERS}" != "" ]]; then
    num_workers=$NUM_WORKERS
fi
if [[ "${LOG_LEVEL}" != "" ]]; then
    log_level=$LOG_LEVEL
fi
if [[ "${LOG_FILE}" != "" ]]; then
    log_file=$LOG_FILE
fi
if [[ "${WORKER_MODULE}" != "" ]]; then
    worker_module=$WORKER_MODULE
fi
if [[ "${WORKER_NAME}" != "" ]]; then
    worker_name=$WORKER_NAME
fi

if [[ "${SHARED_LOG_CFG}" != "" ]]; then
    echo ""
    echo "Logging config: ${SHARED_LOG_CFG}"
    echo ""
fi

# allow overriding the core's broker url
if  [[ "${ANTINEX_CORE_BROKER_URL}" != "" ]]; then
    export BROKER_URL="${ANTINEX_CORE_BROKER_URL}"
fi
if  [[ "${ANTINEX_CORE_NUM_WORKERS}" != "" ]]; then
    num_workers=${ANTINEX_CORE_NUM_WORKERS}
fi

# Use the CORE_EXTRA_ARGS to pass in specific args:
# http://docs.celeryproject.org/en/latest/reference/celery.bin.worker.html
#
# example args from 4.2.0:
# --without-heartbeat
# --heartbeat-interval N
# --without-gossip
# --without-mingle

if [[ "${ANTINEX_CORE_WORKER_ARGS}" != "" ]]; then
    echo "Launch custom core worker: ${ANTINEX_CORE_WORKER_ARGS}"
    celery worker ${ANTINEX_CORE_WORKER_ARGS}
elif [[ "${num_workers}" == "1" ]]; then
    echo "Starting core worker=${worker_module}"
    echo "celery worker -A ${worker_module} -c ${num_workers} -l ${log_level} -n ${worker_name} ${CORE_EXTRA_ARGS}"
    celery worker -A $worker_module -c ${num_workers} -l ${log_level} -n ${worker_name} ${CORE_EXTRA_ARGS}
else
    echo "Starting core workers=${worker_module}"
    echo "celery worker -A ${worker_module} -c ${num_workers} -l ${log_level} -n ${worker_name} ${CORE_EXTRA_ARGS}"
    celery worker -A $worker_module -c ${num_workers} -l ${log_level} -n ${worker_name} ${CORE_EXTRA_ARGS}
fi
echo ""
