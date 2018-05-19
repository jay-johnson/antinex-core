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

if [[ "${num_workers}" == "1" ]]; then
    echo "Starting Worker=${worker_module}"
    echo "celery worker -A ${worker_module} -c ${num_workers} -l ${log_level} -n ${worker_name}"
    celery worker -A $worker_module -c ${num_workers} -l ${log_level} -n ${worker_name}
else
    echo "Starting Workers=${worker_module}"
    echo "celery worker -A ${worker_module} -c ${num_workers} -l ${log_level} -n ${worker_name}"
    celery worker -A $worker_module -c ${num_workers} -l ${log_level} -n ${worker_name}
fi
echo ""
