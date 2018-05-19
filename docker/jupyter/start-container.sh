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

pip list --format=columns

if [[ "${SHARED_LOG_CFG}" != "" ]]; then
    echo ""
    echo "Logging config: ${SHARED_LOG_CFG}"
    echo ""
fi

if [[ "${SUPPORTS_SLIDES}" == "1" ]]; then
    host_as_slides="/home/runner/notebooks/Slides-AntiNex-Protecting-Django.ipynb"
    if [[ "${SLIDE_NOTEBOOK}" != "" ]]; then
        host_as_slides="${SLIDE_NOTEBOOK}"
    fi

    if [[ -e ${host_as_slides} ]]; then
        echo "hosting notebook: ${host_as_slides} as slides"
        nohup ${venv}/bin/jupyter-nbconvert \
            --to slides \
            --ServePostProcessor.port=8889 \
            --ServePostProcessor.ip="0.0.0.0" \
            "${host_as_slides}" \
            --reveal-prefix=reveal.js \
            --post serve &
    else
        echo "failed to find notebook for slides: ${host_as_slides}"
    fi

    host_as_slides="/home/runner/notebooks/Slides-AntiNex-Using-Pre-Trained-Deep-Neural-Networks-For-Defense.ipynb"
    if [[ "${SLIDE_NOTEBOOK_PRE_TRAINED}" != "" ]]; then
        host_as_slides="${SLIDE_NOTEBOOK_PRE_TRAINED}"
    fi

    if [[ -e ${host_as_slides} ]]; then
        echo "hosting notebook: ${host_as_slides} as slides"
        nohup ${venv}/bin/jupyter-nbconvert \
            --to slides \
            --ServePostProcessor.port=8890 \
            --ServePostProcessor.ip="0.0.0.0" \
            "${host_as_slides}" \
            --reveal-prefix=reveal.js \
            --post serve &
    else
        echo "failed to find notebook for slides: ${host_as_slides}"
    fi
fi
# end of slide hosting if needed

echo "Starting Jupyter"
notebook_config=/opt/antinex-core/docker/jupyter/jupyter_notebook_config.py
notebook_dir=/opt/antinex-core/docker/notebooks

if [[ "${JUPYTER_CONFIG}" != "" ]]; then
    if [[ -e ${JUPYTER_CONFIG} ]]; then
        notebook_config=${JUPYTER_CONFIG}
        echo " - using notebook_config: ${notebook_config}"
    else
        echo " - Failed to find notebook_config: ${JUPYTER_CONFIG}"
    fi
fi

if [[ "${NOTEBOOK_DIR}" != "" ]]; then
    if [[ -e ${NOTEBOOK_DIR} ]]; then
        notebook_dir=${NOTEBOOK_DIR}
        echo " - using notebook_dir: ${notebook_dir}"
    else
        echo " - Failed to find notebook_dir: ${NOTEBOOK_DIR}"
    fi
fi

echo ""
echo "Starting Jupyter with command: "
echo "jupyter notebook --ip=* --port=8888 --no-browser --config=${notebook_config} --notebook-dir=${notebook_dir} --allow-root"
jupyter notebook \
    --ip=* \
    --port=8888 \
    --no-browser \
    --config=${notebook_config} \
    --notebook-dir=${notebook_dir} \
    --allow-root

exit 0
