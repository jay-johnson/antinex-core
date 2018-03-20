#!/bin/bash

host_as_slides="/root/notebooks/Slides-AntiNex-Protecting-Django.ipynb"
if [[ "${SLIDE_NOTEBOOK}" != "" ]]; then
    host_as_slides="${SLIDE_NOTEBOOK}"
fi

if [[ -e ${host_as_slides} ]]; then
    echo "hosting notebook: ${host_as_slides} as slides"
    nohup python /usr/local/bin/jupyter-nbconvert \
        --to slides \
        --ServePostProcessor.port=8889 \
        --ServePostProcessor.ip="0.0.0.0" \
        "${host_as_slides}" \
        --reveal-prefix=reveal.js \
        --post serve &
else
    echo "failed to find notebook for slides: ${host_as_slides}"
fi

echo "Starting Jupyter"
jupyter notebook --ip=* --port=8888 \
    --no-browser \
    --config=/root/.jupyter/jupyter_notebook_config.py \
    --notebook-dir=/root/notebooks --allow-root

exit 0
