#!/bin/bash

user_dir="/home/runner"
source $user_dir/.venvs/venvdrfpipeline/bin/activate

pip list --format=columns

host_as_slides="/home/runner/notebooks/Slides-AntiNex-Protecting-Django.ipynb"
if [[ "${SLIDE_NOTEBOOK}" != "" ]]; then
    host_as_slides="${SLIDE_NOTEBOOK}"
fi

if [[ -e ${host_as_slides} ]]; then
    echo "hosting notebook: ${host_as_slides} as slides"
    nohup $user_dir/.venvs/venvdrfpipeline/bin/jupyter-nbconvert \
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
    nohup $user_dir/.venvs/venvdrfpipeline/bin/jupyter-nbconvert \
        --to slides \
        --ServePostProcessor.port=8890 \
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
    --config=/home/runner/.jupyter/jupyter_notebook_config.py \
    --notebook-dir=/home/runner/notebooks --allow-root

exit 0
