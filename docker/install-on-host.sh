#!/bin/bash

venv=~/.venvs/venvdrfpipeline

echo "creating virtualenv: ${venv}"
if [[ ! -e $venv ]]; then
    virtualenv -p python3 ${venv}
fi
if [[ ! -e $venv ]]; then
    virtualenv -p python3 ${venv}
    echo "Failed finding virtualenv: ${venv}"
fi
. ${venv}/bin/activate

install_dir="/opt/antinex"

echo "cloning repos to ${install_dir}: ${venv}"
if [[ ! -e $install_dir ]]; then
    mkdir -p -m 775 ${install_dir}
fi
echo "cloning repos to ${install_dir}: ${venv}"

git clone https://github.com/jay-johnson/train-ai-with-django-swagger-jwt.git $install_dir/api
git clone https://github.com/jay-johnson/antinex-client.git $install_dir/client
git clone https://github.com/jay-johnson/antinex-utils.git $install_dir/utils
git clone https://github.com/jay-johnson/antinex-core.git $install_dir/core
git clone https://github.com/jay-johnson/network-pipeline.git $install_dir/pipeline
git clone https://github.com/jay-johnson/antinex-datasets.git $install_dir/antinex-datasets
git clone https://github.com/jay-johnson/network-pipeline-datasets.git $install_dir/datasets

cd $install_dir/pipeline
pip install -e .
cd $install_dir/utils
pip install -e .
cd $install_dir/client
pip install -e .
cd $install_dir/core
pip install -e .
cd $install_dir/api
pip install -r ./requirements.txt

echo ""
echo "Activate virtualenv with command:"
echo "source ${venv}/bin/activate"

echo ""
echo "set your bashrc alias with:"
echo ""
echo "echo 'alias dev=\"cd /opt/antinex/api && source ~/.venvs/venvdrfpipeline/bin/activate\"' >> ~/.bashrc"

exit 0
