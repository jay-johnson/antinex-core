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

if [[ ! -e $install_dir/api ]]; then
    echo "cloning api"
    git clone https://github.com/jay-johnson/train-ai-with-django-swagger-jwt.git $install_dir/api
else
    if [[ "${UPDATE}" == "1" ]]; then
        cd $install_dir/api
        git pull
    fi
fi

if [[ ! -e $install_dir/client ]]; then
    echo "cloning client"
    git clone https://github.com/jay-johnson/antinex-client.git $install_dir/client
else
    if [[ "${UPDATE}" == "1" ]]; then
        cd $install_dir/client
        git pull
    fi
fi
if [[ ! -e $install_dir/utils ]]; then
    echo "cloning utils"
    git clone https://github.com/jay-johnson/antinex-utils.git $install_dir/utils
else
    if [[ "${UPDATE}" == "1" ]]; then
        cd $install_dir/utils
        git pull
    fi
fi
if [[ ! -e $install_dir/core ]]; then
    echo "cloning core"
    git clone https://github.com/jay-johnson/antinex-core.git $install_dir/core
else
    if [[ "${UPDATE}" == "1" ]]; then
        cd $install_dir/core
        git pull
    fi
fi
if [[ ! -e $install_dir/pipeline ]]; then
    echo "cloning pipeline"
    git clone https://github.com/jay-johnson/network-pipeline.git $install_dir/pipeline
else
    if [[ "${UPDATE}" == "1" ]]; then
        cd $install_dir/pipeline
        git pull
    fi
fi
if [[ ! -e $install_dir/antinex-datasets ]]; then
    echo "cloning antinex datasets"
    git clone https://github.com/jay-johnson/antinex-datasets.git $install_dir/antinex-datasets
else
    if [[ "${UPDATE}" == "1" ]]; then
        cd $install_dir/antinex-datasets
        git pull
    fi
fi
if [[ ! -e $install_dir/datasets ]]; then
    echo "cloning network pipeline datasets"
    git clone https://github.com/jay-johnson/network-pipeline-datasets.git $install_dir/datasets
else
    if [[ "${UPDATE}" == "1" ]]; then
        cd $install_dir/datasets
        git pull
    fi
fi

# legacy docs use these paths. it needs to be removed in the future
if [[ ! -e /opt/antinex-datasets ]]; then
    ln -s $install_dir/antinex-datasets /opt/antinex-datasets
fi
if [[ ! -e /opt/datasets ]]; then
    ln -s $install_dir/datasets /opt/datasets
fi

echo "installing pipeline"
cd $install_dir/pipeline
pip install -e .
echo "installing utils"
cd $install_dir/utils
pip install -e .
echo "installing client"
cd $install_dir/client
pip install -e .
echo "installing core"
cd $install_dir/core
pip install -e .
echo "installing api"
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
