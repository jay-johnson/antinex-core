#!/bin/bash

default_venv="/opt/venv"
venv=${default_venv}
if [[ "${1}" != "" ]]; then
    venv="${1}"
fi

activate_path=$(echo ${venv} | grep activate | wc -l)
if [[ "${activate_path}" == "0" ]]; then
    venv=${venv}/bin/activate
fi

if [[ ! -f "${venv}" ]]; then
    venv=${default_venv}
    echo "creating virtualenv: ${venv}"
    virtualenv -p python3 ${venv}
    if [[ "$?" != "0" ]]; then
        echo "Failed creating virtualenv ${venv} with command:"
        echo ""
        echo "virtualenv -p python3 ${venv}"
        echo ""
        exit 1
    fi
    venv=${default_venv}/bin/activate
fi

echo "loading venv: ${venv}"
. ${venv}
if [[ "$?" != "0" ]]; then
    echo " - failed loading virtualenv: ${venv}"
    echo ""
    echo ". ${venv}"
    echo ""
    exit 1
fi

# the install order matters to prevent pips getting installed from the local directories instead of pulling from pypi
org_dir=$(pwd)
repos="/opt/spylunking /opt/antinex/utils /opt/antinex/pipeline /opt/antinex/client /opt/antinex/core /opt/antinex/api /opt/antinex/antinex-datasets /opt/antinex/datasets"
for d in ${repos}; do
    echo ""
    if [[ ! -e ${d} ]]; then
        repo_url=""
        if [[ "${d}" == "/opt/antinex/antinex-datasets" ]]; then
            repo_url="https://github.com/jay-johnson/antinex-datasets.git"
        elif [[ "${d}" == "/opt/antinex/api" ]]; then
            repo_url="https://github.com/jay-johnson/train-ai-with-django-swagger-jwt.git"
        elif [[ "${d}" == "/opt/antinex/client" ]]; then
            repo_url="https://github.com/jay-johnson/antinex-client.git"
        elif [[ "${d}" == "/opt/antinex/core" ]]; then
            repo_url="https://github.com/jay-johnson/antinex-core.git"
        elif [[ "${d}" == "/opt/antinex/datasets" ]]; then
            repo_url="https://github.com/jay-johnson/network-pipeline-datasets.git"
        elif [[ "${d}" == "/opt/antinex/pipeline" ]]; then
            repo_url="https://github.com/jay-johnson/network-pipeline.git"
        elif [[ "${d}" == "/opt/antinex/utils" ]]; then
            repo_url="https://github.com/jay-johnson/antinex-utils.git"
        elif [[ "${d}" == "/opt/spylunking" ]]; then
            repo_url="https://github.com/jay-johnson/spylunking"
        else
            echo "Unknown directory: ${d}"
            repo_url=""
        fi

        if [[ "${repo_url}" != "" ]]; then
            echo ""
            echo "cloning project: ${d}"
            git clone ${repo_url} ${d}
            if [[ "$?" != "0" ]]; then
                echo ""
                echo "Failed cloning project: ${d} with command:"
                echo "git clone ${repo_url} ${d}"
                echo ""
                exit 1
            fi
        fi
    fi
    cd $d
    if [[ -e .git/config ]]; then
        echo " - pulling latest"
        git pull
        if [[ "$?" != "0" ]]; then
            echo ""
            echo "Failed to update: ${d}"
            echo "cd ${d} && git pull"
            echo ""
        fi
    fi
    if [[ -e setup.py ]]; then
        echo " - installing"
        pip install -e .
        if [[ "$?" != "0" ]]; then
            echo ""
            echo "Failed to install latest: ${d}"
            echo "cd ${d} && pip install -e ."
            echo ""
        fi
    fi
done

cd ${org_dir}
echo ""
echo "load the updated virtualenv with:"
echo ". ${venv}"
echo ""
echo "done"
exit 0
