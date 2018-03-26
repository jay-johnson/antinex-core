#!/bin/bash

source ~/.venvs/venvdrfpipeline/bin/activate

repos="/opt/antinex-api /opt/antinex-core /opt/antinex-utils /opt/antinex-client /opt/antinex-datasets /opt/antinex-pipeline /opt/datasets"

echo "updating repos: ${repos}"

for i in $repos; do
    echo " - updating ${i}"
    cd $i
    echo " - pulling latest"
    git pull
    echo " - installing"

    if [[ -e "${i}/setup.py" ]]; then
        echo " - install using ${i}/setup.py"
        pip install --upgrade -e .
    elif [[ -e "${i}/requirements.txt" ]]; then
        echo " - install using ${i}/requirements.txt"
        pip install --upgrade -r ./requirements.txt
    else
        echo " - nothing to install for ${i}"
    fi
done

exit 0
