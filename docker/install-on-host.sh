#!/bin/bash

logo=" 
        db                              88  888b      88                          
       d88b                      ,d     88  8888b     88                          
      d8  8b                     88         88  8b    88                          
     d8    8b      8b,dPPYba,  MM88MMM  88  88   8b   88   ,adPPYba,  8b,     ,d8 
    d8YaaaaY8b     88P'     8a   88     88  88    8b  88  a8P_____88    Y8, ,8P   
   d88******88b    88       88   88     88  88     8b 88  8PP******      )888(    
  d8          8b   88       88   88,    88  88      8888  *8b,   ,aa   ,d8   8b,  
 d8            8b  88       88   *Y888  88  88       888   *Ybbd88*   8P       Y8 " 

txtund=""
txtbld=""
blddkg=""
bldred=""
bldblu=""
bldylw=""
bldgrn=""
bldgry=""
bldpnk=""
bldwht=""
txtrst=""

# check if stdout is a terminal...
if test -t 1; then
    if [[ -e /usr/bin/tput ]]; then
        # see if it supports colors...
        ncolors=$(tput colors)

        if test -n "$ncolors" && test $ncolors -ge 8; then

            txtund=$(tput sgr 0 1)          # Underline
            txtbld=$(tput bold)             # Bold
            blddkg=${txtbld}$(tput setaf 0) # Dark Gray
            bldred=${txtbld}$(tput setaf 1) # Red
            bldblu=${txtbld}$(tput setaf 2) # Blue
            bldylw=${txtbld}$(tput setaf 3) # Yellow
            bldgrn=${txtbld}$(tput setaf 4) # Green
            bldgry=${txtbld}$(tput setaf 5) # Gray
            bldpnk=${txtbld}$(tput setaf 6) # Pink
            bldwht=${txtbld}$(tput setaf 7) # White
            txtrst=$(tput sgr0)             # Reset
        fi
    fi
fi

anmt() {
    echo "${bldylw}$@ $txtrst"
}
info() {
    echo "${bldgrn}$@ $txtrst"
}
err() {
    echo "${bldred}$@ $txtrst"
}

should_update=0
if [[ $# -ge 1 ]]; then
    if [[ "$1" == "update" ]]; then
        should_update=1
    fi
fi
if [[ "${UPDATE}" == "1" ]]; then
    should_update=1
fi

echo ""
start_date=$(date +"%Y-%m-%d %H:%M:%S")
if [[ "${should_update}" == "0" ]]; then
    anmt "Welcome - Starting install: ${start_date}"
else
    anmt "Welcome - Update started: ${start_date}"
fi
echo ""
info "${logo}"
echo ""

venv=~/.venvs/venvdrfpipeline
transport_prefix="https://github.com/"

echo "creating virtualenv: ${venv}"
if [[ ! -e $venv ]]; then
    virtualenv -p python3 ${venv}
fi
if [[ ! -e $venv ]]; then
    err "Failed finding virtualenv in directory: ${venv}"
    err ""
    err "the command was:"
    err "virtualenv -p python3 ${venv}"
    err ""
    exit 1
fi
. ${venv}/bin/activate

install_dir="/opt/antinex"

# remove this once the github org is setup with repos
if [[ "${GH_TRANSPORT}" != "" ]]; then
    transport_prefix="git@github.com:"
fi

anmt "cloning repos to ${install_dir} with virtualenv: ${venv}"
if [[ ! -e $install_dir ]]; then
    mkdir -p -m 775 ${install_dir}
fi

if [[ ! -e $install_dir/api ]]; then
    anmt "cloning api"
    git clone ${transport_prefix}jay-johnson/train-ai-with-django-swagger-jwt.git $install_dir/api
else
    if [[ "${should_update}" == "1" ]]; then
        anmt "updating api"
        cd $install_dir/api
        git pull
    fi
fi
if [[ ! -e $install_dir/client ]]; then
    anmt "cloning client"
    git clone ${transport_prefix}jay-johnson/antinex-client.git $install_dir/client
else
    if [[ "${should_update}" == "1" ]]; then
        anmt "updating client"
        cd $install_dir/client
        git pull
    fi
fi
if [[ ! -e $install_dir/utils ]]; then
    anmt "cloning utils"
    git clone ${transport_prefix}jay-johnson/antinex-utils.git $install_dir/utils
else
    if [[ "${should_update}" == "1" ]]; then
        anmt "updating utils"
        cd $install_dir/utils
        git pull
    fi
fi
if [[ ! -e $install_dir/core ]]; then
    anmt "cloning core"
    git clone ${transport_prefix}jay-johnson/antinex-core.git $install_dir/core
else
    if [[ "${should_update}" == "1" ]]; then
        anmt "updating core"
        cd $install_dir/core
        git pull
    fi
fi
if [[ ! -e $install_dir/pipeline ]]; then
    anmt "cloning pipeline"
    git clone ${transport_prefix}jay-johnson/network-pipeline.git $install_dir/pipeline
else
    if [[ "${should_update}" == "1" ]]; then
        anmt "updating pipeline"
        cd $install_dir/pipeline
        git pull
    fi
fi
if [[ ! -e $install_dir/antinex-datasets ]]; then
    anmt "cloning antinex datasets"
    git clone ${transport_prefix}jay-johnson/antinex-datasets.git $install_dir/antinex-datasets
else
    if [[ "${should_update}" == "1" ]]; then
        anmt "updating antinex datasets"
        cd $install_dir/antinex-datasets
        git pull
    fi
fi
if [[ ! -e $install_dir/datasets ]]; then
    anmt "cloning network pipeline datasets"
    git clone ${transport_prefix}jay-johnson/network-pipeline-datasets.git $install_dir/datasets
else
    if [[ "${should_update}" == "1" ]]; then
        anmt "updating network pipeline datasets"
        cd $install_dir/datasets
        git pull
    fi
fi
if [[ ! -e /opt/spylunking ]]; then
    anmt "cloning spylunking logging"
    git clone ${transport_prefix}jay-johnson/spylunking.git /opt/spylunking
else
    if [[ "${should_update}" == "1" ]]; then
        anmt "updating spylunking"
        cd /opt/spylunking
        git pull
    fi
fi

anmt "installing spylunking logger"
cd /opt/spylunking
pip install -e .
anmt "installing pipeline"
cd $install_dir/pipeline
pip install -e .
anmt "installing utils"
cd $install_dir/utils
pip install -e .
anmt "installing client"
cd $install_dir/client
pip install -e .
anmt "installing core"
cd $install_dir/core
pip install -e .
anmt "installing api"
cd $install_dir/api
pip install -r ./requirements.txt

finish_date=$(date +"%Y-%m-%d %H:%M:%S")

if [[ "${should_update}" == "0" ]]; then
    anmt "Install finished: ${finish_date}"

    echo ""
    anmt "Activate virtualenv with command:"
    echo "source ${venv}/bin/activate"

    echo ""
    anmt "Set your bashrc alias by running these commands in your shell:"
    echo ""
    echo "echo \"\" >> ~/.bashrc"
    echo "echo '###################################' >> ~/.bashrc"
    echo "echo '#' >> ~/.bashrc"
    echo "echo '# AntiNex nav aliases' >> ~/.bashrc"
    echo "echo '#' >> ~/.bashrc"

    echo "echo 'alias api=\"cd /opt/antinex/api && source ~/.venvs/venvdrfpipeline/bin/activate\"' >> ~/.bashrc"
    echo "echo 'alias core=\"cd /opt/antinex/core && source ~/.venvs/venvdrfpipeline/bin/activate\"' >> ~/.bashrc"
    echo "echo 'alias client=\"cd /opt/antinex/client && source ~/.venvs/venvdrfpipeline/bin/activate\"' >> ~/.bashrc"
    echo "echo 'alias dbmigrate=\"source ~/.venvs/venvdrfpipeline/bin/activate && source /opt/antinex/api/envs/drf-dev.env && cd /opt/antinex/api && ./run-migrations.sh\"' >> ~/.bashrc"
    echo "echo 'alias dev=\"cd /opt/antinex/api && source ~/.venvs/venvdrfpipeline/bin/activate\"' >> ~/.bashrc"
    echo "echo 'alias docs=\"cd /opt/antinex/api/webapp/drf_network_pipeline/docs && source ~/.venvs/venvdrfpipeline/bin/activate && make html\"' >> ~/.bashrc"
    echo "echo 'alias pipeline=\"cd /opt/antinex/pipeline && source ~/.venvs/venvdrfpipeline/bin/activate\"' >> ~/.bashrc"
    echo "echo 'alias sqlmigrate=\"source ~/.venvs/venvdrfpipeline/bin/activate && source /opt/antinex/api/envs/dev.env && cd /opt/antinex/api && ./run-migrations.sh\"' >> ~/.bashrc"
    echo "echo 'alias spylunk=\"source ~/.venvs/venvdrfpipeline/bin/activate && cd /opt/spylunking && pip install -e .\"' >> ~/.bashrc"
    echo "echo 'alias run=\"source ~/.venvs/venvdrfpipeline/bin/activate && source /opt/antinex/api/envs/dev.env && cd /opt/antinex/api/webapp && python manage.py runserver 0.0.0.0:8010\"' >> ~/.bashrc"
    echo "echo 'alias utils=\"cd /opt/antinex/utils && source ~/.venvs/venvdrfpipeline/bin/activate\"' >> ~/.bashrc"
else
    anmt "Update finished: ${finish_date}"
fi

exit 0
