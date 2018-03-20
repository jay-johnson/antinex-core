#!/bin/bash

cd docker
echo ""
echo "stopping full-stack"
echo ""
docker-compose -f ./full-stack.yml stop

echo "stopping"
docker stop redis keras-jupyter >> /dev/null 2>&1
echo "removing"
docker rm redis keras-jupyter >> /dev/null 2>&1
echo ""

exit 0
