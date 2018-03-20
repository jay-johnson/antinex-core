#!/bin/bash

echo ""
echo "stopping stack"
docker-compose -f ./full-stack.yml down

docker stop redis ai-core >> /dev/null 2>&1
docker rm redis ai-core >> /dev/null 2>&1
