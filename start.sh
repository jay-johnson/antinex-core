#!/bin/bash

cd docker
echo ""
echo "starting full-stack"
echo ""
docker-compose -f ./full-stack.yml up -d

exit 0
