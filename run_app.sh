#!/bin/bash
IMAGE_NAME="alpha-dominator"
docker stop $(docker ps -q --filter ancestor=$IMAGE_NAME) 2>/dev/null
docker build --no-cache -t $IMAGE_NAME .
docker run --rm --dns 8.8.8.8 -p 8501:8501 $IMAGE_NAME