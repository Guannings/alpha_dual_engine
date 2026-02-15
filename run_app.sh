#!/bin/bash

IMAGE_NAME="alpha-dual_engine"

# 1. Clear old containers (Silent mode)
docker rm -f $(docker ps -a -q --filter ancestor=$IMAGE_NAME) 2>/dev/null

# 2. Build silently (only show errors)
echo "🔨 Building Alpha Engine..."
docker build -q -t $IMAGE_NAME . > /dev/null

# 3. Run
echo "✅ Starting... (Wait for the URL below)"
docker run --rm --dns 8.8.8.8 -p 8501:8501 $IMAGE_NAME