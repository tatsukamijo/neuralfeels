#!/bin/bash
set -euo pipefail

# --- Constants ---
IMAGE_NAME="neuralfeels:latest"
USER_UID=$(id -u)
USER_GID=$(id -g)
USERNAME="neuralfeels"

# --- Build ---
echo "[INFO] Building Docker image $IMAGE_NAME with UID=$USER_UID, GID=$USER_GID..."
docker build \
  --build-arg USERNAME=$USERNAME \
  --build-arg USER_UID=$USER_UID \
  --build-arg USER_GID=$USER_GID \
  -t $IMAGE_NAME .

echo "[INFO] Docker image $IMAGE_NAME built successfully." 