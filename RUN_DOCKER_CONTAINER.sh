#!/bin/bash
set -euo pipefail

# --- Usage check ---
if [ $# -lt 1 ]; then
  echo "Usage: $0 <NAME>"
  exit 1
fi

NAME="$1"
IMAGE_NAME="neuralfeels:latest"
HOST_DIR=$(pwd)
CONTAINER_DIR="/workspace/neuralfeels"
CONTAINER_NAME="${NAME}_neuralfeels_dev"

# --- Run ---
echo "[INFO] Running Docker container $CONTAINER_NAME from image $IMAGE_NAME..."
docker run --rm -it \
  --gpus all \
  --name "$CONTAINER_NAME" \
  -v "$HOST_DIR":"$CONTAINER_DIR" \
  -w "$CONTAINER_DIR" \
  $IMAGE_NAME 