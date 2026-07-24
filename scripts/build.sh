#!/bin/bash
set -e

# Configuration
IMAGE_NAME="jeremee02/qwen3-tts"
VERSION="0.0.1"

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=========================================="
echo "Building Qwen3-TTS Docker Image"
echo "=========================================="
echo "Image: ${IMAGE_NAME}"
echo "Version: ${VERSION}"
echo "Context: ${PROJECT_ROOT}"
echo "=========================================="

# Change to project root
cd "$PROJECT_ROOT"

# Build the Docker image
echo ""
echo "[1/3] Building Docker image..."
docker build \
    --build-arg BUILD_DATE="$(date -u +"%Y-%m-%dT%H:%M:%SZ")" \
    --build-arg VERSION="${VERSION}" \
    -t "${IMAGE_NAME}:${VERSION}" \
    -t "${IMAGE_NAME}:latest" \
    .

echo ""
echo "[2/3] Build completed successfully!"
echo "  - ${IMAGE_NAME}:${VERSION}"
echo "  - ${IMAGE_NAME}:latest"

# Push to Docker Hub
echo ""
echo "[3/3] Pushing images to Docker Hub..."
docker push "${IMAGE_NAME}:${VERSION}"
docker push "${IMAGE_NAME}:latest"

echo ""
echo "=========================================="
echo "Build and push completed successfully!"
echo "=========================================="
echo "Images available:"
echo "  docker pull ${IMAGE_NAME}:${VERSION}"
echo "  docker pull ${IMAGE_NAME}:latest"
echo "=========================================="
