#!/usr/bin/env bash
# Helper script to locally build and optionally push Nava Docker images to Docker Hub
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DOCKER_USER="${DOCKERHUB_USERNAME:-${1:-}}"
TAG="${2:-latest}"
SERVICE="${3:-all}"
DO_PUSH="${4:-false}"
PLATFORM="${PLATFORM:-${5:-linux/amd64}}"

if [[ -z "$DOCKER_USER" ]]; then
    echo "Usage: ./scripts/docker_build_and_push.sh <dockerhub_username> [tag] [service: all|stt|tts|server] [push: true|false] [platform: linux/amd64]"
    echo "Example: ./scripts/docker_build_and_push.sh mhosseinifar8 latest all false linux/amd64"
    echo "Or set DOCKERHUB_USERNAME environment variable."
    exit 1
fi

# Ensure lower-case username
DOCKER_USER=$(echo "$DOCKER_USER" | tr '[:upper:]' '[:lower:]')

echo "======================================================"
echo " Nava Docker Builder"
echo " User:     $DOCKER_USER"
echo " Tag:      $TAG"
echo " Platform: $PLATFORM"
echo " Service:  $SERVICE"
echo " Push:     $DO_PUSH"
echo "======================================================"

build_and_push() {
    local name="$1"
    local dockerfile="$2"
    local context="$3"
    local full_tag="${DOCKER_USER}/${name}:${TAG}"

    echo ""
    echo "==> Building ${full_tag} for ${PLATFORM}..."
    docker build --platform "${PLATFORM}" -t "${full_tag}" -f "${dockerfile}" "${context}"

    if [[ "$DO_PUSH" == "true" ]]; then
        echo "==> Pushing ${full_tag} to Docker Hub..."
        docker push "${full_tag}"
    fi
}

if [[ "$SERVICE" == "all" || "$SERVICE" == "stt" ]]; then
    build_and_push "nava-stt" "nava-stt/Dockerfile" "."
fi

if [[ "$SERVICE" == "all" || "$SERVICE" == "tts" ]]; then
    build_and_push "nava-tts" "nava-tts/Dockerfile" "."
fi

if [[ "$SERVICE" == "all" || "$SERVICE" == "server" ]]; then
    build_and_push "nava" "Dockerfile" "."
    docker tag "${DOCKER_USER}/nava:${TAG}" "${DOCKER_USER}/nava-server:${TAG}"
    if [[ "$DO_PUSH" == "true" ]]; then
        echo "==> Pushing ${DOCKER_USER}/nava-server:${TAG} to Docker Hub..."
        docker push "${DOCKER_USER}/nava-server:${TAG}"
    fi
fi

echo ""
echo "Docker images processed successfully!"

