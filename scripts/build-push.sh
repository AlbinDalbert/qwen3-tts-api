#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-ghcr.io/albindalbert/qwen3-tts-api:latest}"
PLATFORM_ARG=()

if [[ -n "${PLATFORM:-}" ]]; then
  PLATFORM_ARG=(--platform "$PLATFORM")
fi

echo "Building $IMAGE"
docker build "${PLATFORM_ARG[@]}" -t "$IMAGE" .

echo "Pushing $IMAGE"
docker push "$IMAGE"
