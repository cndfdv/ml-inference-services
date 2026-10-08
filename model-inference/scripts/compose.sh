#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"
mode="${INFERENCE_MODE:-auto}"
if [[ "${1:-}" == --mode ]]; then mode="$2"; shift 2; fi
if [[ "$mode" == auto ]]; then
  if [[ -e /dev/dxg && -d /usr/lib/wsl ]]; then mode=wsl; else mode=gpu; fi
fi
case "$mode" in
  gpu|wsl|cpu) ;;
  *) echo 'Usage: scripts/compose.sh [--mode gpu|wsl|cpu|auto] COMPOSE_ARGS...' >&2; exit 2;;
esac
# Avoid modifying the user's Docker context or credential configuration.
if [[ -z "${DOCKER_CONFIG:-}" ]]; then
  export DOCKER_CONFIG="$root/.docker-client"
  mkdir -p "$DOCKER_CONFIG"
fi
exec docker compose -f compose.yaml -f "compose.$mode.yaml" "$@"
