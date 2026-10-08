#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd -P)"
cd "$root"
mode="${INFERENCE_MODE:-auto}"
if [[ "${1:-}" == --mode ]]; then
  [[ $# -ge 2 ]] || { echo 'Missing --mode value' >&2; exit 2; }
  mode="$2"; shift 2
fi
if [[ "$mode" == auto ]]; then
  if [[ -e /dev/dxg && -d /usr/lib/wsl ]]; then mode=wsl
  elif command -v nvidia-smi >/dev/null && nvidia-smi >/dev/null 2>&1; then mode=gpu
  else mode=cpu; fi
fi
case "$mode" in cpu|gpu|wsl) ;; *) echo 'Mode: auto | cpu | gpu | wsl' >&2; exit 2;; esac
if [[ -z "${DOCKER_CONFIG:-}" ]]; then
  export DOCKER_CONFIG="$root/.docker-client"
  mkdir -p "$DOCKER_CONFIG"
fi
files=(-f compose.yaml -f "compose.$mode.yaml")
[[ -f compose.local.yaml ]] && files+=(-f compose.local.yaml)
exec docker compose "${files[@]}" "$@"
