#!/usr/bin/env bash
set -euo pipefail
model="${1:-}"
shift || true
device=cuda
workers=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --device) device="$2"; shift 2 ;;
    --workers) workers="$2"; shift 2 ;;
    *) echo 'Usage: scripts/run.sh MODEL [--device cpu|cuda] [--workers N]' >&2; exit 2 ;;
  esac
done
[[ "$device" == cpu || "$device" == cuda ]] || exit 2
[[ "$workers" =~ ^[1-9][0-9]*$ ]] || exit 2
case "$model" in
  e5-small) export E5_DEVICE="$device" E5_WORKERS="$workers" ;;
  user-bge-m3) export USER_DEVICE="$device" USER_WORKERS="$workers" ;;
  rapid-v5-mobile) export OCR_DEVICE="$device" OCR_WORKERS="$workers" ;;
  whisper-large-v3) export WHISPER_DEVICE="$device" WHISPER_WORKERS="$workers" ;;
  *) echo 'MODEL: e5-small | user-bge-m3 | rapid-v5-mobile | whisper-large-v3' >&2; exit 2 ;;
esac
mode=auto
[[ "$device" == cpu ]] && mode=cpu
exec bash "$(dirname "$0")/compose.sh" --mode "$mode" up -d --build "$model"
