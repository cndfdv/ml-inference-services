#!/usr/bin/env bash
set -euo pipefail
model="${1:-}"
[[ $# -gt 0 ]] && shift
workers=1
device=''
while [[ $# -gt 0 ]]; do
  [[ $# -ge 2 ]] || { echo 'Missing option value' >&2; exit 2; }
  case "$1" in
    --device) device="$2" ;;
    --workers) workers="$2" ;;
    *) echo 'Usage: scripts/run.sh TASK/MODEL [--device cpu|cuda] [--workers N]' >&2; exit 2 ;;
  esac
  shift 2
done
[[ "$workers" =~ ^[1-9][0-9]*$ ]] || { echo 'Workers must be positive' >&2; exit 2; }
case "$model" in
  embeddings/e5-small|e5-small) service=e5-small; prefix=E5 ;;
  embeddings/user-bge-m3|user-bge-m3) service=user-bge-m3; prefix=USER ;;
  ocr/rapid-v5-mobile|rapid-v5-mobile) service=rapid-v5-mobile; prefix=OCR ;;
  asr/whisper|whisper|whisper-large-v3) service=whisper-large-v3; prefix=WHISPER ;;
  asr/gigaam|gigaam) service=gigaam; prefix=GIGAAM ;;
  ocr/easyocr|easyocr) service=easyocr; prefix=EASYOCR ;;
  *) echo 'Model must identify a folder under asr/, embeddings/ or ocr/' >&2; exit 2 ;;
esac
if [[ "$service" == gigaam || "$service" == easyocr ]]; then
  device="${device:-cpu}"
  [[ "$device" == cpu ]] || { echo "$service supports CPU only" >&2; exit 2; }
else
  device="${device:-cuda}"
fi
[[ "$device" == cpu || "$device" == cuda ]] || { echo 'Device must be cpu or cuda' >&2; exit 2; }
export "${prefix}_DEVICE=$device" "${prefix}_WORKERS=$workers"
mode=cpu
if [[ "$device" == cuda ]]; then
  mode=gpu
  [[ -e /dev/dxg && -d /usr/lib/wsl ]] && mode=wsl
fi
exec bash "$(dirname "$0")/compose.sh" --mode "$mode" up -d --build "$service"
