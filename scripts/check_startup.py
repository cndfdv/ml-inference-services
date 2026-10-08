#!/usr/bin/env python3
"""Exercise an image's real entrypoint, CPU inference, workers and cached restart.

Mount this directory at /checks and run in the model image with docker run --rm.
The test owns and gracefully shuts down its child HTTP processes.
"""
import argparse
import base64
import concurrent.futures
import io
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from smoke import request

BASE = "http://127.0.0.1:8000"


def exercise(model, audio):
    if model in {"gigaam", "easyocr"}:
        result = request(BASE, f"/{model}/health")
        assert result["status"] == "ok" and result["loaded"] is False, result
        return {"lazy_health": result}
    info = request(BASE, "/v1/models")["data"][0]
    assert info["id"] == model and info["metadata"]["device"] == "cpu", info
    report = {"worker_pid": info["worker_pid"], "metadata": info["metadata"]}
    if model in {"e5-small", "user-bge-m3"}:
        import numpy as np
        result = request(BASE, f"/{model}/embed", {"texts": ["Проверка модели", "Архив договоров"], "role": "query"})
        vectors = np.asarray(result["embeddings"])
        assert vectors.shape == (2, 384 if model == "e5-small" else 1024)
        assert np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5)
        report["dimension"] = result["dim"]
    elif model == "rapid-v5-mobile":
        from PIL import Image
        buffer = io.BytesIO()
        Image.new("RGB", (320, 100), "white").save(buffer, format="PNG")
        result = request(BASE, "/rapid-v5-mobile/ocr/batch", {"images": [base64.b64encode(buffer.getvalue()).decode()] * 2})
        assert len(result["pages"]) == 2 and all(not p["text"] for p in result["pages"])
        report["pages"] = 2
    else:
        from check_devices import upload
        assert audio, "--audio required for Whisper"
        result = upload(BASE, "/whisper-large-v3/transcribe", audio.read_bytes(), "en")
        assert result["text"].strip(), result
        report["text"] = result["text"]
    return report


def run_once(args, mode):
    env = dict(os.environ, DEVICE="cpu", WORKERS=str(args.workers), PREPARE_MODE=mode)
    legacy = args.model in {"gigaam", "easyocr"}
    env[("GIGAAM" if args.model == "gigaam" else "EASYOCR") + "_WORKERS"] = str(args.workers)
    command = [sys.executable, "__main__.py"] if legacy else [sys.executable, "-m", "service"]
    process = subprocess.Popen(command, env=env, start_new_session=True, stdout=sys.stderr, stderr=sys.stderr)
    try:
        deadline = time.monotonic() + args.start_timeout
        route = f"/{args.model}/health" if legacy else "/readyz"
        while True:
            if process.poll() is not None:
                raise RuntimeError(f"entrypoint exited with {process.returncode}")
            try:
                request(BASE, route)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise TimeoutError("HTTP startup timed out")
                time.sleep(1)
        report = exercise(args.model, args.audio)
        if args.workers > 1 and not legacy:
            seen = set()
            for _ in range(8):
                with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
                    seen.update(pool.map(lambda _: request(BASE, "/v1/models")["data"][0]["worker_pid"], range(32)))
                if len(seen) == args.workers:
                    break
            assert len(seen) == args.workers, seen
            report["worker_pids"] = sorted(seen)
        return report
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--mode", choices=["auto", "offline"], default="offline")
    parser.add_argument("--cached-restart", action="store_true")
    parser.add_argument("--start-timeout", type=int, default=1200)
    parser.add_argument("--audio", type=Path)
    args = parser.parse_args()
    report = {"model": args.model, "first": run_once(args, args.mode)}
    if args.cached_restart:
        report["cached"] = run_once(args, "offline")
        assert report["first"].get("metadata", {}).get("weight_files") == report["cached"].get("metadata", {}).get("weight_files")
    print(json.dumps({"status": "passed", **report}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
