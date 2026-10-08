#!/usr/bin/env python3
"""Run a real CPU backend in a disposable container against its live GPU API."""

import argparse
import base64
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from smoke import request


def upload(base, route, audio, language):
    boundary = "model-inference-parity-fixture"
    body = b""
    for name, value in (("language", language), ("task", "transcribe")):
        body += f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode()
    body += f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="fixture.wav"\r\nContent-Type: audio/wav\r\n\r\n'.encode()
    body += audio + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(
        base + route,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urllib.request.urlopen(req, timeout=600) as response:
        return json.load(response)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--gpu-base", required=True)
    p.add_argument("--model-dir", default="/models")
    p.add_argument("--image", type=Path)
    p.add_argument("--audio", type=Path)
    p.add_argument("--language", default="ru")
    a = p.parse_args()
    sys.path.insert(0, str(Path.cwd()))
    from service.backend import create_backend
    from service.settings import Settings

    config = Settings(model_id=a.model, model_dir=a.model_dir, device="cpu")
    config.validate()
    backend = create_backend(config)
    report = {
        "model": a.model,
        "cpu_metadata": backend.metadata(),
        "gpu_metadata": request(a.gpu_base, "/v1/models")["data"][0]["metadata"],
    }
    assert report["cpu_metadata"]["device"] == "cpu"
    assert report["gpu_metadata"]["device"] == "cuda"
    for field in (
        "checkpoint_revision",
        "weight_files",
        "contract_fingerprint",
        "preprocessing",
        "precision",
    ):
        assert report["cpu_metadata"][field] == report["gpu_metadata"][field], field
    if a.model in {"e5-small", "user-bge-m3"}:
        import numpy as np

        texts = [
            "Проверка поиска документов",
            "Срок хранения договоров составляет семь лет.",
            "Номер заявки 884.",
        ]
        report["roles"] = {}
        for role in ("query", "passage"):
            cpu = np.asarray(backend.infer([{"text": t, "role": role} for t in texts]))
            gpu = np.asarray(
                request(a.gpu_base, f"/{a.model}/embed", {"texts": texts, "role": role})[
                    "embeddings"
                ]
            )
            cosine = np.sum(cpu * gpu, axis=1) / (
                np.linalg.norm(cpu, axis=1) * np.linalg.norm(gpu, axis=1)
            )
            assert cosine.min() >= 0.99999, cosine
            report["roles"][role] = {
                "min_cosine": float(cosine.min()),
                "max_abs_difference": float(np.abs(cpu - gpu).max()),
            }
    elif a.model == "rapid-v5-mobile":
        assert a.image, "--image required"
        data = a.image.read_bytes()
        cpu = backend.infer([data])[0]
        gpu = request(
            a.gpu_base, "/rapid-v5-mobile/ocr/batch", {"images": [base64.b64encode(data).decode()]}
        )["pages"][0]
        assert cpu["text"] == gpu["text"], "CPU/GPU OCR text differs on fixture"
        report["identical_text"] = True
        report["lines"] = len(cpu["lines"])
    else:
        assert a.audio, "--audio required"
        data = a.audio.read_bytes()
        import io

        import numpy as np
        from faster_whisper.audio import decode_audio

        reference_audio = decode_audio(io.BytesIO(data), sampling_rate=16000)
        bounded_audio = backend.decoder(io.BytesIO(data), sampling_rate=16000)
        assert np.array_equal(reference_audio, bounded_audio), "bounded decoder changed waveform"
        report["decoder_matches_upstream"] = True
        cpu = backend.infer([{"audio": data, "language": a.language, "task": "transcribe"}])[0]
        gpu = upload(a.gpu_base, "/whisper-large-v3/transcribe", data, a.language)

        def words(text):
            import re

            return re.findall(r"\w+", text.casefold())

        reference, hypothesis = words(cpu["text"]), words(gpu["text"])
        assert reference, "Whisper fixture produced no speech"
        previous = list(range(len(hypothesis) + 1))
        for i, token in enumerate(reference, 1):
            row = [i]
            for j, other in enumerate(hypothesis, 1):
                row.append(min(row[-1] + 1, previous[j] + 1, previous[j - 1] + (token != other)))
            previous = row
        wer = previous[-1] / len(reference)
        assert wer <= 0.02, f"CPU/GPU Whisper WER {wer:.3f} exceeds fixture gate"
        report["identical_text"] = cpu["text"].strip() == gpu["text"].strip()
        report["normalized_word_error_rate"] = wer
        report["text"] = cpu["text"]
        report["duration"] = cpu["duration"]
    report["status"] = "passed"
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
