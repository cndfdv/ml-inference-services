import hashlib
import json
from types import SimpleNamespace

import pytest
from inference.backends import (
    _read_bounded_audio_frames,
    decode_whisper_item,
    plan_batches,
    validate_manifest,
)


def test_rapid_checks_eager_sessions_for_each_stage():
    from inference.backends import RapidOCRBackend

    def stage(providers):
        session = SimpleNamespace(get_providers=lambda: providers)
        return SimpleNamespace(session=SimpleNamespace(session=session))

    backend = RapidOCRBackend.__new__(RapidOCRBackend)
    backend.provider_by_stage = {"detector": [], "recognizer": []}
    backend.engine = SimpleNamespace(
        text_det=stage(["CUDAExecutionProvider", "CPUExecutionProvider"]),
        text_rec=stage(["CUDAExecutionProvider", "CPUExecutionProvider"]),
    )
    backend._configure_runtime("cuda")
    assert "CUDAExecutionProvider" in backend.provider_by_stage["detector"]
    backend.engine.text_rec = stage(["CPUExecutionProvider"])
    with pytest.raises(RuntimeError, match="recognizer missing CUDAExecutionProvider"):
        backend._configure_runtime("cuda")


def test_plan_batches_bounds_padded_tokens_and_preserves_order():
    batches = plan_batches([3, 8, 4, 2, 8], max_batch_size=4, max_batch_tokens=16, max_tokens=20)
    assert batches == [[0, 1], [2, 3], [4]]
    assert [index for batch in batches for index in batch] == list(range(5))
    assert all(len(batch) <= 4 for batch in batches)
    assert all(max([3, 8, 4, 2, 8][i] for i in batch) * len(batch) <= 16 for batch in batches)


def test_plan_batches_allows_one_long_item_but_rejects_model_limit():
    assert plan_batches([10, 2], 8, 4, 10) == [[0], [1]]
    with pytest.raises(ValueError, match="outside valid range"):
        plan_batches([11], 8, 4, 10)


def test_manifest_checks_pinned_identity_and_content_hash(tmp_path):
    root = tmp_path / "e5-small"
    root.mkdir()
    payload = root / "weights.bin"
    payload.write_bytes(b"pinned bytes")
    lock = {
        "e5-small": {
            "revision": "r1",
            "weights": "repo/name",
            "contract_fingerprint": "contract-v1",
        }
    }
    manifest = {
        "revision": "r1",
        "weights": "repo/name",
        "contract_fingerprint": "contract-v1",
        "files": {"weights.bin": hashlib.sha256(b"pinned bytes").hexdigest()},
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    assert validate_manifest(tmp_path, "e5-small", lock) == manifest
    payload.write_bytes(b"modified")
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        validate_manifest(tmp_path, "e5-small", lock)


def test_manifest_rejects_path_escape(tmp_path):
    root = tmp_path / "user-bge-m3"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.write_bytes(b"outside")
    lock = {
        "user-bge-m3": {
            "revision": "r2",
            "weights": "repo/name",
            "contract_fingerprint": "contract-v2",
        }
    }
    manifest = {
        "revision": "r2",
        "weights": "repo/name",
        "contract_fingerprint": "contract-v2",
        "files": {"../outside": hashlib.sha256(b"outside").hexdigest()},
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="invalid or missing"):
        validate_manifest(tmp_path, "user-bge-m3", lock)


def test_ocr_result_format_handles_numpy_boxes_and_blank_result():
    import numpy as np
    from inference.backends import RapidOCRBackend

    result = RapidOCRBackend._format(
        SimpleNamespace(
            boxes=np.asarray([[[1, 2], [8, 2], [8, 6], [1, 6]]]),
            txts=("тест",),
            scores=(0.93,),
        ),
        20,
        10,
    )
    assert result == {
        "text": "тест",
        "lines": [
            {
                "text": "тест",
                "bbox": [1.0, 2.0, 8.0, 6.0],
                "polygon": [[1.0, 2.0], [8.0, 2.0], [8.0, 6.0], [1.0, 6.0]],
                "confidence": 0.93,
            }
        ],
        "width": 20,
        "height": 10,
    }
    assert RapidOCRBackend._format(SimpleNamespace(boxes=None, txts=None, scores=None), 20, 10) == {
        "text": "",
        "lines": [],
        "width": 20,
        "height": 10,
    }


def test_whisper_audio_is_decoded_at_16khz_and_defaults_to_russian_transcription():
    seen = {}

    def decoder(stream, sampling_rate):
        seen["sampling_rate"] = sampling_rate
        seen["audio"] = stream.read()
        return [0.0] * 32000

    waveform, language, task, duration = decode_whisper_item(
        {"audio": b"encoded", "language": "ru"},
        max_bytes=100,
        max_duration_s=3,
        decoder=decoder,
    )
    assert waveform == [0.0] * 32000
    assert (language, task, duration) == ("ru", "transcribe", 2.0)
    assert seen == {"sampling_rate": 16000, "audio": b"encoded"}


def test_whisper_rejects_bad_audio_duration_language_and_task():
    def decoder(stream, sampling_rate):
        return [0.0] * 32000

    with pytest.raises(ValueError, match="byte size"):
        decode_whisper_item({"audio": b"123"}, max_bytes=2, max_duration_s=10, decoder=decoder)
    with pytest.raises(ValueError, match="duration"):
        decode_whisper_item({"audio": b"x"}, max_bytes=2, max_duration_s=1, decoder=decoder)
    with pytest.raises(ValueError, match="language"):
        decode_whisper_item(
            {"audio": b"x", "language": "RU"}, max_bytes=2, max_duration_s=3, decoder=decoder
        )
    with pytest.raises(ValueError, match="task"):
        decode_whisper_item(
            {"audio": b"x", "task": "summarize"}, max_bytes=2, max_duration_s=3, decoder=decoder
        )


def test_whisper_duration_cap_rejects_frame_before_converting_it():
    converted = []

    class Frame:
        def __init__(self, samples, content):
            self.samples = samples
            self.content = content

        def to_ndarray(self):
            converted.append(self.content)
            return ArrayData(self.content)

    class ArrayData:
        dtype = "int16"

        def __init__(self, content):
            self.content = content

        def tobytes(self):
            return self.content

    def frames():
        yield Frame(2, b"ok")
        yield Frame(3, b"too-long")
        raise AssertionError("decoder consumed frames after the duration cap")

    with pytest.raises(ValueError, match="duration"):
        _read_bounded_audio_frames(frames(), max_samples=4)
    assert converted == [b"ok"]
