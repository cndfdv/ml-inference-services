"""Focused tests for model-local backend contracts without loading ML runtimes."""
import ast
import importlib

from conftest import SERVICES, load_service


def test_backend_modules_have_only_their_model_class_and_no_sibling_imports():
    expected = {
        "e5-small": "E5Backend",
        "user-bge-m3": "UserBGEM3Backend",
        "rapid-v5-mobile": "RapidOCRBackend",
        "whisper-large-v3": "WhisperBackend",
    }
    for model_id, backend_name in expected.items():
        module_path = SERVICES[model_id] / "backend.py"
        tree = ast.parse(module_path.read_text())
        classes = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}
        assert backend_name in classes
        other_model_classes = set(expected.values()) - {backend_name}
        assert not (classes & other_model_classes)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                assert not node.module.startswith(("embeddings.", "ocr.", "asr."))
            if isinstance(node, ast.Import):
                assert all(not alias.name.startswith(("embeddings.", "ocr.", "asr.")) for alias in node.names)
        lock = __import__("json").loads((module_path.parent / "model.lock.json").read_text())
        assert list(lock) == [model_id]


def test_embedding_batch_planner_preserves_order_and_respects_limits():
    service = load_service("e5-small")
    backend = importlib.import_module(service.__name__ + ".backend")
    assert backend.plan_batches([2, 5, 3, 1], 3, 10, 10) == [[0, 1], [2, 3]]


def test_whisper_item_validation_and_decode_limits():
    service = load_service("whisper-large-v3")
    whisper = importlib.import_module(service.__name__ + ".backend")
    decoder = lambda stream, sampling_rate: [0.0] * 16000
    waveform, language, task, duration = whisper.decode_whisper_item(
        {"audio": b"encoded", "language": "ru", "task": "transcribe"},
        max_bytes=10,
        max_duration_s=2,
        decoder=decoder,
    )
    assert (len(waveform), language, task, duration) == (16000, "ru", "transcribe", 1.0)
    import pytest
    with pytest.raises(ValueError, match="maximum byte size"):
        whisper.decode_whisper_item(b"x" * 11, max_bytes=10, max_duration_s=2, decoder=decoder)
