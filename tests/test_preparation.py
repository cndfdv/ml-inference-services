"""Preparation lifecycle tests use synthetic bundles and never download weights."""
import hashlib
import importlib
import json

import pytest

from conftest import SERVICES, load_service

MODEL_IDS = ("e5-small", "user-bge-m3", "rapid-v5-mobile", "whisper-large-v3")


def write_bundle(service, output, model_id):
    entry = json.loads((SERVICES[model_id] / "model.lock.json").read_text())[model_id]
    root = output / model_id
    root.mkdir(parents=True)
    payload = b"synthetic model artifact"
    (root / "artifact.bin").write_bytes(payload)
    manifest = {
        "model_id": model_id,
        "revision": entry["revision"],
        "weights": entry["weights"],
        "contract_fingerprint": entry["contract_fingerprint"],
        "files": {"artifact.bin": hashlib.sha256(payload).hexdigest()},
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    return root


@pytest.mark.parametrize("model_id", MODEL_IDS)
def test_auto_start_reuses_valid_matching_bundle(model_id, tmp_path, monkeypatch):
    service = load_service(model_id)
    root = write_bundle(service, tmp_path, model_id)
    monkeypatch.setenv("PREPARE_MODE", "auto")
    prepare = importlib.import_module(service.__name__ + ".prepare")
    monkeypatch.setattr(prepare, "prepare", lambda *args, **kwargs: pytest.fail("cached bundle should be reused"))
    settings = service.settings.Settings(model_dir=str(tmp_path), device="cpu")
    prepare = importlib.import_module(service.__name__ + ".prepare")
    assert prepare.ensure_prepared(settings) == root


@pytest.mark.parametrize("model_id", MODEL_IDS)
def test_offline_mode_fails_when_bundle_is_missing(model_id, tmp_path, monkeypatch):
    service = load_service(model_id)
    monkeypatch.setenv("PREPARE_MODE", "offline")
    settings = service.settings.Settings(model_dir=str(tmp_path), device="cpu")
    prepare = importlib.import_module(service.__name__ + ".prepare")
    with pytest.raises(RuntimeError, match="missing"):
        prepare.ensure_prepared(settings)


@pytest.mark.parametrize("model_id", MODEL_IDS)
def test_corrupt_existing_bundle_fails_without_replacement(model_id, tmp_path, monkeypatch):
    service = load_service(model_id)
    root = write_bundle(service, tmp_path, model_id)
    (root / "artifact.bin").write_bytes(b"corrupt")
    monkeypatch.setenv("PREPARE_MODE", "auto")
    prepare = importlib.import_module(service.__name__ + ".prepare")
    monkeypatch.setattr(prepare, "prepare", lambda *args, **kwargs: pytest.fail("corrupt bundle must not be replaced"))
    settings = service.settings.Settings(model_dir=str(tmp_path), device="cpu")
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        prepare.ensure_prepared(settings)


@pytest.mark.parametrize("model_id", MODEL_IDS)
def test_auto_mode_prepares_a_missing_bundle_then_validates_it(model_id, tmp_path, monkeypatch):
    service = load_service(model_id)
    prepare = importlib.import_module(service.__name__ + ".prepare")
    monkeypatch.setenv("PREPARE_MODE", "auto")
    calls = []

    def fake_prepare(output, *args, **kwargs):
        calls.append(output)
        return write_bundle(service, output, model_id)

    monkeypatch.setattr(prepare, "prepare", fake_prepare)
    settings = service.settings.Settings(model_dir=str(tmp_path), device="cpu")
    assert prepare.ensure_prepared(settings) == tmp_path / model_id
    assert calls == [tmp_path]


def write_user_fp32_bundle(service, output):
    model_id = "user-bge-m3"
    entry = json.loads((SERVICES[model_id] / "model.lock.json").read_text())[model_id]
    root = output / model_id
    root.mkdir(parents=True)
    graph = b"synthetic fp32 graph"
    (root / "model-fp32.onnx").write_bytes(graph)
    manifest = {
        "model_id": model_id,
        "revision": entry["revision"],
        "weights": entry["weights"],
        "contract_fingerprint": entry["contract_fingerprint"],
        "files": {"model-fp32.onnx": hashlib.sha256(graph).hexdigest()},
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    return root


def test_user_int8_extends_cached_fp32_bundle_and_updates_manifest(tmp_path, monkeypatch):
    service = load_service("user-bge-m3")
    prepare = importlib.import_module(service.__name__ + ".prepare")
    root = write_user_fp32_bundle(service, tmp_path)
    monkeypatch.setenv("PREPARE_MODE", "auto")
    monkeypatch.setattr(prepare, "_quantize_int8", lambda source, dest: dest.write_bytes(b"int8 graph"))
    settings = service.settings.Settings(model_dir=str(tmp_path), device="cpu", user_variant="int8")

    assert prepare.ensure_prepared(settings) == root
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["files"]["model-int8.onnx"] == hashlib.sha256(b"int8 graph").hexdigest()
    assert (root / "model-fp32.onnx").read_bytes() == b"synthetic fp32 graph"


def test_user_int8_quantization_failure_preserves_cached_bundle(tmp_path, monkeypatch):
    service = load_service("user-bge-m3")
    prepare = importlib.import_module(service.__name__ + ".prepare")
    root = write_user_fp32_bundle(service, tmp_path)
    original_manifest = (root / "manifest.json").read_bytes()
    monkeypatch.setenv("PREPARE_MODE", "auto")

    def fail_quantization(source, dest):
        raise RuntimeError("mock quantizer failure")

    monkeypatch.setattr(prepare, "_quantize_int8", fail_quantization)
    settings = service.settings.Settings(model_dir=str(tmp_path), device="cpu", user_variant="int8")
    with pytest.raises(RuntimeError, match="mock quantizer failure"):
        prepare.ensure_prepared(settings)
    assert (root / "manifest.json").read_bytes() == original_manifest
    assert (root / "model-fp32.onnx").read_bytes() == b"synthetic fp32 graph"
    assert not (root / "model-int8.onnx").exists()


def test_user_int8_offline_mode_requires_the_graph(tmp_path, monkeypatch):
    service = load_service("user-bge-m3")
    prepare = importlib.import_module(service.__name__ + ".prepare")
    root = write_user_fp32_bundle(service, tmp_path)
    monkeypatch.setenv("PREPARE_MODE", "offline")
    settings = service.settings.Settings(model_dir=str(tmp_path), device="cpu", user_variant="int8")
    with pytest.raises(RuntimeError, match="INT8 graph is missing"):
        prepare.ensure_prepared(settings)
    assert not (root / "model-int8.onnx").exists()


def test_user_int8_missing_bundle_requests_int8_during_fresh_prepare(tmp_path, monkeypatch):
    service = load_service("user-bge-m3")
    prepare = importlib.import_module(service.__name__ + ".prepare")
    monkeypatch.setenv("PREPARE_MODE", "auto")
    calls = []

    def fake_prepare(output, benchmark_cache=None, int8=False):
        calls.append(int8)
        root = write_user_fp32_bundle(service, output)
        (root / "model-int8.onnx").write_bytes(b"int8 graph")
        manifest = json.loads((root / "manifest.json").read_text())
        manifest["files"]["model-int8.onnx"] = hashlib.sha256(b"int8 graph").hexdigest()
        (root / "manifest.json").write_text(json.dumps(manifest))
        return root

    monkeypatch.setattr(prepare, "prepare", fake_prepare)
    settings = service.settings.Settings(model_dir=str(tmp_path), device="cpu", user_variant="int8")
    assert prepare.ensure_prepared(settings) == tmp_path / "user-bge-m3"
    assert calls == [True]


@pytest.mark.parametrize(
    ("model_id", "expected", "forbidden"),
    [
        ("e5-small", ("--output", "--benchmark-cache"), ("--ocr-cache", "--int8")),
        ("user-bge-m3", ("--output", "--benchmark-cache", "--int8"), ("--ocr-cache",)),
        ("rapid-v5-mobile", ("--output", "--ocr-cache"), ("--benchmark-cache", "--int8")),
        ("whisper-large-v3", ("--output", "--benchmark-cache"), ("--ocr-cache", "--int8")),
    ],
)
def test_preparation_cli_only_exposes_model_appropriate_options(
    model_id, expected, forbidden, monkeypatch, capsys
):
    import sys

    service = load_service(model_id)
    prepare = importlib.import_module(service.__name__ + ".prepare")
    monkeypatch.setattr(sys, "argv", ["prepare", "--help"])
    with pytest.raises(SystemExit) as error:
        prepare.main()
    assert error.value.code == 0
    help_text = capsys.readouterr().out
    assert all(option in help_text for option in expected)
    assert all(option not in help_text for option in forbidden)
