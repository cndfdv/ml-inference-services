"""Fixtures and model package loader for root-level service tests."""

import importlib
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SERVICES = {
    "whisper-large-v3": ROOT / "asr" / "whisper",
    "e5-small": ROOT / "embeddings" / "e5-small",
    "user-bge-m3": ROOT / "embeddings" / "user-bge-m3",
    "rapid-v5-mobile": ROOT / "ocr" / "rapid-v5-mobile",
}


def load_service(model_id):
    """Load a service with a stable package name despite hyphens in its directory."""
    directory = SERVICES[model_id]
    package_name = "isolated_service_" + model_id.replace("-", "_")
    if package_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            package_name,
            directory / "__init__.py",
            submodule_search_locations=[str(directory)],
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[package_name] = module
        spec.loader.exec_module(module)
    package = importlib.import_module(package_name)
    package.settings = importlib.import_module(package_name + ".settings")
    package.app = importlib.import_module(package_name + ".app")
    package.batching = importlib.import_module(package_name + ".batching")
    return package


@pytest.fixture
def service_loader():
    return load_service
