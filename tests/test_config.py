"""Where model settings are written, and when they must not be."""
import json
import os
import tempfile

os.environ.setdefault("RAGLAB_MOCK", "1")
os.environ.setdefault("RAGLAB_DATA", tempfile.mkdtemp())

import pytest  # noqa: E402

from raglab import config  # noqa: E402
from raglab.main import app  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

client = TestClient(app)
REAL = {"generator": {"provider": "ollama", "model": "llama3.1:8b"}}


@pytest.fixture
def config_file(tmp_path, monkeypatch):
    path = tmp_path / "raglab.config.json"
    monkeypatch.setattr(config, "CONFIG_PATH", path)
    return path


def test_should_not_write_the_config_file_in_mock_mode(config_file, monkeypatch):
    monkeypatch.setenv("RAGLAB_MOCK", "1")
    assert config.save_models({"generator": {"provider": "mock", "model": "mock"}}) is False
    assert not config_file.exists()


def test_should_keep_real_settings_when_a_mock_session_saves(config_file, monkeypatch):
    """A mock session used to overwrite them, and the next real start then ran on stand-ins."""
    config_file.write_text(json.dumps({"models": REAL}))
    monkeypatch.setenv("RAGLAB_MOCK", "1")
    config.save_models({"generator": {"provider": "mock", "model": "mock"}})
    monkeypatch.delenv("RAGLAB_MOCK")
    assert config.load_models()["generator"]["provider"] == "ollama"


def test_should_write_the_config_file_outside_mock_mode(config_file, monkeypatch):
    monkeypatch.delenv("RAGLAB_MOCK", raising=False)
    assert config.save_models(REAL) is True
    assert json.loads(config_file.read_text()) == {"models": REAL}


def test_should_tell_the_caller_when_a_save_was_not_written(config_file):
    models = client.get("/api/models").json()["models"]
    response = client.put("/api/models", json={"models": models}).json()
    assert response["saved"] is False, "the Models page must not claim it saved to the file"
    assert not config_file.exists()
