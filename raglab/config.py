"""Model-role configuration. Four roles, each pointing at a local or cloud endpoint."""
from __future__ import annotations
import json, os
from pathlib import Path

DATA_DIR = Path(os.environ.get("RAGLAB_DATA", "./data")).resolve()
CONFIG_PATH = Path(os.environ.get("RAGLAB_CONFIG", "./raglab.config.json")).resolve()

DEFAULT_MODELS = {
    "embedding": {
        "provider": "ollama", "base_url": "http://localhost:11434", "model": "nomic-embed-text",
        # nomic/e5/bge-style task prefixes. Leave empty for models that do not use them.
        "query_prefix": "search_query: ", "doc_prefix": "search_document: ", "batch_size": 32,
    },
    "reranker": {
        # providers: cross_encoder | cohere_compat | tei | llm | mock
        "provider": "llm", "via": "ollama", "base_url": "http://localhost:11434", "model": "llama3.1:8b",
    },
    "router": {"provider": "ollama", "base_url": "http://localhost:11434", "model": "llama3.1:8b"},
    "generator": {"provider": "ollama", "base_url": "http://localhost:11434", "model": "llama3.1:8b"},
}

MOCK_MODELS = {r: {"provider": "mock", "model": "mock"} for r in DEFAULT_MODELS}


def load_models() -> dict:
    if os.environ.get("RAGLAB_MOCK") == "1":
        return json.loads(json.dumps(MOCK_MODELS))
    if CONFIG_PATH.exists():
        cfg = json.loads(CONFIG_PATH.read_text())
        return {**DEFAULT_MODELS, **cfg.get("models", {})}
    return json.loads(json.dumps(DEFAULT_MODELS))


def save_models(models: dict) -> bool:
    """Write the role table to the config file, except in a RAGLAB_MOCK session.

    Mock mode ignores this file when it loads, so writing to it could only replace real settings
    with stand-ins that the next real start would then silently run on.

    :param models: The role table.
    :returns: Whether the file was written.
    """
    if os.environ.get("RAGLAB_MOCK") == "1":
        return False
    CONFIG_PATH.write_text(json.dumps({"models": models}, indent=2))
    return True


def api_key(role_cfg: dict) -> str | None:
    if role_cfg.get("api_key"):
        return role_cfg["api_key"]
    env = role_cfg.get("api_key_env")
    return os.environ.get(env) if env else None


def redact(models: dict) -> dict:
    out = json.loads(json.dumps(models))
    for cfg in out.values():
        if cfg.get("api_key"):
            cfg["api_key"] = "••••" + cfg["api_key"][-4:]
    return out
