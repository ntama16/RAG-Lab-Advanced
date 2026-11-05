"""Shared fixtures.

Every test module in this suite talks to one process-wide `Store`, because `raglab.main` builds it
at import time. That makes any row a test writes visible to every module that runs after it, and
pytest's module order is alphabetical rather than meaningful. Anything a test changes outside its
own rows therefore has to be put back, or a later module quietly measures something else — which has
already caught the normalisation rules twice.
"""
import os
import tempfile

import pytest

os.environ.setdefault("RAGLAB_MOCK", "1")
os.environ.setdefault("RAGLAB_DATA", tempfile.mkdtemp())


@pytest.fixture
def restore_rules():
    """Put the normalisation rules back, whichever keys the test wrote.

    Rules can be stored shared (`normalization`) or scoped to one index
    (`normalization:<index id>`), so this snapshots every such row rather than just the shared one.
    """
    from raglab.main import store
    before = {r["key"]: r["value"] for r in store.q("SELECT key, value FROM kv WHERE key LIKE 'normalization%'")}
    yield
    after = {r["key"] for r in store.q("SELECT key FROM kv WHERE key LIKE 'normalization%'")}
    for key in after - set(before):
        store.x("DELETE FROM kv WHERE key=?", (key,))
    for key, value in before.items():
        store.x("INSERT OR REPLACE INTO kv VALUES (?, ?)", (key, value))


@pytest.fixture
def own_corpus():
    """Remove documents the test added, so a module that inspects the corpus as a whole is not
    affected by which file pytest happened to run first."""
    from raglab.main import store
    before = {d["id"] for d in store.documents()}
    yield
    for doc in store.documents():
        if doc["id"] not in before:
            store.x("DELETE FROM documents WHERE id=?", (doc["id"],))
