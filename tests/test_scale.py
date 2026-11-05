"""What happens between the sample corpora and a real project.

The sample corpora are 8 and 25 documents. Every cost here is fine at that size and some of them
were quadratic, so this module works at a scale where the difference shows: it asserts on the shape
of the growth, not on wall-clock numbers, which vary too much between machines to pin.
"""
import asyncio
import os
import tempfile

os.environ.setdefault("RAGLAB_MOCK", "1")
os.environ.setdefault("RAGLAB_DATA", tempfile.mkdtemp())

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from raglab import indexer  # noqa: E402
from raglab.graph import CO_OCCURRENCE_LIMIT, GraphIndex  # noqa: E402
from raglab.main import MODELS, app, store  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

client = TestClient(app)

WORDS = ("resolver redis parcel scan depot notifier archive claim courier manifest cache outage "
         "runbook incident retention policy warehouse dispatch tracking exception").split()


def _synthetic(docs: int, paragraphs: int = 6) -> list[str]:
    """A corpus large enough for the growth to show, deterministic so runs are comparable."""
    rng = np.random.default_rng(7)
    ids = []
    for d in range(docs):
        body = "\n\n".join(" ".join(rng.choice(WORDS, size=40)) + f" section {d}.{p}." for p in range(paragraphs))
        ids.append(store.add_document(f"Scale doc {d}", body, {"corpus": "scale", "doc_key": f"SCALE-{d:03d}"}))
    return ids


@pytest.fixture(scope="module")
def big_index(request):
    ids = _synthetic(60)
    iid = store.create_index("scale", indexer.merged_config({"doc_ids": ids, "chunking": {"size": 120, "overlap": 20}}))
    asyncio.run(indexer.build(store, iid, MODELS))
    assert store.index(iid)["status"] == "ready", store.index(iid)["stats"]

    def cleanup():
        for doc in store.documents():
            if doc["metadata"].get("corpus") == "scale":
                store.x("DELETE FROM documents WHERE id=?", (doc["id"],))
    request.addfinalizer(cleanup)
    return iid


# --------------------------------------------------------------------- graph memory
def test_the_entity_graph_does_not_allocate_a_dense_matrix_per_entity_pair():
    """Two dense n-by-n float32 matrices is 8 bytes per pair: ~1.6 GB at 10k entities, rebuilt on
    every index load, for a matrix that is nearly all zeros."""
    n = 4000
    names = [f"entity-{i}" for i in range(n)]
    nodes = {name: {"label": name, "type": "", "chunks": [i % 400]} for i, name in enumerate(names)}
    edges = [{"s": names[i], "t": names[(i * 7 + 3) % n], "r": "rel", "chunk": i % 400} for i in range(n)]
    graph = GraphIndex({"nodes": nodes, "edges": edges}, None)

    stored = graph._vals.nbytes + graph._rows.nbytes + graph._cols.nbytes
    dense = 2 * n * n * 4  # adj and trans, as it was
    assert stored < dense / 50, f"{stored} bytes stored against {dense} for the dense form"
    assert graph.edge_count < n * n / 100, "the graph is sparse and is stored that way"


def test_a_walk_over_a_large_graph_still_ranks_its_seed_highest():
    n = 4000
    names = [f"entity-{i}" for i in range(n)]
    nodes = {name: {"label": name, "type": "", "chunks": [i % 400]} for i, name in enumerate(names)}
    edges = [{"s": names[i], "t": names[(i * 7 + 3) % n], "r": "rel", "chunk": i % 400} for i in range(n)]
    graph = GraphIndex({"nodes": nodes, "edges": edges}, None)
    hits, sub = graph.search([{"mention": "e", "node": names[0], "score": 1.0, "how": "exact"}], k=5, alpha=0.25)
    assert hits and len(hits) <= 5
    assert sub["nodes"] and sub["nodes"][0]["seed"] is True


def test_a_passage_naming_many_entities_does_not_assert_a_relation_between_all_of_them():
    """Co-occurrence pairs grow with the square of a chunk's entities. One wide passage used to
    contribute more edges than the whole rest of the corpus."""
    wide = [f"w{i}" for i in range(CO_OCCURRENCE_LIMIT + 20)]
    nodes = {name: {"label": name, "type": "", "chunks": [0]} for name in wide}
    assert GraphIndex({"nodes": nodes, "edges": []}, None).edge_count == 0
    narrow = wide[:CO_OCCURRENCE_LIMIT]
    nodes = {name: {"label": name, "type": "", "chunks": [0]} for name in narrow}
    graph = GraphIndex({"nodes": nodes, "edges": []}, None)
    assert graph.edge_count == CO_OCCURRENCE_LIMIT * (CO_OCCURRENCE_LIMIT - 1), "narrow passages still connect"


# --------------------------------------------------------------------- runtime cache
@pytest.fixture(scope="module")
def several_indexes(request):
    """More small indexes than the cache holds, built here so the test does not depend on whatever
    another module happened to leave behind."""
    ids = _synthetic(2, paragraphs=2)
    built = []
    for n in range(indexer.RUNTIME_CACHE + 2):
        iid = store.create_index(f"cache-{n}", indexer.merged_config({"doc_ids": ids, "chunking": {"size": 200}}))
        asyncio.run(indexer.build(store, iid, MODELS))
        built.append(iid)
    request.addfinalizer(lambda: [store.delete_index(i) for i in built])
    return built


def test_loaded_indexes_do_not_accumulate_without_limit(several_indexes):
    """A sweep across ten indexes loaded and kept all ten, each with its matrix, BM25 and PCA fit."""
    indexer._RUNTIME.clear()
    for iid in several_indexes:
        indexer.runtime(store, iid)
    assert len(indexer._RUNTIME) == indexer.RUNTIME_CACHE
    assert several_indexes[-1] in indexer._RUNTIME, "the most recent is kept"
    assert several_indexes[0] not in indexer._RUNTIME, "the least recently used is dropped"


def test_reusing_an_index_keeps_it_from_being_evicted(several_indexes):
    indexer._RUNTIME.clear()
    first = indexer.runtime(store, several_indexes[0])
    for iid in several_indexes[1:indexer.RUNTIME_CACHE]:
        indexer.runtime(store, iid)
    indexer.runtime(store, several_indexes[0])                  # touch it again
    indexer.runtime(store, several_indexes[indexer.RUNTIME_CACHE])  # force an eviction
    assert several_indexes[0] in indexer._RUNTIME
    assert indexer.runtime(store, several_indexes[0]) is first, "and it is the same object, not a reload"
    assert several_indexes[1] not in indexer._RUNTIME, "the one not touched went instead"


# --------------------------------------------------------------------- inspection completeness
def test_every_chunk_is_reachable_even_past_the_detail_page(big_index, monkeypatch):
    """An exact-search probe ranges over the whole matrix, so inspection has to reach that far too."""
    monkeypatch.setattr("raglab.main.DETAIL_LIMIT", 10)
    detail = client.get(f"/api/indexes/{big_index}").json()
    total = detail["stats"]["chunks"]
    assert len(detail["chunks"]) == 10 and detail["truncated"] == total - 10
    assert detail["detail_limit"] == 10

    beyond = total - 1
    fetched = client.get(f"/api/indexes/{big_index}/chunks?ids={beyond},0").json()
    assert {c["i"] for c in fetched["chunks"]} == {0, beyond}
    assert all(c["text"] for c in fetched["chunks"]) and fetched["missing"] == []


def test_chunks_can_be_paged_through_in_order(big_index):
    first = client.get(f"/api/indexes/{big_index}/chunks?offset=0&limit=25").json()
    second = client.get(f"/api/indexes/{big_index}/chunks?offset=25&limit=25").json()
    assert [c["i"] for c in first["chunks"]] == list(range(25))
    assert [c["i"] for c in second["chunks"]] == list(range(25, 50))
    assert first["total"] == second["total"] > 50
    assert client.get(f"/api/indexes/{big_index}/chunks?offset=999999").status_code == 416


def test_asking_for_a_chunk_that_is_not_there_says_so_rather_than_failing():
    iid = [i["id"] for i in client.get("/api/indexes").json() if i["status"] == "ready"][0]
    r = client.get(f"/api/indexes/{iid}/chunks?ids=0,99999999").json()
    assert [c["i"] for c in r["chunks"]] == [0] and r["missing"] == [99999999]


# --------------------------------------------------------------------- the corpus itself
def test_a_larger_corpus_builds_and_answers(big_index):
    rt = indexer.runtime(store, big_index)
    assert rt.stats["chunks"] > 200, f"only {rt.stats['chunks']} chunks: raise the fixture size"
    assert rt.revision["vectors"] and rt.revision["chunks"] == len(rt.chunks)
    with client.stream("POST", "/api/query", json={"index_id": big_index,
                                                   "query": "What does the resolver do during a redis outage?",
                                                   "config": {"generate": {"enabled": False}}}) as r:
        body = "".join(r.iter_text())
    assert '"type": "done"' in body, body[:400]


def test_a_skipped_wide_passage_is_reported_rather_than_silently_dropped():
    """A graph smaller than the corpus suggests has to be explainable, not guessed at."""
    wide = [f"w{i}" for i in range(CO_OCCURRENCE_LIMIT + 1)]
    nodes = {name: {"label": name, "type": "", "chunks": [0]} for name in wide}
    nodes["anchor"] = {"label": "anchor", "type": "", "chunks": [1]}
    edges = [{"s": wide[0], "t": "anchor", "r": "rel", "chunk": 1}]
    graph = GraphIndex({"nodes": nodes, "edges": edges}, None)
    assert graph.wide_chunks == [0]
    _, sub = graph.search([{"mention": "w0", "node": wide[0], "score": 1.0, "how": "exact"}], k=5)
    assert sub["skipped"]["chunks"] == 1 and sub["skipped"]["limit"] == CO_OCCURRENCE_LIMIT
    assert "still in the graph" in sub["skipped"]["note"], "and says the explicit relations survived"
    assert sub["entities"] == len(nodes) and sub["graph_edges"] == graph.edge_count


def test_nothing_is_reported_when_no_passage_was_too_wide():
    nodes = {n: {"label": n, "type": "", "chunks": [0]} for n in ("a", "b", "c")}
    graph = GraphIndex({"nodes": nodes, "edges": []}, None)
    assert graph.wide_chunks == []
    _, sub = graph.search([{"mention": "a", "node": "a", "score": 1.0, "how": "exact"}], k=3)
    assert sub["skipped"] is None


def test_the_build_records_how_many_passages_were_too_wide(big_index):
    ids = [d["id"] for d in store.documents() if d["metadata"].get("corpus") == "scale"][:4]
    iid = store.create_index("scale-graph", indexer.merged_config({"doc_ids": ids, "graph": {"enabled": True}}))
    asyncio.run(indexer.build(store, iid, MODELS))
    stats = store.index(iid)["stats"]
    assert stats["graph_cooccurrence_limit"] == CO_OCCURRENCE_LIMIT
    assert stats["graph_wide_chunks"] >= 0, "recorded either way, so its absence is not ambiguous"
    store.delete_index(iid)
