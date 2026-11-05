"""End-to-end test on the mock provider: ingest -> build (every index feature on) -> query (every stage on) -> eval."""
import asyncio, os, tempfile
os.environ["RAGLAB_MOCK"] = "1"
os.environ["RAGLAB_DATA"] = tempfile.mkdtemp()

from fastapi.testclient import TestClient  # noqa: E402
from raglab import indexer  # noqa: E402
from raglab.main import app, store, MODELS  # noqa: E402
from raglab.pipeline import execute  # noqa: E402
from raglab.evals import generate_set, run_eval  # noqa: E402

client = TestClient(app)


def build(name, cfg):
    iid = store.create_index(name, indexer.merged_config(cfg))
    asyncio.run(indexer.build(store, iid, MODELS))
    assert store.index(iid)["status"] == "ready", store.index(iid)["stats"]
    return iid


def test_full_flow():
    assert len(client.post("/api/documents/sample").json()["ids"]) == 8
    iid = build("everything", {"chunking": {"strategy": "semantic", "parent_child": {"enabled": True, "parent_size": 400},
                                            "contextual": {"enabled": True}},
                               "graph": {"enabled": True, "communities": True}, "doc_metadata": {"enabled": True}})
    rt = indexer.runtime(store, iid)
    assert rt.graph is not None and rt.parents and rt.chunks[0]["context"]
    assert "doc_type" in rt.schema()

    cfg = {"self_query": {"enabled": True}, "hyde": {"enabled": True}, "graph": {"enabled": True}, "normalize": {"enabled": True},
           "decompose": {"enabled": True}, "step_back": {"enabled": True}, "crag": {"refine": True}, "communities": {"enabled": True},
           "verify": {"enabled": True}, "context": {"mmr": {"enabled": True}, "order": "edges"}}
    out = asyncio.run(execute(rt, MODELS, "What happens when Redis is unavailable for the Resolver?", cfg))
    names = [s["stage"] for s in out["stages"]]
    assert names == ["normalize", "route", "self_query", "transform", "retrieve", "fuse", "rerank", "crag", "context", "generate", "verify"], names
    assert out["result"]["tokens_in"] > 0 and rt.communities
    retrieve = out["stages"][4]["data"]
    assert {l["retriever"] for l in retrieve["lists"]} == {"bm25", "dense", "graph"}
    assert retrieve["hnsw_recall_vs_exact"] > 0.8
    top = rt.chunks[out["result"]["reranked"][0]]
    assert "Redis" in top["text"]
    assert out["result"]["answer"]

    chit = asyncio.run(execute(rt, MODELS, "hello", {}))
    assert [s["stage"] for s in chit["stages"]] == ["route", "generate"]

    # metadata filter narrows every retriever
    mask = rt.allowed([{"field": "doc_type", "op": "eq", "value": "runbook"}])
    hits, _, _ = rt.vectors.search(rt.matrix[0], 5, "hnsw", 50, mask)
    assert hits and all(rt.chunks[i]["meta"]["doc_type"] == "runbook" for i, _ in hits)

    sid = asyncio.run(generate_set(store, rt, MODELS, 5, "t"))
    es = [s for s in client.get("/api/evalsets").json() if s["id"] == sid][0]
    asyncio.run(run_eval(store, "run1", rt, MODELS, es, {"generate": {"enabled": False}}, "base", False))
    m = client.get(f"/api/evalruns?evalset_id={sid}").json()["runs"][0]["metrics"]
    assert m["stages"]["fused"]["hit"] > 0.5, m


def test_chunk_strategies_cover_text():
    doc = store.documents()[0]
    for strategy in ("fixed", "sentence", "recursive", "markdown", "semantic"):
        r = client.post("/api/chunk-preview", json={"doc_id": doc["id"], "chunking": {"strategy": strategy, "size": 120, "overlap": 20}})
        assert r.status_code == 200, r.text
        chunks = r.json()["chunks"]
        assert len(chunks) > 1 and all(c["end"] > c["start"] for c in chunks), strategy


def test_streaming_query():
    iid = build("plain", {})
    with client.stream("POST", "/api/query", json={"index_id": iid, "query": "How long are archived files kept?"}) as r:
        body = "".join(r.iter_text())
    assert '"type": "done"' in body and '"type": "token"' in body


def test_vector_space_endpoints():
    iid = [i for i in client.get("/api/indexes").json() if i["status"] == "ready"][0]["id"]
    d = client.get(f"/api/indexes/{iid}").json()
    assert d["docs"] and d["projection"][0]["doc_id"] and 0 < d["pca_variance"] <= 1
    near = client.post(f"/api/indexes/{iid}/probe", json={"i": 0, "k": 4}).json()
    assert len(near["neighbours"]) == 4 and all(n["i"] != 0 for n in near["neighbours"])
    free = client.post(f"/api/indexes/{iid}/probe", json={"text": "Redis cache outage", "k": 3}).json()
    assert len(free["xy"]) == 2 and len(free["neighbours"]) == 3


def test_no_chunk_is_pure_overlap():
    doc = store.documents()[0]
    for strategy in ("fixed", "sentence", "recursive", "markdown"):
        chunks = client.post("/api/chunk-preview", json={"doc_id": doc["id"], "chunking": {"strategy": strategy, "size": 90, "overlap": 40}}).json()["chunks"]
        for prev, cur in zip(chunks, chunks[1:]):
            assert cur["end"] > prev["end"], (strategy, prev, cur)


def test_halcyon_dedupe_freshness_normalisation_and_golden_set():
    """Halcyon corpus: near-duplicate removal, freshness filter, normalisation rules and the golden set."""
    assert len(client.post("/api/documents/sample?name=halcyon").json()["ids"]) == 25
    hal = [d["id"] for d in store.documents() if d["metadata"].get("corpus") == "halcyon"]
    iid = build("halcyon", {"chunking": {"strategy": "markdown", "size": 160, "overlap": 20}, "doc_ids": hal})
    rt = indexer.runtime(store, iid)
    assert rt.stats["deduped"] >= 1 and rt.superseded.any()

    out = asyncio.run(execute(rt, MODELS, "Can I send back a ZX-481 after three weeks?", {"normalize": {"enabled": True}}, store=store))
    stages = {s["stage"]: s["data"] for s in out["stages"]}
    rules = {s["rule"] for s in stages["normalize"]["steps"]}
    assert {"terminology", "date"} <= rules and "21 days" in stages["normalize"]["normalized"]
    assert stages["retrieve"]["filtered_out"], "superseded v2 policy should be filtered"
    assert not any(rt.superseded[i] for i in out["result"]["fused"])
    off = asyncio.run(execute(rt, MODELS, "Can I send back a ZX-481 after three weeks?", {"freshness": {"enabled": False}}, store=store))
    assert any(rt.superseded[i] for i in off["result"]["fused"])

    alias = client.post("/api/normalization/test", json={"index_id": iid, "query": "Is the Ridgeline batery covered?"}).json()
    assert "ZX-481" in alias["normalized"] and any(s["rule"] == "spelling" for s in alias["steps"])

    sid = client.post("/api/evalsets/golden?name=halcyon").json()["id"]
    es = [s for s in client.get("/api/evalsets").json() if s["id"] == sid][0]
    assert es["items"][0]["probes"]
    asyncio.run(run_eval(store, "g1", rt, MODELS, es, {"generate": {"enabled": False}}, "golden", False, 3))
    run = [r for r in client.get(f"/api/evalruns?evalset_id={sid}").json()["runs"] if r["id"] == "g1"][0]
    assert run["metrics"]["k"] == 3 and 0 < run["metrics"]["stages"]["context"]["recall"] <= 1
    assert run["rows"][0]["retrieved"] and run["rows"][0]["probes"]


def test_experiment_tooling():
    from raglab.evals import run_sweep, generate_set as gen
    iid = [i for i in client.get("/api/indexes").json() if i["name"] == "halcyon"][0]["id"]
    rt = indexer.runtime(store, iid)
    sid = asyncio.run(gen(store, rt, MODELS, 8, "hard", {"simple": .25, "paraphrased": .25, "multi": .25, "unanswerable": .25}))
    es = [s for s in client.get("/api/evalsets").json() if s["id"] == sid][0]
    assert {i["type"] for i in es["items"]} == {"simple", "paraphrased", "multi", "unanswerable"}
    asyncio.run(run_sweep(store, "sw1", MODELS, es, {"param": "rerank.top_n", "values": [1, 4, 8], "index_id": iid, "config": {}, "k": 5}))
    sw = client.get(f"/api/sweeps?evalset_id={sid}").json()["sweeps"][0]
    assert [p["value"] for p in sw["points"]] == [1, 4, 8] and sw["points"][0]["metrics"]["abstention"] is not None
    recall = [p["metrics"]["stages"]["context"]["recall"] for p in sw["points"]]
    assert recall[0] <= recall[-1]

    assert client.put("/api/presets/mine", json={"config": {"hyde": {"enabled": True}}}).status_code == 200
    assert client.get("/api/presets").json()[0]["name"] == "mine"

    # semantic cache + conversational rewriting
    from raglab.pipeline import Pipeline
    async def run(q, hist=None):
        return [e async for e in Pipeline(rt, MODELS, {"cache": {"enabled": True, "threshold": 0.9}}, store=store, history=hist).run(q)]
    first = asyncio.run(run("How do I open an RMA for a defective headlamp?"))
    assert first[0]["stage"] == "cache" and not first[0]["data"]["hit"]
    again = asyncio.run(run("How do I open an RMA for a defective headlamp?"))
    assert again[0]["data"]["hit"] and again[-1]["type"] == "done" and len(again) == 3
    follow = asyncio.run(run("Is it free?", [{"q": "How do I open an RMA?", "a": "Use the Support portal."}]))
    assert any(e.get("stage") == "condense" for e in follow)


def test_llm_cache_makes_rebuilds_free():
    iid = build("ctx-a", {"chunking": {"contextual": {"enabled": True}}})
    assert indexer.runtime(store, iid).stats["llm_cache_misses"] > 0
    iid2 = build("ctx-b", {"chunking": {"contextual": {"enabled": True}}})
    st = indexer.runtime(store, iid2).stats
    assert st["llm_cache_misses"] == 0 and st["llm_cache_hits"] > 0


def test_blank_name_is_generated_from_the_build_parameters():
    """POST /api/indexes with no name: the server names the index after what it is built from."""
    r = client.post("/api/indexes", json={"config": {"chunking": {"strategy": "fixed", "size": 200, "overlap": 30}}})
    assert r.status_code == 200, r.text
    first = r.json()
    assert first["name"] == "mock-fixed-200x30", first

    # Same parameters again: a suffix keeps the two apart instead of repeating the name.
    again = client.post("/api/indexes", json={"name": "  ", "config": {"chunking": {"strategy": "fixed", "size": 200, "overlap": 30}}}).json()
    assert again["name"] == "mock-fixed-200x30-2", again

    # An explicit name still wins.
    mine = client.post("/api/indexes", json={"name": "my index", "config": {}}).json()
    assert mine["name"] == "my index"

    listed = {i["id"]: i for i in client.get("/api/indexes").json()}
    for created in (first, again, mine):
        assert listed[created["id"]]["created"] > 0, "every index records when it was built"


def test_inspect_reports_when_and_how_an_index_was_built():
    iid = build("inspect-me", {"chunking": {"strategy": "fixed", "size": 180, "overlap": 25,
                                            "parent_child": {"enabled": True, "parent_size": 500}}})
    d = client.get(f"/api/indexes/{iid}").json()
    assert d["created"] > 0, "Inspect needs the build timestamp"
    # The parameters the UI reads back are the ones the index was built with.
    assert d["config"]["chunking"]["strategy"] == "fixed"
    assert d["config"]["chunking"]["size"] == 180 and d["config"]["chunking"]["overlap"] == 25
    assert d["config"]["chunking"]["parent_child"] == {"enabled": True, "parent_size": 500}
    assert d["config"]["hnsw"]["M"] and d["config"]["hnsw"]["ef_construction"]
    assert d["stats"]["embedding_model"] and d["stats"]["dim"] and d["stats"]["documents"]
    assert d["stats"]["build_s"] >= 0


def test_the_ui_route_serves_nothing_outside_the_static_folder():
    """Encoded ".." segments reached the UI route decoded, and it served any file the user could
    read: pyproject.toml here, but equally raglab.config.json with its API keys."""
    for url in ("/..%2f..%2fpyproject.toml", "/%2e%2e/%2e%2e/pyproject.toml", "/..%2F..%2Fraglab.config.example.json"):
        r = client.get(url)
        assert r.status_code == 200 and "<div id=\"root\">" in r.text, f"{url} must fall back to the app shell"
        assert "[project]" not in r.text and '"models"' not in r.text, f"{url} leaked a file from outside static/"


def test_the_ui_route_refuses_an_absolute_path():
    """Joining an absolute path onto static/ discards static/ altogether, so an encoded leading slash
    names any file on disk directly, with no ".." needed."""
    from pathlib import Path
    from urllib.parse import quote
    url = "/" + quote(str(Path("pyproject.toml").resolve()), safe="")
    r = client.get(url)
    assert r.status_code == 200 and "<div id=\"root\">" in r.text
    assert "[project]" not in r.text, "an absolute path leaked a file from outside static/"


def test_the_ui_route_still_serves_files_inside_the_static_folder():
    r = client.get("/index.html")
    assert r.status_code == 200 and "<div id=\"root\">" in r.text
    assert "<div id=\"root\">" in client.get("/workbench").text, "an unknown path falls back to the app shell"
