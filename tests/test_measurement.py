"""The measurement layer: metrics that cannot exceed their range, abstention that reflects what the
user was told, controls that do what their label says, and experiments that isolate one variable.

Every case here pins a defect the lab shipped with, so the numbers below are hand-calculated rather
than captured from a run.
"""
import asyncio
import json
import math
import os
import tempfile
import uuid

os.environ.setdefault("RAGLAB_MOCK", "1")
os.environ.setdefault("RAGLAB_DATA", tempfile.mkdtemp())

import pytest  # noqa: E402

from raglab import evals, indexer  # noqa: E402
from raglab.main import MODELS, app, store  # noqa: E402
from raglab.pipeline import answer_abstains, controlled, execute, inactive, merged, validate  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

client = TestClient(app)


@pytest.fixture(scope="module", autouse=True)
def _own_corpus():
    """These tests add their own documents to the shared store. Remove them afterwards, so a module
    that inspects the corpus as a whole cannot be affected by which file pytest ran first."""
    before = {d["id"] for d in store.documents()}
    yield
    for doc in store.documents():
        if doc["id"] not in before:
            store.x("DELETE FROM documents WHERE id=?", (doc["id"],))


class FakeRuntime:
    """Just enough of a Runtime for the scorers: chunks with text and metadata."""

    def __init__(self, texts, doc_ids=None, metas=None):
        self.chunks = [{"i": n, "text": t, "doc_id": (doc_ids or ["d"] * len(texts))[n],
                        "meta": (metas or [{}] * len(texts))[n]}
                       for n, t in enumerate(texts)]


# --------------------------------------------------------------------- nDCG bounds
def test_ndcg_stays_within_one_when_a_chunk_covers_two_quotes():
    """Two quotes in one top-ranked chunk used to score 1.2263: the DCG counted the position twice
    while the ideal spread the units over two positions."""
    rt = FakeRuntime(["alpha evidence and beta evidence together", "unrelated"])
    item = {"evidence": ["alpha evidence", "beta evidence"]}
    s = evals.score_list([0, 1], rt, item, k=10)
    assert s["ndcg"] == pytest.approx(1.0), s
    assert s["recall"] == pytest.approx(1.0)
    assert s["rank"] == 1


def test_ndcg_never_exceeds_one_for_any_shape():
    rt = FakeRuntime(["alpha evidence beta evidence gamma evidence", "alpha evidence", "beta evidence"])
    for item in ({"evidence": ["alpha evidence", "beta evidence", "gamma evidence"]},
                 {"evidence": ["alpha evidence", "beta evidence"]},
                 {"evidence": "alpha evidence"}):
        for ids in ([0, 1, 2], [2, 1, 0], [1, 2, 0], [0], []):
            s = evals.score_list(ids, rt, item, k=10)
            assert 0.0 <= s["ndcg"] <= 1.0, (item, ids, s)


def test_ndcg_matches_the_textbook_formula_for_a_single_unit():
    """The common case must not have moved: one quote at rank r scores 1 / log2(r + 1)."""
    rt = FakeRuntime(["nope", "nope either", "the evidence sentence"])
    s = evals.score_list([0, 1, 2], rt, {"evidence": "the evidence sentence"}, k=10)
    assert s["ndcg"] == pytest.approx(1 / math.log2(4))
    assert s["mrr"] == pytest.approx(1 / 3)


def test_ndcg_rewards_a_higher_placement():
    rt = FakeRuntime(["the evidence sentence", "filler one", "filler two"])
    item = {"evidence": "the evidence sentence"}
    assert evals.score_list([0, 1, 2], rt, item, 10)["ndcg"] > evals.score_list([1, 2, 0], rt, item, 10)["ndcg"]


def test_recall_counts_units_not_positions():
    rt = FakeRuntime(["alpha evidence and beta evidence", "gamma evidence"])
    s = evals.score_list([0], rt, {"evidence": ["alpha evidence", "beta evidence", "gamma evidence"]}, k=10)
    assert s["recall"] == pytest.approx(2 / 3)


def test_k_excludes_evidence_below_the_cutoff():
    rt = FakeRuntime(["filler", "filler", "the evidence sentence"])
    s = evals.score_list([0, 1, 2], rt, {"evidence": "the evidence sentence"}, k=2)
    assert s["hit"] == 0.0 and s["recall"] == 0.0 and s["ndcg"] == 0.0
    assert s["rank"] == 3, "rank reports where it was, even outside k"


# --------------------------------------------------------------------- delivered evidence
def test_delivered_scoring_reads_the_expanded_parent_not_the_child_chunk():
    """Parent expansion is the point of parent-child chunking; scoring the child hides what it won."""
    blocks = [{"i": 0, "text": "the full parent passage holds the evidence sentence", "expanded": True,
               "doc_id": "d", "title": "T"}]
    assert evals.score_blocks(blocks, {"evidence": "the evidence sentence"})["hit"] == 1.0


def test_delivered_scoring_misses_evidence_that_refinement_removed():
    """The child chunk still contains the quote, but the generator never saw it."""
    item = {"evidence": "the evidence sentence"}
    rt = FakeRuntime(["preamble. the evidence sentence. trailer."])
    assert evals.score_list([0], rt, item, k=10)["hit"] == 1.0, "candidate retrieval did find it"
    refined = [{"i": 0, "text": "preamble. trailer.", "refined": True, "doc_id": "d", "title": "T"}]
    assert evals.score_blocks(refined, item)["hit"] == 0.0, "but it was not delivered"


def test_delivered_scoring_counts_a_community_report():
    blocks = [{"i": -1, "text": "this report mentions the evidence sentence", "title": "Community report: X"}]
    assert evals.score_blocks(blocks, {"evidence": "the evidence sentence"})["hit"] == 1.0


def test_delivered_scoring_resolves_document_keys():
    blocks = [{"i": 0, "text": "anything at all", "doc_id": "d1", "doc_key": "POL-RET-v3", "title": "Returns v3"}]
    assert evals.score_blocks(blocks, {"relevant_docs": ["POL-RET-v3"]})["hit"] == 1.0
    assert evals.score_blocks(blocks, {"relevant_docs": ["POL-RET-v2"]})["hit"] == 0.0


# --------------------------------------------------------------------- abstention
@pytest.mark.parametrize("answer", [
    "The indexed corpus does not cover drone deliveries.",
    "These passages do not contain the answer.",
    "I don't know based on the provided passages.",
    "The retrieval grader found no supporting evidence for this.",
])
def test_answer_abstention_recognises_a_refusal(answer):
    assert answer_abstains(answer) is True


@pytest.mark.parametrize("answer", [
    "Electronics must be returned within 21 days [1].",
    "The restocking fee is 15% [2], waived for dead-on-arrival units [3].",
    "",
])
def test_answer_abstention_does_not_fire_on_a_real_answer(answer):
    assert answer_abstains(answer) is False


def test_answer_abstention_ignores_line_breaks():
    assert answer_abstains("The indexed corpus\n  does not\ncover it.") is True


def test_answer_abstention_does_not_fire_on_the_same_words_inside_a_real_answer():
    """"does not contain" mid-answer is part of an answer, not a refusal."""
    answered = ("Electronics must be returned within 21 days [1]. The policy does not contain an exemption "
                "for opened packaging, so the restocking fee still applies [2].")
    assert answer_abstains(answered) is False


def test_answer_abstention_still_fires_when_the_refusal_leads():
    assert answer_abstains("The indexed corpus does not cover drone deliveries. Nothing in the policies, "
                           "runbooks or manuals mentions aerial despatch at all.") is True


def test_answer_abstention_fires_on_a_refusal_that_cites_the_passages_it_read():
    assert answer_abstains("The passages [1][2] do not contain the answer.") is True


def test_answer_abstention_has_a_documented_miss_when_the_refusal_does_not_lead():
    """Pinned, not endorsed: only the first sentence is matched. The grader's verdict is reported
    beside this one, so the disagreement is visible instead of silently resolved."""
    assert answer_abstains("I checked the passages. The corpus does not cover this.") is False


def test_an_unmeasured_abstention_signal_is_reported_as_unmeasured_not_as_zero():
    """With the grader off there is no grader verdict. Reporting that as 0% would draw "nothing
    abstained" and "nobody was asked" as the same bar, which is the defect this pass removes."""
    sid, iid = _abstention_set(), _ready_index()
    es = [s for s in client.get("/api/evalsets").json() if s["id"] == sid][0]
    asyncio.run(evals.run_eval(store, "ab-off", indexer.runtime(store, iid), MODELS, es,
                               {"crag": {"enabled": False}, "generate": {"enabled": True}}, "crag off", False, 5))
    m = [r for r in client.get(f"/api/evalruns?evalset_id={sid}").json()["runs"] if r["id"] == "ab-off"][0]["metrics"]
    assert m["abstention_by"]["crag"] is None, "the grader produced no verdict at all"
    assert m["measured"]["grader"] is False and m["measured"]["answers"] is True
    assert m["abstention_by"]["answer"] is not None, "but the answer did decide something"


def test_a_genuine_zero_is_still_reported_as_zero():
    sid, iid = _abstention_set(), _ready_index()
    es = [s for s in client.get("/api/evalsets").json() if s["id"] == sid][0]
    asyncio.run(evals.run_eval(store, "ab-on", indexer.runtime(store, iid), MODELS, es,
                               {"crag": {"enabled": True}, "generate": {"enabled": True}}, "crag on", False, 5))
    m = [r for r in client.get(f"/api/evalruns?evalset_id={sid}").json()["runs"] if r["id"] == "ab-on"][0]["metrics"]
    assert m["measured"]["grader"] is True
    assert m["abstention_by"]["crag"] is not None, "a measured 0.0 is a number, not an absence"
    assert m["false_abstention_by"]["crag"] == 0.0


def _abstention_set():
    items = [{"question": "What is the policy on interplanetary drone deliveries?", "evidence": "",
              "type": "unanswerable"},
             {"question": "What does the Resolver do when Redis is unavailable?",
              "evidence": "falls back to the slow path", "type": "simple"}]
    return client.post("/api/evalsets", json={"name": "abstention", "items": items}).json()["id"]


# --------------------------------------------------------------------- config validation
def test_validate_rejects_values_that_crash_or_measure_nothing():
    bad = merged({"dense": {"k": -5}, "fusion": {"top_n": 0}, "bm25": {"b": 3.0},
                  "context": {"max_chunks": 0}, "cache": {"threshold": 2.0}})
    fields = {e["field"] for e in validate(bad)}
    assert {"dense.k", "fusion.top_n", "bm25.b", "context.max_chunks", "cache.threshold"} <= fields


def test_validate_rejects_a_fractional_integer_and_a_bad_choice():
    errors = validate(merged({"dense": {"k": 3.5, "mode": "magic"}}))
    assert {e["field"] for e in errors} == {"dense.k", "dense.mode"}


def test_validate_accepts_the_defaults():
    assert validate(merged({})) == []


def test_query_rejects_an_out_of_range_setting_with_a_field_level_error():
    iid = _ready_index()
    r = client.post("/api/query", json={"index_id": iid, "query": "anything", "config": {"dense": {"k": -5}}})
    assert r.status_code == 422, r.text
    detail = r.json()["detail"]
    assert any(e["field"] == "dense.k" for e in detail["errors"]), detail


def test_pipeline_check_reports_inactive_controls_without_running_anything():
    r = client.post("/api/pipeline/check", json={"config": {"rerank": {"enabled": False}}})
    assert r.status_code == 200
    assert any("rerank.top_n is ignored" in w for w in r.json()["warnings"])


def test_inactive_warns_when_ef_search_is_below_dense_k():
    notes = " ".join(inactive(merged({"dense": {"k": 30, "ef_search": 8}})))
    assert "runs at ef 30" in notes


def test_estimated_call_count_shows_what_a_first_run_costs():
    """The default pipeline asks for ~30 rerank requests before it answers anything."""
    default = client.post("/api/pipeline/check", json={"config": {}}).json()["calls"]
    assert default["reranker"] == 30, default
    fast = [p for p in client.get("/api/defaults").json()["presets"] if p["name"] == "Fast hybrid baseline"][0]
    cheap = client.post("/api/pipeline/check", json={"config": fast["config"]}).json()["calls"]
    assert cheap["reranker"] == 0 and cheap["total"] < default["total"] / 10, cheap


# --------------------------------------------------------------------- controls honouring labels
# These tests own their corpus rather than loading a sample, so they cannot disturb a suite that
# asserts how many documents the sample endpoint adds.
FIXTURE_DOCS = [
    ("Resolver runbook", """The Resolver depends on Redis for address lookups. When Redis is unavailable the
     Resolver falls back to the slow path and queries Postgres directly, which raises p99 latency to about four
     seconds per parcel. Operators should warm the cache with `resolver-cli cache warm --all` once Redis returns,
     which takes roughly eight minutes for a full estate. If Redis has not recovered after fifteen minutes,
     page the platform on-call and open an incident. The Notifier is unaffected because it reads from its own
     queue, but it will emit duplicate scan notifications while the Resolver retries."""),
    ("Notifier specification", """The Notifier consumes scan events from the parcel queue and sends customer
     notifications. It deduplicates on the scan identifier over a rolling one-hour window. The Notifier reads the
     Resolver's output but does not call it synchronously, so a Resolver outage delays notifications rather than
     dropping them. Retention for delivered notifications is thirty days."""),
    ("Parcel lifecycle", """A parcel moves through accepted, in transit, out for delivery and delivered. A parcel
     with no scan for fourteen consecutive days is marked Lost and the claims process begins automatically.
     Claims are handled by the support team and resolved within ten working days."""),
]


def _ready_index(name="measure", cfg=None):
    existing = [i for i in client.get("/api/indexes").json() if i["name"] == name and i["status"] == "ready"]
    if existing:
        return existing[0]["id"]
    have = {d["title"] for d in store.documents()}
    doc_ids = [d["id"] for d in store.documents() if d["title"] in dict(FIXTURE_DOCS)]
    for title, text in FIXTURE_DOCS:
        if title not in have:
            doc_ids.append(store.add_document(title, " ".join(text.split()), {"corpus": "fixture"}))
    iid = store.create_index(name, indexer.merged_config({**(cfg or {}), "doc_ids": doc_ids}))
    asyncio.run(indexer.build(store, iid, MODELS))
    assert store.index(iid)["status"] == "ready", store.index(iid)["stats"]
    return iid


QUESTION = "What happens when Redis is unavailable for the Resolver?"


def test_generation_off_is_honoured_on_the_router_s_direct_path():
    """A chitchat route used to generate anyway, which billed a sweep that had disabled generation
    and left it scoring an empty retrieval run."""
    rt = indexer.runtime(store, _ready_index())
    out = asyncio.run(execute(rt, MODELS, "hello", {"generate": {"enabled": False}}))
    assert [s["stage"] for s in out["stages"]] == ["route", "generate"]
    assert out["result"]["answer"] == ""
    generate = out["stages"][-1]
    assert generate["calls"] == [], "the generate stage must not call a model"
    assert generate["usage"]["tokens_out"] == 0
    assert generate["data"]["skipped"]


def test_rerank_off_keeps_every_fused_candidate():
    """Off used to still truncate to rerank.top_n, under a field the UI hides when it is off."""
    rt = indexer.runtime(store, _ready_index())
    base = {"generate": {"enabled": False}, "route": {"enabled": False}}
    off = asyncio.run(execute(rt, MODELS, QUESTION, {**base, "rerank": {"enabled": False, "top_n": 2}}))
    assert len(off["result"]["reranked"]) == len(off["result"]["fused"]), "top_n must not apply when off"
    on = asyncio.run(execute(rt, MODELS, QUESTION, {**base, "rerank": {"enabled": True, "top_n": 2}}))
    assert len(on["result"]["reranked"]) == 2, "on, it still cuts"


def test_the_context_budget_covers_the_first_block_and_reports_the_overflow():
    rt = indexer.runtime(store, _ready_index())
    out = asyncio.run(execute(rt, MODELS, QUESTION, {"generate": {"enabled": False}, "route": {"enabled": False},
                                                     "context": {"token_budget": 100}}))
    data = [s for s in out["stages"] if s["stage"] == "context"][0]["data"]
    assert len(data["blocks"]) == 1, "one block is always delivered, so the context is never empty"
    assert data["blocks"][0]["overflow"] is True, "and it is marked when it alone exceeds the budget"
    assert data["dropped"], "the passages it displaced are named"
    assert data["prompt_overhead_tokens"] > 0 and data["estimated_total_tokens"] > data["tokens"]


def test_community_reports_are_charged_to_the_context_budget():
    iid = _ready_index("measure-graph", {"graph": {"enabled": True, "communities": True}})
    rt = indexer.runtime(store, iid)
    assert rt.communities, "this test needs an index with community reports"
    out = asyncio.run(execute(rt, MODELS, QUESTION, {
        "generate": {"enabled": False}, "route": {"enabled": False}, "communities": {"enabled": True, "k": 2},
        "context": {"token_budget": 1}}))  # smaller than a single report, so only the first can land
    data = [s for s in out["stages"] if s["stage"] == "context"][0]["data"]
    assert len(data["blocks"]) == 1, "reports no longer bypass the budget"
    assert data["blocks"][0]["community"] == 0 and data["blocks"][0]["overflow"] is True
    assert data["dropped"], "the report and the passages it displaced are named"


def test_filters_matching_nothing_report_no_evidence_by_default():
    rt = indexer.runtime(store, _ready_index())
    allowed = rt.allowed([{"field": "doc_type", "op": "eq", "value": "no-such-type"}])
    assert allowed is not None and not allowed.any(), "the fixture needs a filter that matches nothing"


# --------------------------------------------------------------------- controlled experiments
def test_controlled_pins_the_cache_and_adaptive_routing():
    cfg = controlled({"route": {"enabled": True, "apply": True}, "cache": {"enabled": True}})
    assert cfg["cache"]["enabled"] is False
    assert cfg["route"]["apply"] is False
    assert cfg["route"]["enabled"] is True, "the router still classifies; it just cannot rewrite settings"


def test_controlled_does_not_mutate_the_caller_s_config():
    original = {"route": {"apply": True}, "cache": {"enabled": True}}
    controlled(original)
    assert original == {"route": {"apply": True}, "cache": {"enabled": True}}


def test_a_sweep_of_ef_search_reports_the_value_that_actually_ran():
    """ef below dense k runs at k, so two sweep points can otherwise plot the same run twice."""
    cfg = {"dense": {"k": 30, "mode": "hnsw"}}
    assert evals.pipeline_effective("dense.ef_search", 8, cfg) == 30
    assert evals.pipeline_effective("dense.ef_search", 64, cfg) == 64
    assert evals.pipeline_effective("rerank.top_n", 4, cfg) == 4, "unclamped params pass through"


def test_vector_search_reports_the_ef_it_actually_used():
    """ef below k is raised to k, because HNSW cannot return k neighbours from a narrower search.
    Two sweep points below k therefore execute identically, which the trace now says out loud."""
    import numpy as np
    from raglab.vectors import VectorIndex
    rng = np.random.default_rng(0)
    matrix = rng.normal(size=(200, 16)).astype(np.float32)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    index = VectorIndex(matrix)
    _, _, low = index.search(matrix[0], k=30, ef_search=8)
    assert (low["ef_requested"], low["ef_effective"]) == (8, 30)
    _, _, high = index.search(matrix[0], k=30, ef_search=128)
    assert (high["ef_requested"], high["ef_effective"]) == (128, 128)
    _, _, exact = index.search(matrix[0], k=5, mode="exact")
    assert exact["ef_effective"] is None and exact["verify_ms"] == 0.0


def test_the_retrieval_trace_separates_search_cost_from_verification_cost():
    """compare_exact runs a second, exhaustive search purely to measure recall. Its cost is
    measurement overhead, so it is reported apart from what the approximate search took."""
    rt = indexer.runtime(store, _ready_index())
    out = asyncio.run(execute(rt, MODELS, QUESTION, {"generate": {"enabled": False}, "route": {"enabled": False},
                                                     "dense": {"ef_search": 1, "k": 30, "compare_exact": True}}))
    search = [s for s in out["stages"] if s["stage"] == "retrieve"][0]["data"]["dense_search"]
    assert search["ef_requested"] == 1
    assert search["ef_effective"] == min(30, len(rt.chunks)) > 1, "raised to what k can reach"
    assert search["ef_raised_to_k"] is True
    assert search["search_ms"] >= 0 and search["verify_ms"] > 0, "verification is timed separately"
    off = asyncio.run(execute(rt, MODELS, QUESTION, {"generate": {"enabled": False}, "route": {"enabled": False},
                                                     "dense": {"compare_exact": False}}))
    assert [s for s in off["stages"] if s["stage"] == "retrieve"][0]["data"]["dense_search"]["verify_ms"] == 0.0


# --------------------------------------------------------------------- version-aware dedupe
def test_dedupe_keeps_the_current_version_not_the_one_ingested_first():
    """Keeping the first copy discarded revised policies; the freshness filter then removed the
    superseded original too, so the evidence vanished from the index entirely."""
    shared = "customers may return any unused product in original packaging with tags attached for a refund"
    old_doc = {"id": "old", "title": "Returns v2", "metadata": {"version": "2.0", "superseded_by": "POL-RET-v3"}}
    new_doc = {"id": "new", "title": "Returns v3", "metadata": {"version": "3.0"}}
    kept, removed = indexer.dedupe([{"doc": old_doc, "text": shared}, {"doc": new_doc, "text": shared}], 0.5)
    assert [r["doc"]["id"] for r in kept] == ["new"], kept
    assert removed[0]["duplicate_of"] == "Returns v3" and removed[0]["title"] == "Returns v2"
    assert removed[0]["kept_because"] == "newer version"


def test_dedupe_keeps_corpus_order_for_the_survivors():
    docs = [{"id": f"d{n}", "title": f"T{n}", "metadata": {}} for n in range(3)]
    rows = [{"doc": docs[0], "text": "alpha one"}, {"doc": docs[1], "text": "beta two"},
            {"doc": docs[2], "text": "gamma three"}]
    kept, removed = indexer.dedupe(rows, 0.9)
    assert not removed
    assert [r["text"] for r in kept] == ["alpha one", "beta two", "gamma three"]


def test_dedupe_never_drops_within_one_document():
    doc = {"id": "same", "title": "One", "metadata": {}}
    text = "identical wording repeated across two chunks of the same document body"
    kept, removed = indexer.dedupe([{"doc": doc, "text": text}, {"doc": doc, "text": text}], 0.5)
    assert len(kept) == 2 and not removed


# --------------------------------------------------------------------- compatibility
def test_an_evaluation_refuses_an_index_built_with_another_embedding_model():
    iid = _ready_index()
    sid = client.post("/api/evalsets", json={"name": "tiny", "items": [{"question": "x", "evidence": "y"}]}).json()["id"]
    original = MODELS["embedding"]
    MODELS["embedding"] = {**original, "model": "some-other-model"}
    try:
        r = client.post("/api/evalruns", json={"evalset_id": sid, "index_id": iid})
        assert r.status_code == 409, r.text
        assert "some-other-model" in r.json()["detail"]
        assert client.post("/api/sweeps", json={"evalset_id": sid, "index_id": iid, "param": "dense.k",
                                                "values": [5]}).status_code == 409
        assert client.post(f"/api/indexes/{iid}/probe", json={"text": "anything"}).status_code == 409
    finally:
        MODELS["embedding"] = original


def test_a_changed_embedding_prefix_is_caught_even_though_the_model_is_the_same():
    """Prefixes are not in the fingerprint, so they are compared on their own."""
    iid = _ready_index()
    original = MODELS["embedding"]
    MODELS["embedding"] = {**original, "query_prefix": "totally different: "}
    try:
        r = client.post("/api/query", json={"index_id": iid, "query": "anything"})
        assert r.status_code == 409 and "query_prefix" in r.json()["detail"], r.text
    finally:
        MODELS["embedding"] = original


def test_probing_a_chunk_by_index_needs_no_embedding_call_and_bounds_check_applies():
    iid = _ready_index()
    n = len(indexer.runtime(store, iid).chunks)
    assert client.post(f"/api/indexes/{iid}/probe", json={"i": 0, "k": 3}).status_code == 200
    r = client.post(f"/api/indexes/{iid}/probe", json={"i": n + 50, "k": 3})
    assert r.status_code == 404 and str(n) in r.json()["detail"]


# ================================================================== round-two corrections
def test_headline_abstention_is_unmeasured_when_nothing_decided():
    """The frozen lexical baseline turns off the grader AND generation. `abstained` is a plain bool
    defaulting to False, so averaging it reported a confident 0% for questions nobody ever asked."""
    sid, iid = _abstention_set(), _ready_index()
    es = [s for s in client.get("/api/evalsets").json() if s["id"] == sid][0]
    asyncio.run(evals.run_eval(store, "hl-off", indexer.runtime(store, iid), MODELS, es,
                               {"crag": {"enabled": False}, "generate": {"enabled": False}}, "both off", False, 5))
    m = [r for r in client.get(f"/api/evalruns?evalset_id={sid}").json()["runs"] if r["id"] == "hl-off"][0]["metrics"]
    assert m["measured"] == {"answers": False, "grader": False}
    assert m["abstention"] is None, "no decision was made, so there is no rate to report"
    assert m["false_abstention"] is None


def test_headline_abstention_is_a_number_once_something_decides():
    sid, iid = _abstention_set(), _ready_index()
    es = [s for s in client.get("/api/evalsets").json() if s["id"] == sid][0]
    asyncio.run(evals.run_eval(store, "hl-on", indexer.runtime(store, iid), MODELS, es,
                               {"crag": {"enabled": True}, "generate": {"enabled": False}}, "grader on", False, 5))
    m = [r for r in client.get(f"/api/evalruns?evalset_id={sid}").json()["runs"] if r["id"] == "hl-on"][0]["metrics"]
    assert m["measured"]["grader"] is True
    assert m["abstention"] is not None and m["false_abstention"] is not None


def test_the_prompt_stages_share_one_cut_off_so_the_funnel_cannot_invert():
    """`context` was cut at k while `delivered` was not, so the last bar of the funnel could read
    higher than the one before it although no extra passage reached the model."""
    texts = ["filler one", "filler two", "filler three", "the evidence sentence"]
    rt = FakeRuntime(texts)
    item = {"evidence": "the evidence sentence"}
    blocks = [{"i": n, "text": t, "doc_id": "d", "title": ""} for n, t in enumerate(texts)]
    ctx = evals.score_list([0, 1, 2, 3], rt, item, k=len(texts))  # as evaluate() now scores it
    assert ctx["recall"] == evals.score_blocks(blocks, item)["recall"] == 1.0
    assert evals.WHOLE_PROMPT_STAGES == ("context", "delivered")


def test_context_recall_counts_a_passage_the_model_read_past_k():
    sid, iid = _abstention_set(), _ready_index()
    es = [s for s in client.get("/api/evalsets").json() if s["id"] == sid][0]
    asyncio.run(evals.run_eval(store, "cut", indexer.runtime(store, iid), MODELS, es,
                               {"generate": {"enabled": False}, "context": {"max_chunks": 6}}, "k1", False, 1))
    m = [r for r in client.get(f"/api/evalruns?evalset_id={sid}").json()["runs"] if r["id"] == "cut"][0]["metrics"]
    ctx, delivered = m["stages"]["context"], m["stages"]["delivered"]
    assert ctx["recall"] == delivered["recall"], "k=1 must not hide passages that went into the prompt"
    assert m["whole_prompt_stages"] == ["context", "delivered"]


@pytest.mark.parametrize("spec,field", [
    ({"param": "context.max_chunks", "values": [0, 6]}, "values[0]"),
    ({"param": "dense.k", "values": [-5]}, "values[-5]"),
    ({"param": "bm25.b", "values": [0.5, 3.0]}, "values[3.0]"),
    ({"param": "nonsense.path", "values": [1, 2]}, "param"),
    ({"param": "dense.k", "values": []}, "values"),
    ({"param": "dense.k", "values": [5], "k": 0}, "k"),
])
def test_a_sweep_that_would_not_measure_what_it_claims_is_refused(spec, field):
    errors = evals.validate_sweep({"config": {}, "k": 5, **spec})
    assert any(e["field"] == field for e in errors), (spec, errors)


def test_a_sound_sweep_passes_validation():
    assert evals.validate_sweep({"param": "dense.ef_search", "values": [32, 64, 128], "config": {}, "k": 5}) == []
    assert evals.validate_sweep({"param": "index", "values": ["a", "b"], "config": {}, "k": 5}) == []


def test_the_sweep_endpoint_refuses_the_value_a_single_query_would_refuse():
    iid = _ready_index()
    sid = _abstention_set()
    bad = client.post("/api/sweeps", json={"evalset_id": sid, "index_id": iid,
                                           "param": "context.max_chunks", "values": [0, 6], "k": 5})
    assert bad.status_code == 422, bad.text
    assert any("max_chunks" in e["message"] for e in bad.json()["detail"]["errors"])
    good = client.post("/api/sweeps", json={"evalset_id": sid, "index_id": iid,
                                            "param": "context.max_chunks", "values": [2, 6], "k": 5})
    assert good.status_code == 200, good.text
    client.post(f"/api/sweeps/{good.json()['id']}/stop")


@pytest.mark.parametrize("cfg,field", [
    ({"chunking": {"size": -100}}, "chunking.size"),
    ({"chunking": {"size": 100, "overlap": 500}}, "chunking.overlap"),
    ({"chunking": {"strategy": "telepathy"}}, "chunking.strategy"),
    ({"hnsw": {"M": 0}}, "hnsw.M"),
    ({"hnsw": {"ef_construction": -5}}, "hnsw.ef_construction"),
    ({"dedupe": {"threshold": 2.0}}, "dedupe.threshold"),
    ({"chunking": {"semantic": {"percentile": 150}}}, "chunking.semantic.percentile"),
])
def test_build_settings_that_would_crash_the_builder_are_refused(cfg, field):
    errors = indexer.validate_index(indexer.merged_config(cfg))
    assert any(e["field"] == field for e in errors), (cfg, errors)


def test_the_default_build_config_is_valid():
    assert indexer.validate_index(indexer.merged_config({})) == []


def test_creating_an_index_refuses_settings_that_used_to_return_200_then_die():
    """M=0 and ef_construction=-5 were accepted, then killed the build thread with a raw pybind11
    TypeError, leaving the index at 'failed' with nothing a reader could act on."""
    r = client.post("/api/indexes", json={"name": "doomed", "config": {"hnsw": {"M": 0, "ef_construction": -5}}})
    assert r.status_code == 422, r.text
    fields = {e["field"] for e in r.json()["detail"]["errors"]}
    assert {"hnsw.M", "hnsw.ef_construction"} <= fields
    assert not [i for i in client.get("/api/indexes").json() if i["name"] == "doomed"], "nothing was created"


def test_chunk_preview_and_hnsw_rebuild_validate_too():
    doc = store.documents()[0]
    assert client.post("/api/chunk-preview", json={"doc_id": doc["id"],
                                                   "chunking": {"size": 100, "overlap": 400}}).status_code == 422
    assert client.post(f"/api/indexes/{_ready_index()}/hnsw", json={"M": 0, "ef_construction": 200}).status_code == 422


def test_sweeps_carry_the_delivered_stage():
    """A chunking or parent-expansion sweep needs the one stage that says what the model read."""
    sid, iid = _abstention_set(), _ready_index()
    es = [s for s in client.get("/api/evalsets").json() if s["id"] == sid][0]
    asyncio.run(evals.run_sweep(store, "sw-del", MODELS, es,
                                {"param": "context.max_chunks", "values": [2, 4], "index_id": iid,
                                 "config": {}, "k": 5}))
    points = json.loads(store.q("SELECT points FROM sweeps WHERE id='sw-del'")[0]["points"])
    assert points, "the sweep produced no points"
    for p in points:
        assert "delivered" in p["metrics"]["stages"], p["metrics"]["stages"].keys()


def test_runs_are_stamped_with_the_metric_definitions_that_produced_them():
    """A run saved before `context` moved off the top-k cut-off must stay identifiable, or the UI
    relabels yesterday's numbers with today's meaning."""
    sid, iid = _abstention_set(), _ready_index()
    es = [s for s in client.get("/api/evalsets").json() if s["id"] == sid][0]
    asyncio.run(evals.run_eval(store, "ver", indexer.runtime(store, iid), MODELS, es,
                               {"generate": {"enabled": False}}, "versioned", False, 5))
    m = [r for r in client.get(f"/api/evalruns?evalset_id={sid}").json()["runs"] if r["id"] == "ver"][0]["metrics"]
    assert m["version"] == evals.METRICS_VERSION >= 2
    assert m["whole_prompt_stages"] == ["context", "delivered"]


def test_a_run_saved_without_a_version_is_treated_as_older():
    """Mirrors isLegacy() in frontend/src/stages.ts: absent version means pre-versioning."""
    legacy = {"n": 3, "k": 5, "stages": {"context": {"recall": 0.864}}, "abstention": 0.0}
    assert "version" not in legacy
    assert legacy.get("version", 1) < evals.METRICS_VERSION


# ================================================================== reproducibility
def test_a_run_records_what_produced_it():
    rt = indexer.runtime(store, _ready_index())
    m = evals.manifest(store, rt, MODELS, {"rerank": {"top_n": 4}}, control=True)
    assert m["manifest_version"] == evals.MANIFEST_VERSION
    assert m["index"]["revision"]["vectors"] and m["index"]["revision"]["hnsw"]
    assert m["software"]["prompts"] and m["software"]["metrics"] == evals.METRICS_VERSION
    assert m["models"]["embedding"]["model"], "the roles that answered"
    assert m["pipeline"]["rerank"]["top_n"] == 4
    assert m["controlled"] is True
    assert m["corpus"]["documents"] > 0 and m["corpus"]["sha"]


def test_a_manifest_never_carries_a_secret():
    rt = indexer.runtime(store, _ready_index())
    original = MODELS["generator"]
    MODELS["generator"] = {**original, "api_key": "sk-do-not-store-this-anywhere"}
    try:
        m = evals.manifest(store, rt, MODELS, {}, control=False)
        assert "sk-do-not-store-this-anywhere" not in json.dumps(m)
        assert m["models"]["generator"]["api_key"].startswith("•")
    finally:
        MODELS["generator"] = original


def test_rebuilding_the_graph_is_reported_apart_from_rebuilding_the_index():
    """A rebuilt HNSW graph invalidates the approximation numbers but not exact-mode recall."""
    rt = indexer.runtime(store, _ready_index())
    before = evals.manifest(store, rt, MODELS, {}, control=False)
    after = json.loads(json.dumps(before))
    after["indexes"][0]["revision"]["hnsw"] = "different"
    notes = evals.drift(before, after)
    assert len(notes) == 1 and "HNSW graph has been rebuilt" in notes[0]
    assert "recall, MRR and the ranking score still hold" in notes[0]

    rebuilt = json.loads(json.dumps(before))
    rebuilt["indexes"][0]["revision"]["vectors"] = "different"
    assert "no longer exists" in evals.drift(before, rebuilt)[0]


def test_the_vectors_revision_tracks_the_matrix_and_not_the_index_id():
    """Identical corpus, chunking and embedding model means an identical matrix, and the hash says
    so. A different corpus changes it. That is what makes it usable for judging drift."""
    same_a = _ready_index("measure-same-1", {"chunking": {"size": 260}})
    same_b = _ready_index("measure-same-2", {"chunking": {"size": 260}})
    ra, rb = indexer.runtime(store, same_a), indexer.runtime(store, same_b)
    assert ra.id != rb.id and ra.revision["vectors"] == rb.revision["vectors"]

    subset = [d["id"] for d in store.documents() if d["title"] in dict(FIXTURE_DOCS)][:1]
    fewer = store.create_index("measure-subset", indexer.merged_config({"doc_ids": subset}))
    asyncio.run(indexer.build(store, fewer, MODELS))
    assert indexer.runtime(store, fewer).revision["vectors"] != ra.revision["vectors"], "fewer documents, new matrix"
    store.delete_index(fewer)


def test_a_sweep_comparing_indexes_records_every_one_of_them():
    """Recording only the first made an index comparison's manifest describe one side of it."""
    a, b = _ready_index(), _ready_index("measure-graph", {"graph": {"enabled": True, "communities": True}})
    rts = [indexer.runtime(store, a), indexer.runtime(store, b)]
    m = evals.manifest(store, rts[0], MODELS, {}, control=True, indexes=rts)
    assert [i["id"] for i in m["indexes"]] == [a, b], "both sides of the comparison, in order"
    assert all(i["revision"]["vectors"] and i["revision"]["hnsw"] for i in m["indexes"])
    assert all(i["embedding_fingerprint"] for i in m["indexes"])

    # A rebuild of the second one is named, and named as the second one.
    after = json.loads(json.dumps(m))
    after["indexes"][1]["revision"]["vectors"] = "different"
    notes = evals.drift(m, after)
    assert len(notes) == 1 and rts[1].name in notes[0], notes


def test_drift_names_a_swapped_model_and_edited_rules():
    rt = indexer.runtime(store, _ready_index())
    before = evals.manifest(store, rt, MODELS, {}, control=False)
    after = json.loads(json.dumps(before))
    after["models"]["generator"]["model"] = "something-else"
    after["indexes"][0]["normalization"]["sha"] = "changed"  # rules live per index now
    after["software"]["prompts"] = "changed"
    notes = " | ".join(evals.drift(before, after))
    assert "generator role has changed" in notes and "normalisation rules have changed" in notes
    assert "prompt template has been edited" in notes


def test_drift_still_reads_a_manifest_written_before_rules_were_recorded_per_index():
    """Older manifests kept one rule set at the top level and no per-index entries."""
    rt = indexer.runtime(store, _ready_index())
    current = evals.manifest(store, rt, MODELS, {}, control=False)
    older = json.loads(json.dumps(current))
    older.pop("indexes")
    older["normalization"]["sha"] = "changed"
    older["corpus"]["sha"] = "changed"
    notes = " | ".join(evals.drift(older, current))
    assert "normalisation rules have changed" in notes and "set of documents" in notes


def test_a_run_from_before_manifests_says_so_rather_than_claiming_reproducibility():
    assert evals.drift(None, {"index": {}}) == [
        "This run was recorded before runs captured what produced them, so nothing can be compared."]


def test_the_manifest_endpoint_reports_no_drift_for_a_fresh_run():
    sid, iid = _abstention_set(), _ready_index()
    es = [s for s in client.get("/api/evalsets").json() if s["id"] == sid][0]
    asyncio.run(evals.run_eval(store, "mf", indexer.runtime(store, iid), MODELS, es,
                               {"generate": {"enabled": False}}, "manifested", False, 5))
    r = client.get("/api/evalruns/mf/manifest").json()
    assert r["reproducible"] is True and r["drift"] == []
    assert r["manifest"]["index"]["revision"]["vectors"]


def test_the_semantic_cache_is_namespaced_by_the_normalisation_rules_it_will_apply(restore_rules):
    """The rules live outside the pipeline config, so editing them used to leave the cache valid."""
    from raglab.pipeline import Pipeline
    rt = indexer.runtime(store, _ready_index())
    cfg = {"normalize": {"enabled": True}}
    store.x("INSERT OR REPLACE INTO kv VALUES ('normalization', ?)", (json.dumps({"aliases": {"a": "b"}}),))
    before = Pipeline(rt, MODELS, cfg, store=store).cfg_hash
    store.x("INSERT OR REPLACE INTO kv VALUES ('normalization', ?)", (json.dumps({"aliases": {"a": "totally other"}}),))
    after = Pipeline(rt, MODELS, cfg, store=store).cfg_hash
    assert before != after, "a rule change must not serve an answer retrieved under the old rules"
    off = {"normalize": {"enabled": False}}
    assert Pipeline(rt, MODELS, off, store=store).cfg_hash == Pipeline(rt, MODELS, off, store=store).cfg_hash


def test_latency_reports_the_pipeline_apart_from_the_cost_of_verifying_it():
    sid, iid = _abstention_set(), _ready_index()
    es = [s for s in client.get("/api/evalsets").json() if s["id"] == sid][0]
    asyncio.run(evals.run_eval(store, "ver-ms", indexer.runtime(store, iid), MODELS, es,
                               {"generate": {"enabled": False}, "dense": {"compare_exact": True}}, "timed", False, 5))
    m = [r for r in client.get(f"/api/evalruns?evalset_id={sid}").json()["runs"] if r["id"] == "ver-ms"][0]["metrics"]
    assert m["verification_ms"] >= 0
    assert m["pipeline_ms"] <= m["latency_ms"], "the diagnostic's cost comes out, never gets added"


def test_hnsw_recall_averages_every_retrieval_and_keeps_a_measured_zero():
    stage = lambda recall: {"stage": "retrieve", "data": {"hnsw_recall_vs_exact": recall}}  # noqa: E731
    assert evals.hnsw_recall({"stages": [stage(1.0), stage(0.5)]}) == 0.75, "a corrective second retrieval counts too"
    assert evals.hnsw_recall({"stages": [stage(0.0)]}) == 0.0, "measured, and it was zero"
    assert evals.hnsw_recall({"stages": [{"stage": "retrieve", "data": {}}, {"stage": "rerank", "data": {}}]}) is None
    assert evals.hnsw_recall({}) is None


def test_hnsw_recall_is_reported_only_when_it_was_measured():
    """None means not measured. Reading a run with the diagnostic off as 0% would invent a failure."""
    sid, iid = _abstention_set(), _ready_index()
    es = [s for s in client.get("/api/evalsets").json() if s["id"] == sid][0]
    for run_id, compare in (("recall-on", True), ("recall-off", False)):
        asyncio.run(evals.run_eval(store, run_id, indexer.runtime(store, iid), MODELS, es,
                                   {"generate": {"enabled": False}, "dense": {"compare_exact": compare}}, run_id, False, 5))
    runs = {r["id"]: r["metrics"] for r in client.get(f"/api/evalruns?evalset_id={sid}").json()["runs"]}
    assert 0.0 <= runs["recall-on"]["hnsw_recall"] <= 1.0
    assert runs["recall-off"]["hnsw_recall"] is None


def test_an_ef_search_sweep_always_measures_what_the_knob_trades_away():
    """ef_search only buys HNSW recall, so a sweep of it that does not measure that recall has
    nothing to plot. The "Fast hybrid baseline" starter turns the diagnostic off."""
    sid, iid = _abstention_set(), _ready_index()
    es = [s for s in client.get("/api/evalsets").json() if s["id"] == sid][0]
    asyncio.run(evals.run_sweep(store, "ef-recall", MODELS, es,
                                {"param": "dense.ef_search", "values": [32, 64], "index_id": iid, "k": 5,
                                 "config": {"dense": {"compare_exact": False}}}))
    saved = [s for s in client.get(f"/api/sweeps?evalset_id={sid}").json()["sweeps"] if s["id"] == "ef-recall"][0]
    assert all(p["metrics"]["hnsw_recall"] is not None for p in saved["points"])
    assert saved["spec"]["config"]["dense"]["compare_exact"] is True, "the record says what actually ran"


def test_a_sweep_of_another_knob_keeps_the_diagnostic_setting_it_was_given():
    sid, iid = _abstention_set(), _ready_index()
    es = [s for s in client.get("/api/evalsets").json() if s["id"] == sid][0]
    asyncio.run(evals.run_sweep(store, "k-recall", MODELS, es,
                                {"param": "rerank.top_n", "values": [4], "index_id": iid, "k": 5,
                                 "config": {"dense": {"compare_exact": False}}}))
    saved = [s for s in client.get(f"/api/sweeps?evalset_id={sid}").json()["sweeps"] if s["id"] == "k-recall"][0]
    assert saved["points"][0]["metrics"]["hnsw_recall"] is None


def test_repeats_and_warmup_are_recorded_and_bounded():
    sid, iid = _abstention_set(), _ready_index()
    es = [s for s in client.get("/api/evalsets").json() if s["id"] == sid][0]
    asyncio.run(evals.run_eval(store, "rep", indexer.runtime(store, iid), MODELS, es,
                               {"generate": {"enabled": False}}, "repeated", False, 5, False, 3, True))
    run = [r for r in client.get(f"/api/evalruns?evalset_id={sid}").json()["runs"] if r["id"] == "rep"][0]
    assert run["metrics"]["repeats"] == 3 and run["metrics"]["warmup"] is True
    assert run["rows"][0]["timings"] == 2, "3 runs, the first discarded as warm-up"
    assert evals.validate_sweep({"param": "dense.k", "values": [5], "k": 5, "repeats": 0})[0]["field"] == "repeats"
    assert evals.validate_sweep({"param": "dense.k", "values": [5], "k": 5, "warmup": True})[0]["field"] == "warmup"


# ================================================================== round-three corrections
@pytest.mark.parametrize("value,why", [(None, "null"), (float("nan"), "NaN"), (float("inf"), "infinity")])
def test_a_setting_that_is_not_a_number_is_refused_like_one_that_is_out_of_range(value, why):
    """`max_chunks: 0` was refused while `max_chunks: null` was accepted, and null is what a typo
    becomes: Number("typo") is NaN, and JSON.stringify(NaN) is null."""
    errors = validate(merged({"context": {"max_chunks": value}}))
    assert any(e["field"] == "context.max_chunks" for e in errors), (why, errors)


def test_an_absent_setting_is_still_allowed_to_be_absent():
    """Only explicit nulls are refused; leaving a setting out must keep using the default."""
    assert validate({"context": {}}) == []
    assert validate(merged({})) == []
    assert validate({}) == []


def test_null_is_refused_by_the_build_validator_too():
    """chunking.size=null returned 200 and then killed the build thread with a native type error."""
    for field, cfg in (("chunking.size", {"chunking": {"size": None}}),
                       ("hnsw.M", {"hnsw": {"M": None}}),
                       ("dedupe.threshold", {"dedupe": {"threshold": float("nan")}})):
        errors = indexer.validate_index(indexer.merged_config(cfg))
        assert any(e["field"] == field for e in errors), (field, errors)


def test_the_api_refuses_null_wherever_it_refuses_zero():
    iid, sid = _ready_index(), _abstention_set()
    assert client.post("/api/query", json={"index_id": iid, "query": "x",
                                           "config": {"context": {"max_chunks": None}}}).status_code == 422
    assert client.post("/api/indexes", json={"name": "nulled", "config": {"chunking": {"size": None}}}).status_code == 422
    r = client.post("/api/sweeps", json={"evalset_id": sid, "index_id": iid, "param": "context.max_chunks",
                                         "values": [2, None, 6], "k": 5})
    assert r.status_code == 422
    assert any("typo" in e["message"] for e in r.json()["detail"]["errors"]), r.json()
    assert not [i for i in client.get("/api/indexes").json() if i["name"] == "nulled"]


def test_the_rules_a_run_used_are_the_rules_it_recorded(restore_rules):
    """Read per question, an edit between question 1 and question 2 measured the two under
    different rules while the manifest recorded only the first set."""
    sid, iid = _abstention_set(), _ready_index()
    es = [s for s in client.get("/api/evalsets").json() if s["id"] == sid][0]
    store.x("INSERT OR REPLACE INTO kv VALUES ('normalization', ?)", (json.dumps({"aliases": {"w": "alpha"}}),))
    seen = []
    real = evals.execute

    async def spy(rt, models, q, cfg=None, **kw):
        seen.append(kw.get("rules"))
        store.x("INSERT OR REPLACE INTO kv VALUES ('normalization', ?)", (json.dumps({"aliases": {"w": "beta"}}),))
        return await real(rt, models, q, cfg, **kw)

    evals.execute = spy
    try:
        asyncio.run(evals.run_eval(store, "frozen", indexer.runtime(store, iid), MODELS, es,
                                   {"generate": {"enabled": False}, "normalize": {"enabled": True}}, "f", False, 5))
    finally:
        evals.execute = real
    assert len(seen) == len(es["items"]) > 1
    assert all(r == seen[0] for r in seen), f"every question ran under one rule set, got {seen}"
    saved = json.loads(store.q("SELECT manifest FROM evalruns WHERE id='frozen'")[0]["manifest"])
    assert saved["normalization"]["rules"] == seen[0], "and the manifest names that same set"


def test_a_pipeline_given_rules_ignores_a_later_edit(restore_rules):
    from raglab.pipeline import Pipeline
    rt = indexer.runtime(store, _ready_index())
    frozen = {"aliases": {"w": "alpha"}}
    pipe = Pipeline(rt, MODELS, {"normalize": {"enabled": True}}, store=store, rules=frozen)
    store.x("INSERT OR REPLACE INTO kv VALUES ('normalization', ?)", (json.dumps({"aliases": {"w": "beta"}}),))
    assert pipe._rules() == frozen, "the job's set, not whatever the store now holds"
    fresh = Pipeline(rt, MODELS, {"normalize": {"enabled": True}}, store=store)
    assert fresh._rules()["aliases"]["w"] == "beta", "an interactive query still reads the current set"


def test_execute_refuses_a_mismatched_index_without_going_through_the_api():
    """The guard was only at the API boundary, so a script or a new endpoint could skip it."""
    from raglab.pipeline import IncompatibleIndex, execute
    rt = indexer.runtime(store, _ready_index())
    original = MODELS["embedding"]
    MODELS["embedding"] = {**original, "model": "a-different-model"}
    try:
        with pytest.raises(IncompatibleIndex, match="a-different-model"):
            asyncio.run(execute(rt, MODELS, "anything", {"generate": {"enabled": False}}))
    finally:
        MODELS["embedding"] = original
    asyncio.run(execute(rt, MODELS, "anything", {"generate": {"enabled": False}}))  # matched again


# ================================================================== round-four corrections
def _scoped_pair(restore):
    """Two indexes, each with its own normalisation rules."""
    from raglab.normalize import rules_key
    a = _ready_index("scoped-a", {"chunking": {"size": 240}})
    b = _ready_index("scoped-b", {"chunking": {"size": 360}})
    store.x("INSERT OR REPLACE INTO kv VALUES (?,?)", (rules_key(a), json.dumps({"aliases": {"w": "alpha"}})))
    store.x("INSERT OR REPLACE INTO kv VALUES (?,?)", (rules_key(b), json.dumps({"aliases": {"w": "beta"}})))
    return a, b


def _evalset(sid=None):
    """Fetch a stored question set. The id has to exist before the list is fetched, which is why
    this is a function rather than a comprehension over a list captured earlier."""
    sid = sid or _abstention_set()
    return [s for s in client.get("/api/evalsets").json() if s["id"] == sid][0]


def _rules_seen(sweep_id, values, es, shared=False):
    """Which rules each index in an index sweep was actually measured under."""
    seen = []
    real = evals.evaluate

    async def spy(store_, rt, models, items, cfg, k, judge, job, on_row=None, control=False,
                  repeats=1, warmup=False, rules=None):
        seen.append((rt.id, (rules or {}).get("aliases", {}).get("w")))
        return await real(store_, rt, models, items, cfg, k, judge, job, on_row, control, repeats, warmup, rules)

    evals.evaluate = spy
    try:
        asyncio.run(evals.run_sweep(store, sweep_id, MODELS, es,
                                    {"param": "index", "values": values, "config": {"normalize": {"enabled": True}},
                                     "k": 5, "shared_rules": shared}))
    finally:
        evals.evaluate = real
    return seen


def test_each_index_in_a_sweep_is_measured_under_its_own_rules(restore_rules):
    """Freezing one set stopped rules drifting mid-job and handed the first index's rules to every
    other one, so reordering the values changed what each index was measured under."""
    a, b = _scoped_pair(restore_rules)
    es = _evalset()
    assert _rules_seen("ord-ab", [a, b], es) == [(a, "alpha"), (b, "beta")]
    assert _rules_seen("ord-ba", [b, a], es) == [(b, "beta"), (a, "alpha")]


def test_a_shared_rule_set_is_available_but_has_to_be_asked_for(restore_rules):
    """Holding the rules fixed is a legitimate controlled comparison — as a choice, not a default."""
    a, b = _scoped_pair(restore_rules)
    es = _evalset()
    assert _rules_seen("shared", [a, b], es, shared=True) == [(a, "alpha"), (b, "alpha")]


def test_a_sweep_records_the_rules_and_corpus_of_every_index_it_compared(restore_rules):
    """One rule set recorded for a comparison describes one side of it — the same mistake as one
    revision, one field over."""
    a, b = _scoped_pair(restore_rules)
    rts = [indexer.runtime(store, a), indexer.runtime(store, b)]
    by_index = {a: {"aliases": {"w": "alpha"}}, b: {"aliases": {"w": "beta"}}}
    m = evals.manifest(store, rts[0], MODELS, {}, control=True, indexes=rts, rules_by_index=by_index)
    assert [i["normalization"]["rules"]["aliases"]["w"] for i in m["indexes"]] == ["alpha", "beta"]
    assert m["normalization"]["uniform"] is False, "the comparison varies the rules as well as the index"
    assert all(i["corpus"]["sha"] for i in m["indexes"]), "and each index's corpus, not just the first"

    same = evals.manifest(store, rts[0], MODELS, {}, control=True, indexes=rts,
                          rules_by_index={a: {"x": 1}, b: {"x": 1}})
    assert same["normalization"]["uniform"] is True


def test_drift_names_which_index_changed_its_rules(restore_rules):
    a, b = _scoped_pair(restore_rules)
    rts = [indexer.runtime(store, a), indexer.runtime(store, b)]
    before = evals.manifest(store, rts[0], MODELS, {}, control=True, indexes=rts,
                            rules_by_index={a: {"w": "alpha"}, b: {"w": "beta"}})
    after = json.loads(json.dumps(before))
    after["indexes"][1]["normalization"]["sha"] = "changed"
    notes = evals.drift(before, after)
    assert len(notes) == 1 and rts[1].name in notes[0] and "normalisation rules" in notes[0], notes


@pytest.mark.parametrize("param,values", [
    ("dense.mode", ["hnsw", "exact"]),
    ("fusion.method", ["rrf", "weighted"]),
    ("context.order", ["rank", "edges"]),
])
def test_a_categorical_setting_declared_sweepable_can_be_swept(param, values):
    """A blanket numeric check refused the settings CHOICES declares sweepable, which include the
    canonical approximate-against-exact experiment."""
    assert param in evals.SWEEPABLE
    assert evals.validate_sweep({"param": param, "values": values, "config": {}, "k": 5}) == []


@pytest.mark.parametrize("param,bad,expect", [
    ("dense.mode", ["hnsw", "telepathy"], "must be one of"),
    ("dense.mode", ["hnsw", 4], "must be one of"),
    ("dense.k", ["hnsw"], "must be a number"),
    ("dense.k", [8, None], "typo"),
])
def test_a_sweep_value_is_judged_by_the_parameter_s_own_type(param, bad, expect):
    errors = evals.validate_sweep({"param": param, "values": bad, "config": {}, "k": 5})
    assert errors and any(expect in e["message"] for e in errors), errors


def test_a_categorical_sweep_runs_end_to_end():
    sid, iid = _abstention_set(), _ready_index()
    es = [s for s in client.get("/api/evalsets").json() if s["id"] == sid][0]
    asyncio.run(evals.run_sweep(store, "cat", MODELS, es,
                                {"param": "dense.mode", "values": ["hnsw", "exact"], "index_id": iid,
                                 "config": {}, "k": 5}))
    assert evals.SWEEPS["cat"]["status"] == "done"
    points = json.loads(store.q("SELECT points FROM sweeps WHERE id='cat'")[0]["points"])
    assert [p["value"] for p in points] == ["hnsw", "exact"]
    assert all(p["metrics"]["stages"] for p in points)


def test_the_api_and_the_pipeline_answer_compatibility_from_one_place():
    """The API kept its own copy of the fingerprint and prefix checks."""
    import inspect
    from raglab import main as m
    source = inspect.getsource(m._check_embedding)
    assert "indexer.compatibility" in source
    assert "embedding_fingerprint" not in source, "no second implementation to drift from the first"
    rt = indexer.runtime(store, _ready_index())
    original = MODELS["embedding"]
    MODELS["embedding"] = {**original, "model": "mismatched"}
    try:
        assert indexer.compatibility(rt, MODELS), "the shared function sees the problem"
        assert client.post("/api/query", json={"index_id": rt.id, "query": "x"}).status_code == 409
    finally:
        MODELS["embedding"] = original


# ================================================================== delete semantics
def _index_with(docs_titles=None):
    """An index built from the fixture corpus, rebuilt fresh so deletions in other tests cannot
    leak into this one."""
    have = {d["title"]: d["id"] for d in store.documents()}
    ids = [have.get(t) or store.add_document(t, " ".join(text.split()), {"corpus": "fixture"})
           for t, text in FIXTURE_DOCS]
    iid = store.create_index(f"del-{uuid.uuid4().hex[:6]}", indexer.merged_config({"doc_ids": ids}))
    asyncio.run(indexer.build(store, iid, MODELS))
    return iid, ids


def test_a_removed_document_stops_being_retrieved():
    """An index keeps its own chunk copies, so deleting the document left its text being retrieved
    and answered from while the Corpus page said it was gone."""
    iid, ids = _index_with()
    victim = ids[0]
    rt = indexer.runtime(store, iid)
    mine = [c["i"] for c in rt.chunks if c["doc_id"] == victim]
    assert mine, "the fixture needs the victim to have chunks"

    before = asyncio.run(execute(rt, MODELS, "What does the Resolver do during a Redis outage?",
                                 {"generate": {"enabled": False}, "route": {"enabled": False}}))
    assert any(i in mine for i in before["result"]["fused"]), "it was retrievable to begin with"

    assert client.delete(f"/api/documents/{victim}").status_code == 200
    rt = indexer.runtime(store, iid)                       # cache was cleared by the delete
    after = asyncio.run(execute(rt, MODELS, "What does the Resolver do during a Redis outage?",
                                {"generate": {"enabled": False}, "route": {"enabled": False}}))
    assert not any(i in mine for i in after["result"]["fused"]), "and is gone from every retriever"
    assert not any(i in mine for i in after["result"]["context"]), "and never reaches the prompt"
    store.delete_index(iid)


def test_the_stored_copy_is_reported_rather_than_hidden():
    iid, ids = _index_with()
    victim = ids[0]
    usage = client.get(f"/api/documents/{victim}/usage").json()["indexes"]
    assert [u["index_id"] for u in usage] == [iid] and usage[0]["chunks"] > 0

    r = client.delete(f"/api/documents/{victim}").json()
    assert r["chunks_excluded"] == usage[0]["chunks"]
    assert [i["index_id"] for i in r["indexes"]] == [iid]
    assert "stored copy" in r["note"] and "rebuild" in r["note"].lower(), r["note"]
    assert "One index still holds" in r["note"], "and reads as English for a single index"

    detail = client.get(f"/api/indexes/{iid}").json()["removed_documents"]
    assert detail["documents"] == 1 and detail["chunks"] == usage[0]["chunks"]

    rt = indexer.runtime(store, iid)
    out = asyncio.run(execute(rt, MODELS, "Redis", {"generate": {"enabled": False}, "route": {"enabled": False}}))
    note = [s["data"].get("removed_documents") for s in out["stages"] if s["stage"] == "retrieve"][0]
    assert note and "excluded from retrieval" in note["note"]
    store.delete_index(iid)


def test_a_self_query_filter_cannot_bring_a_removed_document_back():
    """The exclusion is ANDed with the filter mask, and survives relaxation."""
    iid, ids = _index_with()
    victim = ids[0]
    client.delete(f"/api/documents/{victim}")
    rt = indexer.runtime(store, iid)
    mine = [c["i"] for c in rt.chunks if c["doc_id"] == victim]
    for cfg in ({"self_query": {"enabled": True}},
                {"self_query": {"enabled": True, "relax_empty": True}}):
        out = asyncio.run(execute(rt, MODELS, "What does the Resolver do during a Redis outage?",
                                  {**cfg, "generate": {"enabled": False}, "route": {"enabled": False}}))
        assert not any(i in mine for i in out["result"]["fused"]), cfg
    store.delete_index(iid)


def test_deleting_a_document_no_index_was_built_from_says_so():
    did = store.add_document("Orphan note", "nothing was ever built from this one", {"corpus": "fixture"})
    r = client.delete(f"/api/documents/{did}").json()
    assert r["indexes"] == [] and r["chunks_excluded"] == 0
    assert "No index had been built from it" in r["note"]
    assert client.delete(f"/api/documents/{did}").status_code == 404, "and it is really gone"


def test_an_untouched_index_pays_nothing_for_the_check():
    """live_mask returns None when every document is still there, keeping retrievers unmasked."""
    iid, _ = _index_with()
    rt = indexer.runtime(store, iid)
    assert rt.removed.any() == False and rt.live_mask() is None
    assert rt.removed_docs == []
    store.delete_index(iid)


# ================================================================== corpus search
def test_search_covers_document_text_not_just_titles(own_corpus):
    """The list deliberately returns length(text), so the client has nothing to match against —
    content search has to happen on the server."""
    store.add_document("Unrelated title", "the restocking fee is waived for dead-on-arrival units",
                       {"corpus": "searchtest"})
    hits = client.get("/api/documents?q=restocking").json()
    mine = [h for h in hits if h["title"] == "Unrelated title"]
    assert mine, "a word that appears only in the body still finds the document"
    assert mine[0]["match"] == "text" and "restocking" in mine[0]["snippet"]
    assert "text" not in mine[0], "the full text is not shipped to the list"


def test_search_reports_where_the_match_was():
    store.add_document("Restocking policy", "this mentions restocking twice, restocking", {"corpus": "searchtest"})
    store.add_document("Quite separate", "restocking appears only down here", {"corpus": "searchtest"})
    by_title = {h["title"]: h for h in client.get("/api/documents?q=restocking").json()}
    assert by_title["Restocking policy"]["match"] == "title and text"
    assert by_title["Quite separate"]["match"] == "text"
    for d in store.documents():
        if d["metadata"].get("corpus") == "searchtest":
            store.x("DELETE FROM documents WHERE id=?", (d["id"],))


def test_search_is_case_insensitive_and_empty_means_everything():
    did = store.add_document("CaseTest", "The Restocking Fee Applies", {"corpus": "searchtest"})
    assert any(h["id"] == did for h in client.get("/api/documents?q=RESTOCKING").json())
    assert any(h["id"] == did for h in client.get("/api/documents?q=restocking").json())
    assert len(client.get("/api/documents?q=").json()) == len(client.get("/api/documents").json())
    assert client.get("/api/documents?q=zzz-no-such-string").json() == []
    store.x("DELETE FROM documents WHERE id=?", (did,))


def test_a_snippet_shows_the_match_in_context():
    from raglab.main import _snippet
    text = "a" * 300 + " the restocking fee " + "b" * 300
    s = _snippet(text, "restocking")
    assert s and "restocking" in s
    assert s.startswith("…") and s.endswith("…"), "trimmed at both ends"
    assert len(s) < len(text) / 2
    assert _snippet("no match here", "restocking") is None


def test_a_router_reply_that_parses_as_a_list_falls_back_instead_of_failing_the_run(monkeypatch):
    """A 1B router ran past max_tokens mid-object, extract_json recovered an inner list, and the
    route stage called .get on it: every sweep over that question set failed outright."""
    from raglab import pipeline

    async def truncated(*args, **kwargs):
        return [{"id": "Meridian 45 Pack", "type": "product"}]

    monkeypatch.setattr(pipeline, "chat_json", truncated)
    rt = indexer.runtime(store, _ready_index())
    out = asyncio.run(execute(rt, MODELS, QUESTION, {"generate": {"enabled": False},
                                                       "route": {"enabled": True, "apply": True}}))
    route = next(s for s in out["stages"] if s["stage"] == "route")
    assert route["data"]["applied"] == ["dense weight x1.5"], "an unusable reply routes as the default, conceptual"
