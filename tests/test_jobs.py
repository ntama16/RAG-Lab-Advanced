"""Long-running local work: results survive a failure partway through, a run can be stopped, and a
process that dies mid-build does not leave an index claiming to be building for ever.
"""
import asyncio
import inspect
import json
import os
import tempfile

os.environ.setdefault("RAGLAB_MOCK", "1")
os.environ.setdefault("RAGLAB_DATA", tempfile.mkdtemp())

import pytest  # noqa: E402

from raglab import evals, indexer  # noqa: E402
from raglab import main as main_mod  # noqa: E402
from raglab.main import MODELS, app, store  # noqa: E402
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

DOCS = [
    ("Jobs alpha", "The alpha service writes its checkpoints to durable storage every thirty seconds and "
                   "replays the journal on restart, so a crash costs at most one interval of progress."),
    ("Jobs beta", "The beta service batches writes and flushes them only at the end of a run, which means an "
                  "interruption discards everything the run had produced so far."),
]


def _index(name="jobs"):
    ready = [i for i in client.get("/api/indexes").json() if i["name"] == name and i["status"] == "ready"]
    if ready:
        return ready[0]["id"]
    have = {d["title"]: d["id"] for d in store.documents()}
    ids = [have.get(t) or store.add_document(t, text, {"corpus": "jobs"}) for t, text in DOCS]
    iid = store.create_index(name, indexer.merged_config({"doc_ids": ids}))
    asyncio.run(indexer.build(store, iid, MODELS))
    return iid


def _evalset(n=4):
    items = [{"question": f"What does the alpha service do about checkpoints, variant {i}?",
              "evidence": "writes its checkpoints to durable storage"} for i in range(n)]
    return client.post("/api/evalsets", json={"name": f"jobs-{n}", "items": items}).json()["id"]


# --------------------------------------------------------------------- checkpointing
def test_every_question_is_saved_as_it_finishes():
    """A provider failure on the last question used to discard every result before it."""
    sid, iid = _evalset(4), _index()
    seen = []
    es = [s for s in client.get("/api/evalsets").json() if s["id"] == sid][0]

    async def watch():
        task = asyncio.create_task(evals.run_eval(store, "ck1", indexer.runtime(store, iid), MODELS, es,
                                                  {"generate": {"enabled": False}}, "checkpointed", False, 5))
        while not task.done():
            await asyncio.sleep(0)
            rows = store.q("SELECT rows FROM evalruns WHERE id='ck1'")
            if rows:
                seen.append(len(json.loads(rows[0]["rows"])))
        await task
    asyncio.run(watch())
    assert seen, "rows should be visible before the run finishes"
    assert max(seen) == 4 and min(seen) >= 1
    assert seen == sorted(seen), "the saved row count only ever grows"


def test_a_failure_partway_through_keeps_the_questions_already_scored():
    sid, iid = _evalset(4), _index()
    es = [s for s in client.get("/api/evalsets").json() if s["id"] == sid][0]
    real = evals.execute
    calls = {"n": 0}

    async def flaky(*a, **kw):
        calls["n"] += 1
        if calls["n"] > 2:
            raise RuntimeError("the provider went away")
        return await real(*a, **kw)

    evals.execute = flaky
    try:
        asyncio.run(evals.run_eval(store, "ck2", indexer.runtime(store, iid), MODELS, es,
                                   {"generate": {"enabled": False}}, "partial", False, 5))
    finally:
        evals.execute = real
    assert evals.RUNS["ck2"]["status"] == "failed"
    assert "provider went away" in evals.RUNS["ck2"]["error"]
    saved = json.loads(store.q("SELECT rows FROM evalruns WHERE id='ck2'")[0]["rows"])
    assert len(saved) == 2, "the two questions that did finish are still there"
    assert json.loads(store.q("SELECT metrics FROM evalruns WHERE id='ck2'")[0]["metrics"])["partial"] is True


def test_each_sweep_point_is_saved_as_it_finishes():
    sid, iid = _evalset(2), _index()
    es = [s for s in client.get("/api/evalsets").json() if s["id"] == sid][0]
    real = evals.evaluate
    calls = {"n": 0}

    async def flaky(*a, **kw):
        calls["n"] += 1
        if calls["n"] > 2:
            raise RuntimeError("provider gone")
        return await real(*a, **kw)

    evals.evaluate = flaky
    try:
        asyncio.run(evals.run_sweep(store, "sw-partial", MODELS, es,
                                    {"param": "dense.k", "values": [2, 4, 6], "index_id": iid, "config": {}, "k": 5}))
    finally:
        evals.evaluate = real
    assert evals.SWEEPS["sw-partial"]["status"] == "failed"
    points = json.loads(store.q("SELECT points FROM sweeps WHERE id='sw-partial'")[0]["points"])
    assert [p["value"] for p in points] == [2, 4], "the points that did finish survive"
    assert json.loads(store.q("SELECT spec FROM sweeps WHERE id='sw-partial'")[0]["spec"])["partial"] is True


def test_a_checkpoint_row_carries_the_shape_a_finished_run_has():
    """A row saved mid-run is still read by anything listing runs. Without the usual keys, a reader
    reaching for metrics.stages gets nothing back and the page it is building blanks."""
    sid, iid = _evalset(3), _index()
    es = [s for s in client.get("/api/evalsets").json() if s["id"] == sid][0]
    real = evals.execute
    calls = {"n": 0}

    async def flaky(*a, **kw):
        calls["n"] += 1
        if calls["n"] > 1:
            raise RuntimeError("gone")
        return await real(*a, **kw)

    evals.execute = flaky
    try:
        asyncio.run(evals.run_eval(store, "shape1", indexer.runtime(store, iid), MODELS, es,
                                   {"generate": {"enabled": False}}, "shape", False, 5))
    finally:
        evals.execute = real
    run = [r for r in client.get(f"/api/evalruns?evalset_id={sid}").json()["runs"] if r["id"] == "shape1"][0]
    assert run["metrics"]["partial"] is True
    for key in ("stages", "by_type", "n", "k", "latency_ms", "tokens", "cost"):
        assert key in run["metrics"], f"a partial run must still expose {key}"
    assert run["metrics"]["stages"] == {}, "empty, not absent"
    assert run["rows"][0]["ranks"], "the rows it did finish are complete"


def test_a_finished_run_is_not_marked_partial():
    sid, iid = _evalset(2), _index()
    es = [s for s in client.get("/api/evalsets").json() if s["id"] == sid][0]
    asyncio.run(evals.run_eval(store, "whole1", indexer.runtime(store, iid), MODELS, es,
                               {"generate": {"enabled": False}}, "whole", False, 5))
    run = [r for r in client.get(f"/api/evalruns?evalset_id={sid}").json()["runs"] if r["id"] == "whole1"][0]
    assert run["metrics"]["partial"] is False and run["metrics"]["stages"]


def test_deleting_a_run_removes_the_traces_it_stored():
    sid, iid = _evalset(2), _index()
    es = [s for s in client.get("/api/evalsets").json() if s["id"] == sid][0]
    asyncio.run(evals.run_eval(store, "tr-del", indexer.runtime(store, iid), MODELS, es,
                               {"generate": {"enabled": False}}, "traced", False, 5))
    rows = json.loads(store.q("SELECT rows FROM evalruns WHERE id='tr-del'")[0]["rows"])
    tids = [r["trace_id"] for r in rows]
    assert all(store.q("SELECT 1 FROM traces WHERE id=?", (t,)) for t in tids)
    assert client.delete("/api/evalruns/tr-del").json()["traces_removed"] == len(tids)
    assert not any(store.q("SELECT 1 FROM traces WHERE id=?", (t,)) for t in tids), "no orphans left behind"


# --------------------------------------------------------------------- stopping
def test_stopping_a_run_keeps_what_it_had_and_says_so():
    sid, iid = _evalset(6), _index()
    es = [s for s in client.get("/api/evalsets").json() if s["id"] == sid][0]

    async def start_then_stop():
        task = asyncio.create_task(evals.run_eval(store, "stop1", indexer.runtime(store, iid), MODELS, es,
                                                  {"generate": {"enabled": False}}, "stopped", False, 5))
        while evals.RUNS.get("stop1", {}).get("partial", 0) < 2:
            await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    asyncio.run(start_then_stop())
    assert evals.RUNS["stop1"]["status"] == "cancelled"
    assert "Stopped after" in evals.RUNS["stop1"]["error"]
    saved = json.loads(store.q("SELECT rows FROM evalruns WHERE id='stop1'")[0]["rows"])
    assert 2 <= len(saved) < 6, "the questions already scored are kept"


def test_stop_endpoints_report_whether_there_was_anything_to_stop():
    assert client.post("/api/evalruns/no-such-run/stop").json() == {"stopped": False}
    assert client.post("/api/sweeps/no-such-sweep/stop").json() == {"stopped": False}


# --------------------------------------------------------------------- task lifetime
def test_background_work_is_held_while_it_runs():
    """asyncio keeps only weak references to tasks, so an untracked build can be collected mid-run."""
    async def scenario():
        started = asyncio.Event()
        release = asyncio.Event()

        async def work():
            started.set()
            await release.wait()

        task = main_mod._track(work(), "held")
        await started.wait()
        assert main_mod.TASKS["held"] is task
        release.set()
        await task
        assert "held" not in main_mod.TASKS, "and released once it is done"
    asyncio.run(scenario())


def test_batch_work_runs_one_at_a_time():
    """Two evaluations against the same local model server would make both slower and their
    measured latencies meaningless."""
    async def scenario():
        order = []

        async def work(tag):
            order.append(f"{tag} start")
            await asyncio.sleep(0)
            order.append(f"{tag} end")

        await asyncio.gather(main_mod._batch(work("a")), main_mod._batch(work("b")))
        assert order == ["a start", "a end", "b start", "b end"], order
    asyncio.run(scenario())


# --------------------------------------------------------------------- interrupted builds
def test_an_index_left_building_by_a_dead_process_is_marked_failed():
    iid = store.create_index("interrupted", indexer.merged_config({}))
    assert store.index(iid)["status"] == "building"
    assert main_mod._mark_interrupted_builds() >= 1
    row = store.index(iid)
    assert row["status"] == "failed"
    assert "Interrupted" in row["stats"]["error"]


def test_marking_interrupted_builds_leaves_ready_indexes_alone():
    iid = _index()
    main_mod._mark_interrupted_builds()
    assert store.index(iid)["status"] == "ready"


# --------------------------------------------------------------------- traces from evaluations
def test_every_question_stores_a_trace_the_trace_endpoint_can_serve():
    sid, iid = _evalset(2), _index()
    es = [s for s in client.get("/api/evalsets").json() if s["id"] == sid][0]
    asyncio.run(evals.run_eval(store, "tr1", indexer.runtime(store, iid), MODELS, es,
                               {"generate": {"enabled": False}}, "traced", False, 5))
    rows = json.loads(store.q("SELECT rows FROM evalruns WHERE id='tr1'")[0]["rows"])
    assert all(r["trace_id"] for r in rows)
    served = client.get(f"/api/traces/{rows[0]['trace_id']}").json()
    assert served["query"] == rows[0]["question"]
    assert [s["stage"] for s in served["trace"]], "the stages are there to open"
    assert served["config"]["fusion"]["method"], "and the settings that actually ran"


# --------------------------------------------------------------------- ergonomics
def test_a_build_can_be_stopped_and_says_so():
    """A half-built index must not sit at 'building' with nobody working on it."""
    from raglab import main as m

    async def scenario():
        have = {d["title"]: d["id"] for d in store.documents()}
        ids = [have.get(t) or store.add_document(t, text, {"corpus": "jobs"}) for t, text in DOCS]
        iid = store.create_index("stoppable", indexer.merged_config({"doc_ids": ids}))
        task = m._track(indexer.build(store, iid, MODELS), iid)
        while indexer.JOBS.get(iid, {}).get("stage") in (None, "starting"):
            await asyncio.sleep(0)
        assert m._stop(iid) is True
        with pytest.raises(asyncio.CancelledError):
            await task
        return iid
    iid = asyncio.run(scenario())
    row = store.index(iid)
    assert row["status"] == "failed", "not left pending"
    assert "Stopped" in row["stats"]["error"]
    assert indexer.JOBS[iid]["stage"] == "cancelled"


@pytest.fixture
def own_permit(monkeypatch):
    """A contended semaphore binds to its event loop, and each test here runs in a new one."""
    monkeypatch.setattr(main_mod, "BATCH", asyncio.Semaphore(1))


def test_a_build_stopped_while_queued_is_marked_failed_not_left_building(own_permit):
    """Stopped behind the batch permit, the build never starts, so its own cancellation handling
    never runs. It used to sit at 'building' until the server restarted."""
    from raglab import main as m

    async def scenario():
        iid = store.create_index("queued-stop", indexer.merged_config({}))
        await m.BATCH.acquire()  # another job holds the permit, so this one has to wait
        try:
            task = m._start_build(iid)
            await asyncio.sleep(0)
            assert m._stop(iid) is True
            with pytest.raises(asyncio.CancelledError):
                await task
        finally:
            m.BATCH.release()
        await asyncio.sleep(0)  # the done callback runs on the next loop pass
        return iid
    row = store.index(asyncio.run(scenario()))
    assert row["status"] == "failed", "not left looking as if it is still building"
    assert "Stopped" in row["stats"]["error"]


def test_marking_a_stopped_build_leaves_a_finished_index_alone():
    """The cancel path and the done callback can both report a stop, and a stop that arrives as a
    build completes must not turn a good index into a failed one."""
    iid = store.create_index("already-done", indexer.merged_config({}))
    store.x("UPDATE indexes SET status='ready', stats=? WHERE id=?", (json.dumps({"chunks": 3}), iid))
    indexer.mark_stopped(store, iid)
    row = store.index(iid)
    assert row["status"] == "ready" and row["stats"] == {"chunks": 3}


def test_work_stopped_while_queued_is_closed_rather_than_abandoned(own_permit):
    """Nothing would ever await it, which is a RuntimeWarning and a leak for every stopped job."""
    async def scenario():
        async def work():
            raise AssertionError("a job stopped in the queue must never start")

        coro = work()
        await main_mod.BATCH.acquire()
        try:
            task = asyncio.create_task(main_mod._batch(coro))
            await asyncio.sleep(0)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        finally:
            main_mod.BATCH.release()
        assert inspect.getcoroutinestate(coro) == inspect.CORO_CLOSED
        assert main_mod.BATCH.locked() is False, "the permit is not lost with it"
    asyncio.run(scenario())


def test_stopping_a_build_that_is_not_running_reports_it():
    assert client.post("/api/indexes/no-such-index/stop").json() == {"stopped": False}


def test_normalisation_rules_can_belong_to_one_index(restore_rules):
    """One global row while the page asks you to pick an index means tuning one corpus retunes all."""
    a, b = _index("rules-a"), _index("rules-b")
    client.put("/api/normalization", json={"rules": {"aliases": {"shared": "everyone"}}, "index_id": None})
    scoped = client.put("/api/normalization", json={"rules": {"aliases": {"only": "index a"}}, "index_id": a}).json()
    assert scoped["scope"] == "index"
    assert client.get(f"/api/normalization?index_id={a}").json()["rules"]["aliases"] == {"only": "index a"}
    other = client.get(f"/api/normalization?index_id={b}").json()
    assert other["scope"] == "shared" and other["rules"]["aliases"] == {"shared": "everyone"}, "b is untouched"
    back = client.request("DELETE", f"/api/normalization?index_id={a}").json()
    assert back["scope"] == "shared" and back["rules"]["aliases"] == {"shared": "everyone"}


def test_the_pipeline_applies_the_rules_that_belong_to_the_index_it_queries(restore_rules):
    from raglab.pipeline import Pipeline
    a, b = _index("rules-a"), _index("rules-b")
    client.put("/api/normalization", json={"rules": {"aliases": {"shared": "everyone"}}, "index_id": None})
    client.put("/api/normalization", json={"rules": {"aliases": {"only": "index a"}}, "index_id": a})
    cfg = {"normalize": {"enabled": True}}
    ha = Pipeline(indexer.runtime(store, a), MODELS, cfg, store=store).cfg_hash
    hb = Pipeline(indexer.runtime(store, b), MODELS, cfg, store=store).cfg_hash
    assert ha != hb, "two indexes under different rules must not share a cache namespace"
    client.request("DELETE", f"/api/normalization?index_id={a}")


def test_duplicate_provenance_is_not_cut_to_eight():
    from raglab.indexer import dedupe
    shared = "identical wording that will collide across every one of these separate documents"
    rows = [{"doc": {"id": f"d{n}", "title": f"Doc {n}", "metadata": {}}, "text": shared} for n in range(30)]
    kept, removed = dedupe(rows, 0.5)
    assert len(kept) == 1 and len(removed) == 29, "every removal is recorded, not the first eight"
