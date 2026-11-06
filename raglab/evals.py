"""Evaluation and experiments.

Gold labels are chunking-independent, in either of two forms:
  evidence       verbatim quote(s) + optional doc_id  (LLM-generated sets; a list means every quote is needed)
  relevant_docs  document keys a careful human would cite (hand-written golden sets)
Items may carry `probes` (the failure mode the question is there to expose) and `type`
(simple | paraphrased | multi | unanswerable | golden). Unanswerable items score abstention, not retrieval.
"""
from __future__ import annotations
import asyncio, copy, hashlib, json, math, random, re, time, uuid
import numpy as np
from . import prompts
from .config import redact
from .indexer import Runtime, runtime
from .pipeline import CHOICES, SWEEPABLE, execute, merged, validate
from .providers import chat_json
from .store import Store

try:
    from . import __version__
except ImportError:  # pragma: no cover
    __version__ = "0"

RUNS: dict[str, dict] = {}
SWEEPS: dict[str, dict] = {}
# Three different questions, kept apart on purpose:
#   bm25/dense/graph   what each channel reached, best query variant per channel (a diagnostic,
#                      selected with the gold label, so not comparable with the stages after it)
#   fused/reranked     one consistently defined ranked list, scored over its top k
#   context/delivered  the prompt itself, scored whole (see WHOLE_PROMPT_STAGES); `context` reads
#                      the original chunk text, `delivered` the text the model was actually given,
#                      so the pair isolates what parent expansion won and what refinement cut
STAGES = ("bm25", "dense", "graph", "fused", "reranked", "context", "delivered")
DIAGNOSTIC_STAGES = ("bm25", "dense", "graph")
#: Stages measured over the whole prompt rather than the top k. The stages before these are ranked
#: candidate lists, where "top k" is the question being asked. The prompt is not a candidate list:
#: every passage in it was handed to the model, and it is already capped by context.max_chunks.
#: Applying k on top hid passages the model demonstrably read, and — because `delivered` had no k
#: at all — let the last bar of the funnel read higher than the one before it with nothing gained.
WHOLE_PROMPT_STAGES = ("context", "delivered")

#: Stamped on every run so a reader can tell which definitions produced its numbers. Bump it when a
#: metric changes meaning, not when one is added. A run saved before this existed carries no version
#: and must not be relabelled with today's semantics — its `context` recall was cut at k, and its
#: abstention rate averaged questions that nothing had decided.
#:   2  context scored over the whole prompt; abstention excludes undecided questions
METRICS_VERSION = 2

#: Shape of the manifest stored with each run. Bump when a field's meaning changes.
MANIFEST_VERSION = 1


def context_blocks(out: dict) -> list[dict]:
    """The context blocks from a finished :func:`raglab.pipeline.execute` result, or []."""
    return next((s["data"].get("blocks", []) for s in out.get("stages", []) if s["stage"] == "context"), [])


def _norm(t: str) -> str:
    return re.sub(r"\W+", " ", t.lower()).strip()


def _quote_in(text: str, quote: str) -> bool:
    ev = _norm(quote)
    if not ev:
        return False
    if ev[:90] in text or ev[-90:] in text:
        return True
    toks = ev.split()
    return len(toks) > 3 and sum(t in text for t in toks) / len(toks) >= 0.8 and \
        any(" ".join(toks[n:n + 4]) in text for n in range(len(toks) - 3))


def units(item: dict) -> list:
    """The things that must be found: document keys, or evidence quotes."""
    if item.get("relevant_docs"):
        return [("doc", d) for d in item["relevant_docs"]]
    ev = item.get("evidence") or []
    return [("quote", q) for q in ([ev] if isinstance(ev, str) else ev) if q]


def covers(chunk: dict, unit, item: dict) -> bool:
    kind, val = unit
    if kind == "doc":
        m = chunk["meta"]
        return val in (m.get("doc_key"), chunk["doc_id"], m.get("title"), str(m.get("source", "")).rsplit(".", 1)[0])
    if item.get("doc_id") and chunk["doc_id"] != item["doc_id"] and not isinstance(item.get("evidence"), list):
        return False
    return _quote_in(_norm(chunk["text"]), val)


def is_relevant(chunk: dict, item: dict) -> bool:
    return any(covers(chunk, u, item) for u in units(item))


def covers_block(block: dict, unit, item: dict) -> bool:
    """Whether a delivered context block carries a unit, judged on the text the generator was given.

    Unlike :func:`covers` this reads ``block["text"]`` — the parent passage when the chunk was
    expanded, the surviving strips when refinement cut it, or a community report — so evidence that
    expansion added counts and evidence that refinement removed does not.
    """
    kind, val = unit
    if kind == "doc":
        return val in (block.get("doc_key"), block.get("doc_id"), block.get("title"))
    if item.get("doc_id") and block.get("doc_id") and block["doc_id"] != item["doc_id"] \
            and not isinstance(item.get("evidence"), list):
        return False
    return _quote_in(_norm(block.get("text", "")), val)


def score_blocks(blocks: list[dict], item: dict) -> dict:
    """Score the assembled prompt, using the same metric definitions as :func:`score_list`.

    Blocks are in the order the generator reads them and every one of them was delivered, so there
    is no ``k``: this measures what the model had, not what retrieval reached. ``context`` is scored
    the same way for the same reason — see :data:`WHOLE_PROMPT_STAGES`.
    """
    us = units(item)
    first_of = [next((n + 1 for n, b in enumerate(blocks) if covers_block(b, u, item)), None) for u in us]
    found = [r for r in first_of if r]
    first = min(found) if found else None
    recall = len(found) / max(len(us), 1)
    ranks = sorted(set(found))
    dcg = sum(1 / math.log2(r + 1) for r in ranks)
    ideal = sum(1 / math.log2(n + 2) for n in range(len(ranks)))
    return {"rank": first, "hit": 1.0 if first else 0.0, "mrr": 1 / first if first else 0.0,
            "recall": recall, "ndcg": (dcg / ideal) * recall if ideal else 0.0}


def score_list(ids: list[int], rt: Runtime, item: dict, k: int) -> dict:
    """Score one ranked list of chunk ids against an item's expected units.

    Every metric is defined over *units* (document keys, or evidence quotes), never over chunks,
    so the numbers do not move when chunking changes:

    ``rank``   1-based position of the first chunk covering any unit, over the whole list.
    ``hit``    1.0 when that position is within ``k``.
    ``mrr``    reciprocal of that position.
    ``recall`` share of units covered by some chunk within ``k``.
    ``ndcg``   position quality x coverage, in [0, 1]. Position quality is the DCG of the
               *distinct* ranks that carry evidence over the DCG of those same ranks packed at
               the top of the list, so one chunk covering several units counts once and cannot
               push the score past 1. Coverage is ``recall``. For a single-unit item this is
               exactly ``1 / log2(rank + 1)``.

    :param ids: Retrieved chunk ids, best first.
    :param rt: Runtime holding the chunks ``ids`` index into.
    :param item: Gold item, as described in this module's docstring.
    :param k: Cut-off for ``hit``, ``recall`` and ``ndcg``.
    :returns: The metrics above; ``rank`` is ``None`` when nothing was found.
    """
    us = units(item)
    first_of = [next((r + 1 for r, i in enumerate(ids) if covers(rt.chunks[i], u, item)), None) for u in us]
    found = [r for r in first_of if r]
    first = min(found) if found else None
    in_k = [r for r in found if r <= k]
    recall = len(in_k) / max(len(us), 1)
    ranks = sorted(set(in_k))  # one chunk covering two quotes is one well-placed position, not two
    dcg = sum(1 / math.log2(r + 1) for r in ranks)
    ideal = sum(1 / math.log2(n + 2) for n in range(len(ranks)))
    return {"rank": first, "hit": 1.0 if first and first <= k else 0.0, "mrr": 1 / first if first else 0.0,
            "recall": recall, "ndcg": (dcg / ideal) * recall if ideal else 0.0}


# ------------------------------------------------------------------ run manifest
def manifest(store: Store, rt: Runtime, models: dict, cfg: dict, control: bool,
             rules: dict | None = None, indexes: list[Runtime] | None = None,
             rules_by_index: dict[str, dict] | None = None) -> dict:
    """An immutable record of what produced a run, captured when the run starts.

    A saved run used to keep only its pipeline settings and an index id. Neither says which models
    answered, which normalisation rules were in force, or whether the index has been rebuilt since —
    all of which change the numbers. This is what you would need to reproduce it, and what to
    compare against when a later run disagrees.

    Secrets never enter it: model roles are redacted the same way the API redacts them.

    :param store: Store to read the active normalisation rules from.
    :param rt: The loaded index.
    :param models: The model roles as they stood when the run was queued.
    :param cfg: The pipeline config the run was given, before routing may rewrite it per question.
    :param control: Whether the run pins the cache off and stops adaptive routing.
    :param rules: The rules the run froze, when it froze a set. Read fresh otherwise.
    :param indexes: Every index the run touches, for a sweep that compares several. Recording only
        the first made the manifest of an index comparison describe one side of it.
    :param rules_by_index: The rules each index was measured under, when they differ. Normalisation
        rules are scoped per index, so one set recorded for a comparison describes one side of it —
        the same mistake as recording one revision, one field over.
    :returns: A JSON-serialisable manifest.
    """
    touched = indexes or [rt]
    rules_by_index = rules_by_index or {}
    if rules is None:
        rules = _active_rules(store, rt.id)

    def corpus_of(r: Runtime) -> dict:
        docs = sorted({c["doc_id"] for c in r.chunks})
        return {"documents": len(docs), "sha": _sha(docs)}

    per_index = [{"id": r.id, "name": r.name, "revision": r.revision,
                  "embedding_fingerprint": r.stats.get("embedding_fingerprint"),
                  "corpus": corpus_of(r),
                  "normalization": {"sha": _sha(rules_by_index.get(r.id, rules)),
                                    "rules": rules_by_index.get(r.id, rules)}}
                 for r in touched]
    # True when every index was measured under the same rules. False means the comparison varies
    # the rules as well as the index, which a reader has to be told rather than left to infer.
    uniform = len({i["normalization"]["sha"] for i in per_index}) <= 1
    return {
        "manifest_version": MANIFEST_VERSION,
        "captured": time.time(),
        "software": {"raglab": __version__, "metrics": METRICS_VERSION, "prompts": prompts_revision()},
        "models": redact(models),
        "pipeline": merged(cfg),
        "controlled": control,
        "normalization": {"active": bool(rules), "sha": _sha(rules), "rules": rules, "uniform": uniform},
        "index": {"id": rt.id, "name": rt.name, "created": rt.created, "config": rt.config,
                  "revision": rt.revision, "embedding": {k: rt.stats.get(k) for k in
                  ("embedding_model", "embedding_fingerprint", "embedding_query_prefix", "embedding_doc_prefix")}},
        "corpus": corpus_of(rt),
        "indexes": per_index,
    }


def _active_rules(store: Store, index_id: str | None = None) -> dict:
    """The rules that would apply to this index: its own if it has them, else the shared set."""
    from .normalize import rules_key
    for key in (rules_key(index_id), "normalization") if index_id else ("normalization",):
        rows = store.q("SELECT value FROM kv WHERE key=?", (key,))
        if rows:
            return json.loads(rows[0]["value"])
    return {}


def _sha(value) -> str:
    return hashlib.sha1(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()[:12]


def prompts_revision() -> str:
    """Hash of every prompt template, so a reworded prompt is visible as a reason results moved."""
    templates = {k: v for k, v in vars(prompts).items() if k.isupper() and isinstance(v, (str, tuple))}
    return _sha(templates)


def drift(saved: dict, current: dict) -> list[str]:
    """What has changed between a saved manifest and the conditions now, in plain words.

    :param saved: The manifest stored with a run; a run from before manifests returns one note.
    :param current: A manifest captured now for the same index.
    :returns: Human-readable differences, empty when the run could be repeated as it stands.
    """
    if not saved:
        return ["This run was recorded before runs captured what produced them, so nothing can be compared."]
    notes = []
    now_by_id = {i["id"]: i for i in current.get("indexes", [])}
    for entry in saved.get("indexes", []) or [{**saved.get("index", {}), "id": saved.get("index", {}).get("id")}]:
        was, now = entry.get("revision", {}) or {}, (now_by_id.get(entry.get("id")) or {}).get("revision", {}) or {}
        where = f" ({entry.get('name')})" if len(saved.get("indexes", [])) > 1 else ""
        if not now:
            notes.append(f"The index{where} this run used is no longer loadable.")
        elif was.get("vectors") and was.get("vectors") != now.get("vectors"):
            notes.append(f"The index{where} has been rebuilt: its vectors differ, so every number here describes a "
                         f"corpus that no longer exists under this id.")
        elif was.get("hnsw") and was.get("hnsw") != now.get("hnsw"):
            notes.append(f"The HNSW graph{where} has been rebuilt with different parameters. Exact-mode recall, MRR "
                         f"and the ranking score still hold; the approximation numbers and latency do not.")
    for role in sorted(set(saved.get("models", {})) | set(current.get("models", {}))):
        a, b = saved.get("models", {}).get(role, {}), current.get("models", {}).get(role, {})
        if a.get("model") != b.get("model") or a.get("provider") != b.get("provider"):
            notes.append(f"The {role} role has changed: {a.get('model') or 'unset'} -> {b.get('model') or 'unset'}.")
    many = len(saved.get("indexes", [])) > 1
    for entry in saved.get("indexes", []):
        now = now_by_id.get(entry.get("id")) or {}
        where = f" ({entry.get('name')})" if many else ""
        if now and entry.get("normalization", {}).get("sha") != now.get("normalization", {}).get("sha"):
            notes.append(f"The normalisation rules{where} have changed since this run.")
        if now and entry.get("corpus", {}).get("sha") != now.get("corpus", {}).get("sha"):
            notes.append(f"The set of documents{where} has changed.")
    if not saved.get("indexes"):  # a manifest from before rules and corpus were recorded per index
        if saved.get("normalization", {}).get("sha") != current.get("normalization", {}).get("sha"):
            notes.append("The normalisation rules have changed since this run.")
        if saved.get("corpus", {}).get("sha") != current.get("corpus", {}).get("sha"):
            notes.append("The set of documents in this index has changed.")
    if saved.get("software", {}).get("prompts") != current.get("software", {}).get("prompts"):
        notes.append("A prompt template has been edited since this run.")
    if saved.get("software", {}).get("metrics") != current.get("software", {}).get("metrics"):
        notes.append("The metric definitions have changed since this run, so its numbers mean something else.")
    return notes


# ------------------------------------------------------------------ question generation
async def generate_set(store: Store, rt: Runtime, models: dict, n: int, name: str, mix: dict | None = None) -> str:
    """mix: share of each type, e.g. {"simple": .4, "paraphrased": .3, "multi": .2, "unanswerable": .1}"""
    mix = mix or {"simple": 1.0}
    rng = random.Random(11)
    pool = [c for c in rt.chunks if len(c["text"]) > 200] or rt.chunks
    total = sum(mix.values()) or 1.0
    plan = [t for t, share in mix.items() for _ in range(round(n * share / total))][:n] or ["simple"] * n
    gen, items = models["generator"], []

    async def ask(prompt: str, task: str):
        out = await chat_json(gen, [{"role": "user", "content": prompt}], task=task, default={}) or {}
        return out if isinstance(out, dict) else {}

    for kind in plan:
        c = rng.choice(pool)
        if kind == "multi":
            other = rng.choice([x for x in pool if x["doc_id"] != c["doc_id"]] or pool)
            out = await ask(prompts.EVALGEN_MULTI.format(a=c["text"][:1800], b=other["text"][:1800]), "evalgen_multi")
            ev = out.get("evidence")
            if out.get("question") and isinstance(ev, list) and len(ev) >= 2:
                items.append({"question": out["question"], "answer": out.get("answer", ""), "evidence": [str(e) for e in ev[:2]],
                              "type": "multi", "probes": "Multi-hop: needs a fact from each of two documents."})
        elif kind == "unanswerable":
            out = await ask(prompts.EVALGEN_UNANSWERABLE.format(titles="; ".join(rt.titles[:40]), passage=c["text"][:1200]), "evalgen_unanswerable")
            if out.get("question"):
                items.append({"question": out["question"], "answer": "", "evidence": "", "type": "unanswerable",
                              "probes": "Out of corpus: the right behaviour is to abstain, not to improvise."})
        else:
            tmpl, task = (prompts.EVALGEN_PARAPHRASE, "evalgen_paraphrase") if kind == "paraphrased" else (prompts.EVALGEN, "evalgen")
            out = await ask(tmpl.format(passage=c["text"][:2500]), task)
            if out.get("question") and out.get("evidence"):
                items.append({"question": out["question"], "answer": out.get("answer", ""), "evidence": str(out["evidence"]),
                              "doc_id": c["doc_id"], "type": kind,
                              "probes": "Semantic gap: the question avoids the passage's own vocabulary." if kind == "paraphrased"
                              else "Direct lookup written from a single passage (the easy case)."})
    sid = uuid.uuid4().hex[:8]
    store.x("INSERT INTO evalsets VALUES (?,?,?,?)", (sid, name, json.dumps(items), time.time()))
    return sid


# ------------------------------------------------------------------ one evaluation pass
def verification_ms(out: dict) -> float:
    """Time the run spent proving its own approximation, which is measurement, not pipeline.

    ``dense.compare_exact`` runs a second exhaustive search purely to report HNSW recall. Leaving
    its cost inside the latency of an ef_search sweep makes the approximation look more expensive
    than it is — the thing being measured pays for the measurement.
    """
    return sum(s["data"].get("dense_search", {}).get("verify_ms", 0.0)
               for s in out.get("stages", []) if s["stage"] == "retrieve")


def hnsw_recall(out: dict) -> float | None:
    """HNSW recall against exact search for one question, or None if it was not measured.

    :param out: The pipeline result. A corrective second retrieval is its own stage, so it counts.
    :returns: The mean over the question's retrievals. ``dense.compare_exact`` off means None, never 0.
    """
    recalls = [s["data"]["hnsw_recall_vs_exact"] for s in out.get("stages", [])
               if s["stage"] == "retrieve" and "hnsw_recall_vs_exact" in s["data"]]
    return round(float(np.mean(recalls)), 4) if recalls else None


async def evaluate(store: Store, rt: Runtime, models: dict, items: list[dict], cfg: dict, k: int, judge: bool, job: dict,
                   on_row=None, control: bool = False, repeats: int = 1, warmup: bool = False,
                   rules: dict | None = None):
    """Run every item and score it.

    :param on_row: Called as ``on_row(item, row, out)`` when each question finishes, so a caller can
        checkpoint results that a later failure would otherwise discard.
    :param control: Run each question under :func:`raglab.pipeline.controlled`.
    :param repeats: Time each question this many times. Scores come from the last run; latency is
        the median, which a single sample cannot give you on a local model server.
    :param warmup: Discard the first timing of each question. The first request after a model loads
        can be several times slower than steady state, and with one sample that lands in the chart.
    :param rules: Normalisation rules frozen for the whole run. Read per question, an edit between
        question 1 and question 2 measured the two under different rules while the manifest recorded
        only the first set.
    """
    if rules is None:
        rules = _active_rules(store, rt.id)
    rows = []
    for item in items:
        timings: list[float] = []
        for attempt in range(max(1, repeats)):
            out = await execute(rt, models, item["question"], cfg, store=store, control=control, rules=rules)
            if not (warmup and attempt == 0 and repeats > 1):
                timings.append(out["total_ms"] - verification_ms(out))
        res = out["result"]
        row = {"question": item["question"], "type": item.get("type", "golden" if item.get("relevant_docs") else "simple"),
               "probes": item.get("probes", ""), "expected": [u[1] if u[0] == "doc" else u[1][:60] for u in units(item)],
               "answer": res.get("answer", ""), "ms": out["total_ms"],
               # Median of the timed runs, with the verification diagnostic taken out of each.
               "pipeline_ms": round(float(np.median(timings)), 1) if timings else out["total_ms"],
               "verification_ms": round(verification_ms(out), 1), "timings": len(timings),
               "hnsw_recall": hnsw_recall(out),
               "abstained": bool(res.get("abstained")),
               "crag_abstained": res.get("crag_abstained"), "answer_abstained": res.get("answer_abstained"),
               "tokens": int(res.get("tokens_in", 0) + res.get("tokens_out", 0)), "cost": res.get("cost", 0.0),
               "groundedness": res.get("groundedness"), "scores": {}, "ranks": {}}
        if row["type"] != "unanswerable":
            per = {}
            for l in res.get("lists", []):
                s = score_list([h["i"] for h in l["hits"]], rt, item, k)
                if l["retriever"] not in per or s["mrr"] > per[l["retriever"]]["mrr"]:
                    per[l["retriever"]] = s
            for st in ("fused", "reranked"):
                per[st] = score_list(res.get(st, []), rt, item, k)
            # The prompt is scored over all of itself, not its top k: see WHOLE_PROMPT_STAGES.
            ctx_ids = res.get("context", [])
            per["context"] = score_list(ctx_ids, rt, item, max(len(ctx_ids), 1))
            # Candidate retrieval is what the index could reach; delivered evidence is the text the
            # generator actually read, after parent expansion, refinement and community reports.
            # They differ, and scoring only the first hides what expansion won and refinement cut.
            per["delivered"] = score_blocks(context_blocks(out), item)
            row["scores"] = per
            row["ranks"] = {st: v["rank"] for st, v in per.items()}
            row["retrieved"] = [{"doc": rt.chunks[i]["meta"].get("doc_key") or rt.chunks[i]["meta"].get("title", ""),
                                 "ok": is_relevant(rt.chunks[i], item)} for i in res.get("context", [])[:k]]
        if judge and res.get("answer") and row["type"] != "unanswerable":
            j = await chat_json(models["generator"], [{"role": "user", "content": prompts.JUDGE.format(
                query=item["question"], reference=item.get("answer", ""), context=res.get("context_text", "")[:6000],
                answer=res["answer"])}], task="judge", default={}) or {}
            if isinstance(j, dict):
                row["judge"] = {x: j.get(x) for x in ("faithfulness", "correctness", "reason")}
        rows.append(row)
        job["done"] += 1
        if on_row is not None:
            on_row(item, row, out)

    answerable = [r for r in rows if r["type"] != "unanswerable"]
    unans = [r for r in rows if r["type"] == "unanswerable"]

    def agg(stage: str):
        vals = [r["scores"][stage] for r in answerable if stage in r["scores"]]
        return {m: round(float(np.mean([v[m] for v in vals])), 3) for m in ("hit", "mrr", "ndcg", "recall")} if vals else None
    mean = lambda xs: round(float(np.mean(xs)), 4) if xs else None  # noqa: E731

    def share(rs: list[dict], key: str) -> float | None:
        """Share of rows where `key` is true, over the rows that produced that signal at all.

        None means the signal was never produced — the grader was off, or generation was. It must
        not collapse to 0.0: "nothing abstained" and "nobody was asked" are different findings, and
        a chart that draws them the same way is the defect this whole pass exists to remove.
        """
        return mean([1.0 if r[key] else 0.0 for r in rs if r[key] is not None])

    def decided(rs: list[dict]) -> list[dict]:
        """Rows where something actually made an abstention decision.

        ``abstained`` is a plain bool and defaults to False, so a run with the grader *and*
        generation both off — which is exactly what the frozen lexical baseline button asks for —
        would otherwise average a row of Falses and report a confident 0%. A question nobody
        answered and nobody graded did not decline; it was never asked.
        """
        return [r for r in rs if r["answer_abstained"] is not None or r["crag_abstained"] is not None]
    metrics = {"version": METRICS_VERSION,
               "n": len(rows), "k": k, "stages": {s: a for s in STAGES if (a := agg(s))},
               "diagnostic_stages": list(DIAGNOSTIC_STAGES), "whole_prompt_stages": list(WHOLE_PROMPT_STAGES),
               "latency_ms": int(np.mean([r["ms"] for r in rows])) if rows else 0,
               # What the pipeline costs, apart from what proving its approximation costs.
               "pipeline_ms": int(np.mean([r["pipeline_ms"] for r in rows])) if rows else 0,
               "verification_ms": int(np.mean([r["verification_ms"] for r in rows])) if rows else 0,
               "hnsw_recall": mean([r["hnsw_recall"] for r in rows if r["hnsw_recall"] is not None]),
               "repeats": repeats, "warmup": warmup,
               "tokens": int(np.mean([r["tokens"] for r in rows])) if rows else 0, "cost": mean([r["cost"] for r in rows]) or 0.0,
               "abstention": mean([1.0 if r["abstained"] else 0.0 for r in decided(unans)]),
               "false_abstention": mean([1.0 if r["abstained"] else 0.0 for r in decided(answerable)]),
               # The components behind those two, so turning the grader off cannot read as a
               # generator that answered anyway. None means that signal was not produced at all.
               "abstention_by": {
                   "answer": share(unans, "answer_abstained"), "crag": share(unans, "crag_abstained")},
               "false_abstention_by": {
                   "answer": share(answerable, "answer_abstained"), "crag": share(answerable, "crag_abstained")},
               "measured": {"answers": any(r["answer_abstained"] is not None for r in rows),
                            "grader": any(r["crag_abstained"] is not None for r in rows)},
               "groundedness": mean([r["groundedness"] for r in rows if isinstance(r["groundedness"], (int, float))])}
    by_type = {}
    for t in sorted({r["type"] for r in answerable}):
        vals = [r["scores"]["context"] for r in answerable if r["type"] == t and "context" in r["scores"]]
        by_type[t] = {"n": len(vals), "hit": mean([v["hit"] for v in vals]), "mrr": mean([v["mrr"] for v in vals])}
    metrics["by_type"] = by_type
    for key in ("faithfulness", "correctness"):
        vals = [float(r["judge"][key]) for r in rows if isinstance(r.get("judge", {}).get(key), (int, float))]
        if vals:
            metrics[key] = round(float(np.mean(vals)), 3)
    return metrics, rows


def save_trace(store: Store, index_id: str, query: str, out: dict) -> str:
    """Store one run's stages in the shared `traces` table and return its id.

    Evaluation rows keep only this id, so opening the trace behind a failed question uses the same
    endpoint the Workbench does instead of a second copy of the data.
    """
    tid = uuid.uuid4().hex[:10]
    store.x("INSERT INTO traces VALUES (?,?,?,?,?,?)",
            (tid, index_id, query, json.dumps(out.get("config", {})), json.dumps(out.get("stages", [])), time.time()))
    return tid


async def run_eval(store: Store, run_id: str, rt: Runtime, models: dict, evalset: dict, cfg: dict, label: str,
                   judge: bool, k: int = 10, control: bool = False, repeats: int = 1, warmup: bool = False):
    """Run an evaluation. ``models`` must be a snapshot taken when the job was queued, not the live
    role table: a job can sit behind the batch permit while the user edits the Models page, and it
    would otherwise run under models that never passed the compatibility check."""
    job = RUNS[run_id] = {"done": 0, "total": len(evalset["items"]), "error": None, "status": "running",
                          "partial": 0, "started": time.time()}
    done_rows: list[dict] = []
    frozen_rules = _active_rules(store, rt.id)  # one set for the whole job, and the one recorded
    record = manifest(store, rt, models, cfg, control, rules=frozen_rules)

    def checkpoint(item, row, out):
        # Written per question: a provider failure on question 19 of 20 used to discard the other 18.
        row["trace_id"] = save_trace(store, rt.id, item["question"], out)
        done_rows.append(row)
        job["partial"] = len(done_rows)
        # Carries the same keys a finished run does, so a reader that reaches for metrics.stages on
        # a row still in flight finds an empty mapping rather than nothing at all.
        partial = {"partial": True, "n": len(done_rows), "k": k, "stages": {}, "by_type": {},
                   "latency_ms": 0, "tokens": 0, "cost": 0.0}
        store.x("INSERT OR REPLACE INTO evalruns VALUES (?,?,?,?,?,?,?,?,?)",
                (run_id, evalset["id"], rt.id, label, json.dumps(cfg), json.dumps(partial),
                 json.dumps(done_rows), time.time(), json.dumps(record)))
    try:
        metrics, rows = await evaluate(store, rt, models, evalset["items"], cfg, k, judge, job, on_row=checkpoint,
                                       control=control, repeats=repeats, warmup=warmup, rules=frozen_rules)
        metrics["controlled"] = control
        metrics["partial"] = False
        store.x("INSERT OR REPLACE INTO evalruns VALUES (?,?,?,?,?,?,?,?,?)",
                (run_id, evalset["id"], rt.id, label, json.dumps(cfg), json.dumps(metrics), json.dumps(rows),
                 time.time(), json.dumps(record)))
        job["status"] = "done"
    except asyncio.CancelledError:
        job.update(status="cancelled", error=f"Stopped after {len(done_rows)} of {len(evalset['items'])} questions.")
        raise
    except Exception as e:  # noqa: BLE001
        job.update(status="failed", error=str(e)[:500])


# ------------------------------------------------------------------ parameter sweeps
def set_path(cfg: dict, path: str, value):
    node = cfg
    *parents, leaf = path.split(".")
    for p in parents:
        node = node.setdefault(p, {})
    node[leaf] = value


def validate_sweep(spec: dict) -> list[dict]:
    """Check a whole sweep plan — every point it will run — before any work starts.

    Validating only the base config let each swept value slip past the bounds that the same value
    is rejected by on a single query: a sweep over ``context.max_chunks`` including 0 finished with
    status ``done`` and plotted 0% recall as though it were a measurement. A parameter path outside
    :data:`raglab.pipeline.SWEEPABLE` is worse than invalid, because it runs cleanly and draws a
    flat line that reads as "this knob does nothing".

    :param spec: The sweep request: ``param``, ``values``, ``config``, ``k``.
    :returns: One ``{"field", "message"}`` per problem, empty when every point is runnable.
    """
    errors: list[dict] = []
    param, values = spec.get("param"), spec.get("values") or []
    if not values:
        errors.append({"field": "values", "message": "a sweep needs at least one value"})
    k = spec.get("k", 10)
    if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= 1000:
        errors.append({"field": "k", "message": f"must be a whole number between 1 and 1000, got {k!r}"})
    repeats = spec.get("repeats", 1)
    if isinstance(repeats, bool) or not isinstance(repeats, int) or not 1 <= repeats <= 20:
        errors.append({"field": "repeats", "message": f"must be a whole number between 1 and 20, got {repeats!r}"})
    elif spec.get("warmup") and repeats < 2:
        errors.append({"field": "warmup", "message": "discarding a warm-up run needs at least 2 repeats, "
                                                     "or there would be no timing left"})
    if param == "index":
        if len(values) < 2:
            errors.append({"field": "values", "message": "comparing indexes needs at least two of them"})
        return errors  # the ids themselves are resolved by the caller, which can 404 properly
    if param not in SWEEPABLE:
        errors.append({"field": "param", "message": f"{param!r} is not a sweepable setting. "
                                                    f"Sweepable: {', '.join(sorted(SWEEPABLE))}"})
        return errors  # without a real path there is nothing meaningful to check the values against
    # By the parameter's own type. A blanket numeric check refused the settings CHOICES declares
    # sweepable — dense.mode, fusion.method, context.order — which are among the more interesting
    # things to sweep: approximate against exact, one fusion rule against another.
    allowed = CHOICES.get(param)
    for value in values:
        if allowed is not None:
            if value not in allowed:
                errors.append({"field": "values", "message": f"{param} must be one of {', '.join(allowed)}, "
                                                             f"got {value!r}"})
                continue
        # Checked before the per-point validation, because null is what a typo in the values box
        # becomes: Number("typo") is NaN and JSON.stringify(NaN) is null. "one of your values is not
        # a number" is a more useful thing to read than a bound being violated.
        elif value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
            errors.append({"field": "values", "message": f"every value must be a number, got {value!r}. "
                                                         f"Check the list for a typo."})
            continue
        elif not math.isfinite(value):
            errors.append({"field": "values", "message": f"every value must be finite, got {value!r}"})
            continue
        cfg = copy.deepcopy(spec.get("config") or {})
        set_path(cfg, param, value)
        for err in validate(merged(cfg)):
            errors.append({"field": f"values[{value!r}]", "message": f"{err['field']} {err['message']}"})
    return errors


async def run_sweep(store: Store, sweep_id: str, models: dict, evalset: dict, spec: dict):
    """spec: {param: "dense.ef_search" | "index", values: [...], index_id, config, k, label}.
    param "index" sweeps over index ids, which is how index-time knobs (chunk size, strategy) are compared.

    Every point runs under :func:`raglab.pipeline.controlled`: the cache is bypassed and adaptive
    routing cannot rewrite the setting under test. Each point is saved as it finishes.

    A ``dense.ef_search`` sweep always measures HNSW recall: that is the one thing the knob trades
    against latency, and gold-label metrics on a small corpus do not move with it. The check's cost
    is kept out of ``pipeline_ms``, and the saved spec records that it ran.
    """
    if spec["param"] == "dense.ef_search":
        measured_cfg = copy.deepcopy(spec.get("config") or {})
        measured_cfg.setdefault("dense", {})["compare_exact"] = True
        spec = {**spec, "config": measured_cfg}
    values = spec["values"]
    job = SWEEPS[sweep_id] = {"done": 0, "total": len(values) * len(evalset["items"]) * max(1, int(spec.get("repeats", 1))),
                              "error": None, "status": "running", "partial": 0}
    label = spec.get("label") or spec["param"]
    points: list[dict] = []
    touched = [runtime(store, i) for i in (values if spec["param"] == "index" else [spec["index_id"]])]
    # Frozen per index, not once for the run. Freezing a single set stopped the rules drifting
    # mid-job, but handed the first index's rules to every other one: reordering the values then
    # changed what each index was measured under, which is a different way to be wrong.
    # `shared_rules` asks for the first index's set everywhere, for a comparison that deliberately
    # holds the rules fixed and varies only the index.
    shared = bool(spec.get("shared_rules"))
    rules_by_index = {rt.id: _active_rules(store, touched[0].id if shared else rt.id) for rt in touched}
    record = manifest(store, touched[0], models, spec.get("config") or {}, control=True,
                      rules=rules_by_index[touched[0].id], indexes=touched, rules_by_index=rules_by_index)

    def save(final: bool):
        store.x("INSERT OR REPLACE INTO sweeps VALUES (?,?,?,?,?,?,?)",
                (sweep_id, evalset["id"], label, json.dumps({**spec, "partial": not final}),
                 json.dumps(points), time.time(), json.dumps(record)))
    try:
        for v in values:
            cfg = copy.deepcopy(spec.get("config") or {})
            cfg.setdefault("generate", {})["enabled"] = False
            if spec["param"] == "index":
                rt = runtime(store, v)
                shown = rt.name
            else:
                rt = runtime(store, spec["index_id"])
                set_path(cfg, spec["param"], v)
                shown = v
            # Controlled: the cache is bypassed and routing cannot rewrite the setting under test.
            metrics, _ = await evaluate(store, rt, models, evalset["items"], cfg, int(spec.get("k", 10)), False, job,
                                        control=True, repeats=int(spec.get("repeats", 1)),
                                        warmup=bool(spec.get("warmup")), rules=rules_by_index[rt.id])
            # What was asked for against what the pipeline could honour: ef_search below dense k
            # runs at k, so two sweep points can otherwise plot as different runs of one setting.
            requested, effective = v, pipeline_effective(spec["param"], v, cfg)
            points.append({"value": shown, "metrics": metrics, "requested": requested, "effective": effective,
                           "degenerate": effective != requested, "index_id": rt.id,
                           "rules_sha": _sha(rules_by_index[rt.id])})
            job["partial"] = len(points)
            save(final=False)
        save(final=True)
        job["status"] = "done"
    except asyncio.CancelledError:
        job.update(status="cancelled", error=f"Stopped after {len(points)} of {len(values)} points.")
        raise
    except Exception as e:  # noqa: BLE001
        job.update(status="failed", error=str(e)[:500])


def pipeline_effective(param: str, value, cfg: dict):
    """The value a swept parameter will actually take, where the pipeline clamps it.

    :param param: Dotted parameter path, e.g. ``dense.ef_search``.
    :param value: The requested value.
    :param cfg: The config for this sweep point.
    :returns: The effective value; equal to ``value`` for parameters that are not clamped.
    """
    if param == "dense.ef_search":
        merged_cfg = merged(cfg)
        if merged_cfg["dense"]["mode"] == "hnsw":
            return max(int(value), int(merged_cfg["dense"]["k"]))
    return value
