"""Query pipeline. Every stage is optional and emits a trace event with its inputs, outputs,
timings and the raw model calls it made."""
from __future__ import annotations
import asyncio, copy, hashlib, json, re, time
from typing import AsyncIterator
import numpy as np
from . import normalize as norm, prompts
from .chunking import sentences
from .indexer import Runtime, compatibility
from .pipeline_bounds import MISSING, at as _at, number_error
from .providers import Recorder, chat_json, chat, chat_stream, embed, rerank, strip_think

DEFAULT_PIPELINE = {
    "cache": {"enabled": False, "threshold": 0.92},
    "condense": {"enabled": True},
    "normalize": {"enabled": False},
    "route": {"enabled": True, "apply": True},
    "self_query": {"enabled": False, "relax_empty": False},
    "decompose": {"enabled": False, "n": 3},
    "step_back": {"enabled": False},
    "multi_query": {"enabled": True, "n": 3},
    "hyde": {"enabled": False},
    "bm25": {"enabled": True, "k": 30, "k1": 1.5, "b": 0.75},
    "dense": {"enabled": True, "k": 30, "mode": "hnsw", "ef_search": 64, "compare_exact": True},
    "graph": {"enabled": False, "k": 20, "alpha": 0.25},
    "freshness": {"enabled": True},
    "fusion": {"method": "rrf", "rrf_k": 60, "weights": {"bm25": 1.0, "dense": 1.0, "graph": 1.0}, "top_n": 30},
    "rerank": {"enabled": True, "top_n": 8},
    "crag": {"enabled": True, "max_retries": 1, "refine": False},
    "communities": {"enabled": False, "k": 2},
    "context": {"expand_parents": True, "mmr": {"enabled": False, "lambda": 0.7}, "max_chunks": 6,
                "token_budget": 3000, "order": "rank"},
    "generate": {"enabled": True, "temperature": 0.1, "max_tokens": 900},
    "verify": {"enabled": False},
}


def merged(cfg: dict | None) -> dict:
    out = json.loads(json.dumps(DEFAULT_PIPELINE))
    for k, v in (cfg or {}).items():
        if isinstance(out.get(k), dict) and isinstance(v, dict):
            for k2, v2 in v.items():
                out[k][k2] = {**out[k][k2], **v2} if isinstance(out[k].get(k2), dict) and isinstance(v2, dict) else v2
        else:
            out[k] = v
    return out


# Bounds for every numeric knob: path -> (low, high, integer?). A value outside its range either
# crashes a library call (a negative k reaches hnswlib as a bad argument) or produces a run that
# looks finished but measured nothing, so both are rejected at the API boundary instead.
BOUNDS: dict[str, tuple[float, float, bool]] = {
    "cache.threshold": (0.0, 1.0, False),
    "decompose.n": (1, 10, True),
    "multi_query.n": (0, 10, True),
    "bm25.k": (1, 500, True),
    "bm25.k1": (0.0, 5.0, False),
    "bm25.b": (0.0, 1.0, False),
    "dense.k": (1, 500, True),
    "dense.ef_search": (1, 2000, True),
    "graph.k": (1, 500, True),
    "graph.alpha": (0.0, 1.0, False),
    "fusion.rrf_k": (1, 1000, True),
    "fusion.top_n": (1, 500, True),
    "fusion.weights.bm25": (0.0, 100.0, False),
    "fusion.weights.dense": (0.0, 100.0, False),
    "fusion.weights.graph": (0.0, 100.0, False),
    "rerank.top_n": (1, 500, True),
    "crag.max_retries": (0, 5, True),
    "communities.k": (1, 20, True),
    "context.max_chunks": (1, 100, True),
    "context.token_budget": (100, 200_000, True),
    "context.mmr.lambda": (0.0, 1.0, False),
    "generate.temperature": (0.0, 2.0, False),
    "generate.max_tokens": (1, 32_000, True),
}

CHOICES = {"fusion.method": ("rrf", "weighted"), "dense.mode": ("hnsw", "exact"), "context.order": ("rank", "edges")}

#: Every setting a sweep may vary: the numeric knobs plus the enumerated ones. A path outside this
#: set silently does nothing — `set_path` would happily create `foo.bar` and the pipeline would
#: never read it, drawing a flat curve that reads as "this parameter has no effect".
SWEEPABLE = frozenset(BOUNDS) | frozenset(CHOICES)





def validate(cfg: dict) -> list[dict]:
    """Check a merged pipeline config against :data:`BOUNDS` and the enumerated choices.

    :param cfg: A merged config (see :func:`merged`).
    :returns: One ``{"field", "message"}`` per problem, empty when the config is usable. Callers
        surface these per field rather than letting a bad value reach a model or a native call.
    """
    errors = []
    for path, (lo, hi, is_int) in BOUNDS.items():
        value = _at(cfg, path)
        if value is MISSING:
            continue
        if (err := number_error(path, value, lo, hi, is_int)):
            errors.append(err)
    for path, allowed in CHOICES.items():
        value = _at(cfg, path)
        if value is not MISSING and value not in allowed:
            errors.append({"field": path, "message": f"must be one of {', '.join(allowed)}, got {value!r}"})
    return errors


def inactive(cfg: dict) -> list[str]:
    """Settings the run will ignore, so the UI can say so instead of leaving them looking live.

    :param cfg: A merged config (see :func:`merged`).
    :returns: Human-readable notes, empty when every enabled control has an effect.
    """
    notes = []
    if not cfg["rerank"]["enabled"]:
        notes.append("Reranking is off, so rerank.top_n is ignored and every fused candidate carries forward.")
    if not cfg["generate"]["enabled"]:
        if cfg["verify"]["enabled"]:
            notes.append("Generation is off, so the groundedness check has no answer to verify and will not run.")
        if cfg["cache"]["enabled"]:
            notes.append("Generation is off, so no answer is written to the semantic cache.")
    if cfg["route"]["enabled"] and cfg["route"]["apply"]:
        notes.append("Adaptive routing may override fusion weights, HyDE, decomposition, multi-query and graph "
                     "retrieval per question. Turn off 'let the route change the settings' to hold them fixed.")
    if cfg["dense"]["enabled"] and cfg["dense"]["mode"] == "hnsw" and cfg["dense"]["ef_search"] < cfg["dense"]["k"]:
        notes.append(f"ef_search {cfg['dense']['ef_search']} is below dense k {cfg['dense']['k']}, so the search runs "
                     f"at ef {cfg['dense']['k']}: HNSW cannot return k neighbours with a smaller search width.")
    if cfg["dense"]["enabled"] and cfg["dense"]["mode"] == "exact" and cfg["dense"]["compare_exact"]:
        notes.append("Exact search is selected, so there is no approximation to compare against.")
    return notes


def answer_abstains(answer: str) -> bool:
    """Whether a generated answer declined to answer, judged from its own words.

    This is a different question from the retrieval grader's verdict: the grader decides whether
    evidence was found, the answer decides what to tell the user, and either can be right when the
    other is wrong. Matching is a documented heuristic over
    :data:`raglab.prompts.ABSTENTION_MARKERS`, not a model call, so it costs nothing to run on every
    question.

    Only the **first sentence** is matched, because GENERATE_SYSTEM tells the model to lead with the
    refusal and then say what is missing. The same words later in an answer usually belong to it
    ("the policy does not contain an exemption for...") and mean the opposite of declining.

    Known limitation: an answer that opens with something else ("I checked the passages. The corpus
    does not cover this.") reads as not abstaining. The grader's verdict is reported alongside this,
    so a disagreement between the two is visible rather than hidden.

    :param answer: The generated answer, already stripped of any reasoning block.
    :returns: True when the answer opens by declining.
    """
    text = re.sub(r"\s+", " ", (answer or "").lower()).strip()
    first = re.split(r"(?<=[.!?])\s+", text, maxsplit=1)[0] if text else ""
    return any(marker in first for marker in prompts.ABSTENTION_MARKERS)


def _ranked(hits: list[tuple[int, float]]) -> list[dict]:
    return [{"i": i, "score": round(s, 5), "rank": r + 1} for r, (i, s) in enumerate(hits)]


def fuse(lists: list[dict], cfg: dict) -> list[tuple[int, float, dict]]:
    """lists: [{"retriever","query","hits":[{i,score,rank}]}] -> [(i, fused, contributions)]"""
    w = cfg["weights"]
    total: dict[int, float] = {}
    parts: dict[int, dict] = {}
    if cfg["method"] == "rrf":
        for l in lists:
            for h in l["hits"]:
                add = w.get(l["retriever"], 1.0) / (cfg["rrf_k"] + h["rank"])
                total[h["i"]] = total.get(h["i"], 0.0) + add
                p = parts.setdefault(h["i"], {})
                p[l["retriever"]] = round(p.get(l["retriever"], 0.0) + add, 5)
    else:  # weighted: min-max normalise each retriever (best score across query variants), then weighted sum
        best: dict[str, dict[int, float]] = {}
        for l in lists:
            for h in l["hits"]:
                b = best.setdefault(l["retriever"], {})
                b[h["i"]] = max(b.get(h["i"], -1e9), h["score"])
        for name, scores in best.items():
            lo, hi = min(scores.values()), max(scores.values())
            for i, s in scores.items():
                val = w.get(name, 1.0) * ((s - lo) / (hi - lo) if hi > lo else 1.0)
                total[i] = total.get(i, 0.0) + val
                parts.setdefault(i, {})[name] = round(val, 5)
    return [(i, s, parts[i]) for i, s in sorted(total.items(), key=lambda x: -x[1])][:cfg["top_n"]]


def mmr(cands: list[int], rel: dict[int, float], matrix: np.ndarray, lam: float, n: int) -> list[int]:
    if not cands:
        return []
    vals = np.array([rel[c] for c in cands], dtype=np.float32)
    span = float(vals.max() - vals.min())
    nrel = {c: (rel[c] - float(vals.min())) / span if span else 1.0 for c in cands}
    chosen: list[int] = []
    pool = list(cands)
    while pool and len(chosen) < n:
        def gain(c):
            red = max((float(matrix[c] @ matrix[s]) for s in chosen), default=0.0)
            return lam * nrel[c] - (1 - lam) * red
        pick = max(pool, key=gain)
        chosen.append(pick)
        pool.remove(pick)
    return chosen


class Pipeline:
    def __init__(self, rt: Runtime, models: dict, cfg: dict | None = None, store=None, history: list[dict] | None = None,
                 rules: dict | None = None):
        """:param rules: Normalisation rules to apply. When given they are used as-is; a batch job
        passes the set it captured at the start so editing them mid-run cannot change what half the
        questions were measured under. When omitted they are read from the store per query, which is
        what an interactive question wants."""
        self.rt, self.models, self.cfg = rt, models, merged(cfg)
        self.store, self.history = store, history or []
        self._frozen_rules = rules
        self.rec = Recorder()
        self.seen: set[int] = set()
        self.usage = {"tokens_in": 0, "tokens_out": 0, "cost": 0.0}
        # The cache namespace covers everything that changes the answer: the settings (bar the cache
        # itself), which models wrote it, and the conversation it was written in. Leaving the last
        # two out let a follow-up like "is it free?" match an entry from an unrelated thread, and
        # survived a change of generator.
        hashed = {k: v for k, v in self.cfg.items() if k != "cache"}
        roles = {r: {k: cfg_.get(k) for k in ("provider", "via", "base_url", "model")}
                 for r, cfg_ in sorted((self.models or {}).items())}
        turns = [{"q": h.get("q", ""), "a": h.get("a", "")} for h in self.history]
        # The rules live outside the config, in the kv table, so editing them in the Normalisation
        # lab changed what a query retrieves while leaving the cache namespace untouched.
        rules = self._rules() if self.cfg["normalize"]["enabled"] else {}
        self.cfg_hash = hashlib.sha1(json.dumps(
            {"pipeline": hashed, "models": roles, "history": turns, "rules": rules}, sort_keys=True).encode()).hexdigest()[:12]

    def _rules(self) -> dict:
        """The normalisation rules that would be applied: this index's own, else the shared set."""
        if self._frozen_rules is not None:
            return self._frozen_rules
        if self.store is None:
            return {}
        for key in (norm.rules_key(self.rt.id), "normalization"):
            rows = self.store.q("SELECT value FROM kv WHERE key=?", (key,))
            if rows:
                return json.loads(rows[0]["value"])
        return {}

    # -- helpers
    def _chunks(self, ids) -> dict:
        out = {}
        for i in ids:
            if i not in self.seen:
                self.seen.add(i)
                c = self.rt.chunks[i]
                out[i] = {"text": c["text"], "context": c["context"], "doc_id": c["doc_id"], "meta": c["meta"],
                          "start": c["start"], "end": c["end"]}
        return out

    def _event(self, stage: str, title: str, t0: float, data: dict, ids=()) -> dict:
        calls = self.rec.drain()
        use = {"tokens_in": sum(c.get("tokens_in", 0) for c in calls), "tokens_out": sum(c.get("tokens_out", 0) for c in calls),
               "cost": round(sum(c.get("cost", 0.0) for c in calls), 6)}
        for k in use:
            self.usage[k] = round(self.usage[k] + use[k], 6)
        return {"type": "stage", "stage": stage, "title": title, "ms": round((time.perf_counter() - t0) * 1000),
                "data": data, "calls": calls, "usage": use, "chunks": self._chunks(ids)}

    def _done(self, result: dict, t_all: float) -> dict:
        result.update(self.usage)
        return {"type": "done", "result": result, "usage": self.usage, "total_ms": round((time.perf_counter() - t_all) * 1000)}

    async def _retrieve(self, queries: list[dict], allowed, cfg: dict) -> tuple[list[dict], dict]:
        """queries: [{"label","text","kind": original|variant|hyde}]"""
        rt, lists, extra = self.rt, [], {}
        dense_q = [q for q in queries] if cfg["dense"]["enabled"] else []
        if dense_q:
            vecs = await embed(self.models["embedding"],
                               [q["text"] for q in dense_q], "query", self.rec)
            # HyDE passages live in document space: embed them the way documents were embedded
            hy = [n for n, q in enumerate(dense_q) if q["kind"] == "hyde"]
            if hy:
                vecs[hy] = await embed(self.models["embedding"], [dense_q[n]["text"] for n in hy], "doc", self.rec)
            extra["projection"] = [{"label": q["label"], "kind": q["kind"], "xy": xy}
                                   for q, xy in zip(dense_q, rt.project(vecs))]
            recalls, infos = [], []
            for q, v in zip(dense_q, vecs):
                hits, recall, info = rt.vectors.search(v, cfg["dense"]["k"], cfg["dense"]["mode"],
                                                       cfg["dense"]["ef_search"], allowed, cfg["dense"]["compare_exact"])
                if recall is not None:
                    recalls.append(recall)
                infos.append(info)
                lists.append({"retriever": "dense", "query": q["label"], "hits": _ranked(hits)})
            if recalls:
                extra["hnsw_recall_vs_exact"] = round(float(np.mean(recalls)), 4)
            if infos:
                # Requested against effective, and search cost apart from the cost of checking it,
                # so an ef sweep cannot silently plot the same run twice or bill verification to it.
                extra["dense_search"] = {
                    "mode": cfg["dense"]["mode"], "ef_requested": infos[0]["ef_requested"],
                    "ef_effective": infos[0]["ef_effective"],
                    "ef_raised_to_k": any(i["ef_requested"] is not None and i["ef_effective"] > i["ef_requested"] for i in infos),
                    "exact_fallback": sum(i["exact_fallback"] for i in infos),
                    "search_ms": round(sum(i["search_ms"] for i in infos), 3),
                    "verify_ms": round(sum(i["verify_ms"] for i in infos), 3)}
        if cfg["bm25"]["enabled"]:
            for q in queries:
                if q["kind"] == "hyde":
                    continue
                hits, idf = rt.bm25.search(q["text"], cfg["bm25"]["k"], cfg["bm25"]["k1"], cfg["bm25"]["b"], allowed)
                lists.append({"retriever": "bm25", "query": q["label"], "hits": _ranked(hits), "idf": idf})
        if cfg["graph"]["enabled"] and rt.graph is not None:
            mentions, links = await rt.graph.link(queries[0]["text"], self.models["router"],
                                                  self.models["embedding"], self.rec)
            hits, sub = rt.graph.search(links, cfg["graph"]["k"], cfg["graph"]["alpha"], allowed)
            lists.append({"retriever": "graph", "query": queries[0]["label"], "hits": _ranked(hits)})
            extra["graph"] = {"mentions": mentions, "links": links, "subgraph": sub}
        elif cfg["graph"]["enabled"]:
            extra["graph"] = {"skipped": "This index was built without a knowledge graph."}
        if rt.removed.any():
            extra["removed_documents"] = {
                "documents": len(rt.removed_docs), "chunks": int(rt.removed.sum()),
                "note": f"{len(rt.removed_docs)} document(s) in this index were removed from the corpus. Their "
                        f"{int(rt.removed.sum())} chunk(s) are excluded from retrieval. Rebuild the index to drop "
                        f"the stored copies."}
        if cfg["freshness"]["enabled"] and rt.superseded.any():  # metadata filter: drop superseded documents
            gone: dict[int, str] = {}
            for l in lists:
                for h in l["hits"]:
                    if rt.superseded[h["i"]]:
                        gone[h["i"]] = str(rt.chunks[h["i"]]["meta"].get("superseded_by"))
                l["hits"] = [{**h, "rank": r + 1} for r, h in enumerate(h for h in l["hits"] if not rt.superseded[h["i"]])]
            extra["filtered_out"] = [{"i": i, "reason": f"Superseded by {by}"} for i, by in gone.items()]
        return lists, extra

    async def _rerank(self, query: str, fused: list[tuple[int, float, dict]], cfg: dict) -> list[dict]:
        ids = [i for i, _, _ in fused]
        if not cfg["rerank"]["enabled"] or not ids:
            # Rerank off means "do not reorder", not "keep eight": truncating here by rerank.top_n
            # would silently shrink the candidate pool under a control the UI has hidden.
            return [{"i": i, "score": round(s, 5), "rank": r + 1, "from_rank": r + 1}
                    for r, (i, s, _) in enumerate(fused)]
        texts = ["\n".join(x for x in (self.rt.chunks[i]["context"], self.rt.chunks[i]["text"]) if x) for i in ids]
        scores = await rerank(self.models["reranker"], query, texts, self.rec)
        order = sorted(range(len(ids)), key=lambda n: -scores[n])[:cfg["rerank"]["top_n"]]
        return [{"i": ids[n], "score": round(scores[n], 5), "rank": r + 1, "from_rank": n + 1}
                for r, n in enumerate(order)]

    async def _grade(self, query: str, ids: list[int]) -> dict[int, dict]:
        sem = asyncio.Semaphore(4)

        async def one(i):
            async with sem:
                out = await chat_json(self.models["router"], [{"role": "user", "content": prompts.GRADE.format(
                    query=query, passage=self.rt.chunks[i]["text"][:2500])}], task="grade", rec=self.rec,
                    max_tokens=120, default={}) or {}
            g = str(out.get("grade", "partial")).lower() if isinstance(out, dict) else "partial"
            return i, {"grade": g if g in ("relevant", "partial", "irrelevant") else "partial",
                       "reason": str(out.get("reason", ""))[:200] if isinstance(out, dict) else ""}
        return dict(await asyncio.gather(*[one(i) for i in ids]))

    # -- main
    async def run(self, query: str) -> AsyncIterator[dict]:
        rt, cfg, M = self.rt, self.cfg, self.models
        t_all = time.perf_counter()
        # abstained is the effective decision; the two components below say who made it, so that
        # turning the grader off cannot masquerade as a generator that answered anyway.
        result = {"query": query, "lists": [], "fused": [], "reranked": [], "context": [], "answer": "",
                  "abstained": False, "crag_abstained": None, "answer_abstained": None}
        search_query, filters, direct = query, [], False
        qvec = None

        # 0a. semantic cache: a close-enough earlier question under the same settings skips everything
        if cfg["cache"]["enabled"] and self.store is not None:
            t0 = time.perf_counter()
            qvec = (await embed(M["embedding"], [query], "query", self.rec))[0]
            rows = self.store.q("SELECT query, embedding, answer FROM qcache WHERE index_id=? AND cfg_hash=?", (rt.id, self.cfg_hash))
            best, sim = None, 0.0
            for r in rows:
                s_ = float(np.frombuffer(r["embedding"], dtype=np.float32) @ qvec)
                if s_ > sim:
                    best, sim = r, s_
            hit = best is not None and sim >= cfg["cache"]["threshold"]
            yield self._event("cache", "Semantic cache", t0, {"hit": hit, "similarity": round(sim, 4), "entries": len(rows),
                                                             "threshold": cfg["cache"]["threshold"], "matched": best["query"] if best else None})
            if hit:
                result["answer"] = best["answer"]
                yield {"type": "token", "text": best["answer"]}
                yield self._done(result, t_all)
                return

        # 0b. conversational rewriting: make a follow-up standalone
        if cfg["condense"]["enabled"] and self.history:
            t0 = time.perf_counter()
            hist = "\n".join(f"User: {h['q']}\nAssistant: {h['a'][:400]}" for h in self.history[-4:])
            out = await chat_json(M["router"], [{"role": "user", "content": prompts.CONDENSE.format(history=hist, query=query)}],
                                  task="condense", rec=self.rec, max_tokens=150, default={}) or {}
            rq = str(out.get("query", "")).strip() if isinstance(out, dict) else ""
            if rq:
                search_query = rq
            yield self._event("condense", "Conversational rewriting", t0, {"turns": len(self.history), "original": query, "standalone": search_query})

        # 0c. rule-based normalisation (rules live in the Normalisation lab)
        if cfg["normalize"]["enabled"]:
            t0 = time.perf_counter()
            rules = self._rules() or norm.EMPTY_RULES
            res = norm.normalize(search_query, rules, rt.vocab)
            search_query = res["normalized"]
            yield self._event("normalize", "Query normalisation", t0, res)

        # 1. route / intent gate
        if cfg["route"]["enabled"]:
            t0 = time.perf_counter()
            corpus = "\n".join(f"- {t}" for t in rt.titles[:40]) or "- (untitled)"
            out = await chat_json(M["router"], [{"role": "user", "content": prompts.ROUTE.format(corpus=corpus, query=search_query)}],
                                  task="route", rec=self.rec, max_tokens=2048, default={}) or {}
            # A small router that runs past max_tokens leaves the object unclosed, and extract_json
            # then parses a list nested inside it instead. Treat that as no answer, like the others.
            if not isinstance(out, dict):
                out = {}
            intent = str(out.get("intent", "conceptual"))
            changes = []
            if cfg["route"]["apply"]:
                if intent == "chitchat" or out.get("needs_retrieval") is False:
                    direct = True
                    changes.append("skip retrieval, answer directly")
                elif intent == "lookup":
                    cfg["fusion"]["weights"]["bm25"] *= 1.5
                    cfg["hyde"]["enabled"] = False
                    changes += ["BM25 weight x1.5", "HyDE off"]
                elif intent == "conceptual":
                    cfg["fusion"]["weights"]["dense"] *= 1.5
                    changes.append("dense weight x1.5")
                elif intent == "multi_hop":
                    cfg["multi_query"]["enabled"] = True
                    changes.append("multi-query on")
                    cfg["decompose"]["enabled"] = True
                    changes.append("decomposition on")
                    if rt.graph is not None:
                        cfg["graph"]["enabled"] = True
                        changes.append("graph retrieval on")
                elif intent == "global" and rt.communities:
                    cfg["communities"]["enabled"] = True
                    changes.append("community reports on")
            yield self._event("route", "Route and intent gate", t0, {**out, "applied": changes})

        if direct:
            t0 = time.perf_counter()
            if not cfg["generate"]["enabled"]:
                # Generation off has to mean off on every path, or a sweep that disables it to
                # measure retrieval still pays for a model call and scores an empty run.
                yield self._event("generate", "Direct reply skipped (generation off)", t0,
                                  {"answer": "", "skipped": "The router chose to answer without retrieval, but "
                                                            "generation is disabled, so nothing was written."})
                yield self._done(result, t_all)
                return
            answer = ""
            async for tok in chat_stream(M["generator"], [{"role": "system", "content": prompts.DIRECT_SYSTEM},
                                                          {"role": "user", "content": query}], task="direct", rec=self.rec):
                answer += tok
                yield {"type": "token", "text": tok}
            result["answer"] = strip_think(answer) or "Hello. Ask me something about the indexed corpus."
            result["answer_abstained"] = answer_abstains(result["answer"])
            result["abstained"] = bool(result["answer_abstained"])
            yield self._event("generate", "Direct reply (no retrieval)", t0, {"answer": result["answer"]})
            yield self._done(result, t_all)
            return

        # 2. self-query: pull metadata filters out of the question
        # Chunks whose document was deleted are excluded before anything else looks at them. This is
        # not a setting: an index keeps its own copy of every chunk, so without it "Remove" left the
        # text being retrieved and answered from, which is the opposite of what the button says.
        allowed = rt.live_mask()
        if cfg["self_query"]["enabled"]:
            t0 = time.perf_counter()
            schema = rt.schema()
            out = await chat_json(M["router"], [{"role": "user", "content": prompts.SELF_QUERY.format(
                schema=json.dumps(schema, ensure_ascii=False), query=search_query)}], task="self_query", rec=self.rec,
                max_tokens=300, default={}) or {}
            raw = out.get("filters", []) if isinstance(out, dict) else []
            filters = [f for f in raw if isinstance(f, dict) and f.get("field") in schema]
            rejected = [f for f in raw if f not in filters]
            by_filter = rt.allowed(filters)
            live = rt.live_mask()
            allowed = by_filter if live is None else (live if by_filter is None else (by_filter & live))
            note, relaxed = "", False
            if allowed is not None and not allowed.any():
                # Dropping the filters here quietly widens the search to the whole corpus and the
                # answer then cites documents the question excluded. Honour the filter by default;
                # relaxing it is a setting, not a silent fallback.
                if cfg["self_query"].get("relax_empty"):
                    # Back to the whole corpus, which still excludes deleted documents.
                    allowed, note, relaxed = rt.live_mask(), "Filters matched nothing, so they were relaxed to the whole corpus.", True
                else:
                    note = "Filters matched nothing in this index, so there is no evidence to retrieve. " \
                           "Turn on 'relax filters that match nothing' to search the whole corpus instead."
            if isinstance(out, dict) and str(out.get("query", "")).strip():
                search_query = str(out["query"]).strip()
            yield self._event("self_query", "Self-query metadata filters", t0, {
                "schema": schema, "search_query": search_query, "filters": filters, "rejected": rejected,
                "matching_chunks": int(allowed.sum()) if allowed is not None else (0 if note and not relaxed else len(rt.chunks)),
                "total_chunks": len(rt.chunks), "note": note, "relaxed": relaxed,
                "empty": bool(note) and not relaxed})

        # 3. query transformation
        queries = [{"label": "original", "text": search_query, "kind": "original"}]
        if cfg["multi_query"]["enabled"] or cfg["hyde"]["enabled"] or cfg["decompose"]["enabled"] or cfg["step_back"]["enabled"]:
            t0 = time.perf_counter()

            async def subs():
                if not cfg["decompose"]["enabled"]:
                    return []
                out = await chat_json(M["router"], [{"role": "user", "content": prompts.DECOMPOSE.format(
                    n=cfg["decompose"]["n"], query=search_query)}], task="decompose", rec=self.rec, default={}) or {}
                qs = [str(q).strip() for q in (out.get("sub_questions", []) if isinstance(out, dict) else []) if isinstance(q, str) and q.strip()]
                return qs[:cfg["decompose"]["n"]] if len(qs) > 1 else []

            async def step_back():
                if not cfg["step_back"]["enabled"]:
                    return ""
                out = await chat_json(M["router"], [{"role": "user", "content": prompts.STEP_BACK.format(query=search_query)}],
                                      task="step_back", rec=self.rec, max_tokens=120, default={}) or {}
                return str(out.get("query", "")).strip() if isinstance(out, dict) else ""

            async def variants():
                if not cfg["multi_query"]["enabled"]:
                    return []
                out = await chat_json(M["router"], [{"role": "user", "content": prompts.MULTI_QUERY.format(
                    n=cfg["multi_query"]["n"], query=search_query)}], task="multi_query", rec=self.rec, default={}) or {}
                qs = out.get("queries", []) if isinstance(out, dict) else []
                return [str(q) for q in qs if isinstance(q, str) and q.strip()][:cfg["multi_query"]["n"]]

            async def hyde():
                if not cfg["hyde"]["enabled"]:
                    return ""
                return strip_think(await chat(M["generator"], [{"role": "user", "content": prompts.HYDE.format(
                    query=search_query)}], task="hyde", rec=self.rec, temperature=0.3, max_tokens=300))

            vs, hy, sq, sb = await asyncio.gather(variants(), hyde(), subs(), step_back())
            queries += [{"label": f"sub-question {n + 1}", "text": v, "kind": "sub"} for n, v in enumerate(sq)]
            if sb:
                queries.append({"label": "step-back", "text": sb, "kind": "stepback"})
            queries += [{"label": f"variant {n + 1}", "text": v, "kind": "variant"} for n, v in enumerate(vs)]
            if hy:
                queries.append({"label": "HyDE passage", "text": hy, "kind": "hyde"})
            yield self._event("transform", "Query transformation", t0, {"queries": queries})

        # 4. retrieve
        t0 = time.perf_counter()
        lists, extra = await self._retrieve(queries, allowed, cfg)
        result["lists"] = lists
        yield self._event("retrieve", "Retrieval", t0, {"lists": lists, **extra},
                          [h["i"] for l in lists for h in l["hits"][:12]] + [f["i"] for f in extra.get("filtered_out", [])])

        # 5. fuse
        t0 = time.perf_counter()
        fused = fuse(lists, cfg["fusion"])
        result["fused"] = [i for i, _, _ in fused]
        yield self._event("fuse", f"Fusion ({cfg['fusion']['method']})", t0, {
            "weights": cfg["fusion"]["weights"],
            "fused": [{"i": i, "score": round(s, 5), "rank": r + 1, "parts": p} for r, (i, s, p) in enumerate(fused)]},
            [i for i, _, _ in fused])

        # 6. rerank
        t0 = time.perf_counter()
        reranked = await self._rerank(search_query, fused, cfg)
        result["reranked"] = [h["i"] for h in reranked]
        yield self._event("rerank", "Rerank" if cfg["rerank"]["enabled"] else "Rerank (off: fused order kept)", t0,
                          {"reranked": reranked, "model": M["reranker"].get("model"),
                           "provider": M["reranker"].get("provider")}, [h["i"] for h in reranked])

        # 7. corrective RAG
        keep = [h["i"] for h in reranked]
        scores = {h["i"]: h["score"] for h in reranked}
        abstain = False
        refined: dict[int, str] = {}
        if cfg["crag"]["enabled"] and keep:
            t0 = time.perf_counter()
            grades = await self._grade(search_query, keep)
            rounds = [{"query": search_query, "grades": [{"i": i, **grades[i]} for i in keep]}]

            def verdict(g):
                vals = [x["grade"] for x in g.values()]
                return "correct" if "relevant" in vals else "ambiguous" if "partial" in vals else "incorrect"
            v = verdict(grades)
            retries = 0
            new_ids: list[int] = []
            while v != "correct" and retries < cfg["crag"]["max_retries"]:
                retries += 1
                seen = "; ".join(dict.fromkeys(rt.chunks[i]["meta"].get("section") or rt.chunks[i]["meta"].get("title", "")
                                               for i in keep))[:400]
                out = await chat_json(M["router"], [{"role": "user", "content": prompts.REWRITE.format(
                    query=search_query, seen=seen)}], task="rewrite", rec=self.rec, max_tokens=150, default={}) or {}
                rq = str(out.get("query", "")).strip() if isinstance(out, dict) else ""
                if not rq:
                    break
                wide = json.loads(json.dumps(cfg))
                wide["bm25"]["k"], wide["dense"]["k"] = cfg["bm25"]["k"] * 2, cfg["dense"]["k"] * 2
                l2, _ = await self._retrieve([{"label": "corrective", "text": rq, "kind": "variant"}], allowed, wide)
                r2 = await self._rerank(rq, [f for f in fuse(l2, cfg["fusion"]) if f[0] not in grades], cfg)
                g2 = await self._grade(search_query, [h["i"] for h in r2])
                grades.update(g2)
                scores.update({h["i"]: h["score"] for h in r2})
                new_ids += [h["i"] for h in r2]
                rounds.append({"query": rq, "grades": [{"i": h["i"], **g2[h["i"]]} for h in r2]})
                v = verdict(grades)
            pool = keep + new_ids
            keep = [i for i in pool if grades[i]["grade"] == "relevant"] + \
                   ([i for i in pool if grades[i]["grade"] == "partial"] if v != "incorrect" else [])
            abstain = v == "incorrect"
            result["crag_abstained"] = abstain
            result["abstained"] = abstain  # provisional: the generated answer overrides it below
            refined_info = []
            if cfg["crag"]["refine"] and keep:  # knowledge refinement: keep only the strips that bear on the question
                sem = asyncio.Semaphore(4)

                async def refine(i):
                    text = rt.chunks[i]["text"]
                    strips = [text[a:b] for a, b in sentences(text)]
                    if len(strips) < 3:
                        return
                    async with sem:
                        out = await chat_json(M["router"], [{"role": "user", "content": prompts.REFINE.format(
                            query=search_query, strips=" ".join(f"[{n + 1}] {t}" for n, t in enumerate(strips)))}],
                            task="refine", rec=self.rec, max_tokens=120, default={}) or {}
                    ks = sorted({int(k) for k in (out.get("keep", []) if isinstance(out, dict) else []) if str(k).isdigit() and 0 < int(k) <= len(strips)})
                    if ks and len(ks) < len(strips):
                        refined[i] = " ".join(strips[k - 1] for k in ks)
                        refined_info.append({"i": i, "strips": len(strips), "kept": ks, "tokens_before": len(text) // 4, "tokens_after": len(refined[i]) // 4})
                await asyncio.gather(*[refine(i) for i in keep])
            yield self._event("crag", "Corrective RAG", t0, {
                "refined": refined_info,
                "verdict": v, "rounds": rounds, "kept": keep, "dropped": [i for i in pool if i not in keep],
                "action": {"correct": "Evidence found. Irrelevant passages dropped.",
                           "ambiguous": "Only partial evidence, even after corrective retrieval. Answer will be hedged.",
                           "incorrect": "No supporting evidence in the corpus. The generator is told to abstain."}[v]},
                pool)

        # 8. context assembly
        t0 = time.perf_counter()
        c = cfg["context"]
        picked = mmr(keep, scores, rt.matrix, c["mmr"]["lambda"], c["max_chunks"]) if c["mmr"]["enabled"] \
            else keep[:c["max_chunks"]]
        blocks, used_parents, tokens, dropped = [], set(), 0, []

        def admit(block: dict) -> bool:
            """Take a block if the budget allows it. The first block is always taken, so a single
            oversized passage yields a marked overflow rather than an empty context."""
            nonlocal tokens
            if blocks and tokens + block["tokens"] > c["token_budget"]:
                dropped.append({"i": block["i"], "title": block["title"], "tokens": block["tokens"]})
                return False
            block["overflow"] = not blocks and block["tokens"] > c["token_budget"]
            tokens += block["tokens"]
            blocks.append(block)
            return True

        if cfg["communities"]["enabled"] and rt.communities and rt.community_vecs is not None:
            if qvec is None:
                qvec = (await embed(M["embedding"], [search_query], "query", self.rec))[0]
            sims = rt.community_vecs @ qvec
            for k in np.argsort(-sims)[:cfg["communities"]["k"]]:
                com = rt.communities[int(k)]
                # Community reports are charged to the same budget as passages: they displace
                # evidence in the same prompt, so exempting them would overrun it silently.
                if not admit({"i": -1, "title": f"Community report: {com['title']}", "section": "", "text": com["summary"],
                              "expanded": False, "tokens": len(com["summary"]) // 4, "community": int(k),
                              "similarity": round(float(sims[k]), 4)}):
                    break
        for i in picked:
            ch = rt.chunks[i]
            text, expanded = refined.get(i, ch["text"]), False
            if i in refined:
                pass
            elif c["expand_parents"] and ch["parent"] is not None and ch["parent"] in rt.parents:
                if ch["parent"] in used_parents:
                    continue
                used_parents.add(ch["parent"])
                text, expanded = rt.parents[ch["parent"]]["text"], True
            # doc_id/doc_key travel with the block so evidence can be scored on the delivered text
            # without going back to the chunk it came from.
            if not admit({"i": i, "title": ch["meta"].get("title", ""), "section": ch["meta"].get("section", ""),
                          "text": text, "expanded": expanded, "refined": i in refined, "tokens": len(text) // 4,
                          "doc_id": ch["doc_id"], "doc_key": ch["meta"].get("doc_key")}):
                break
        if c["order"] == "edges":  # strongest evidence at the start and end (lost-in-the-middle mitigation)
            blocks = blocks[0::2] + blocks[1::2][::-1]
        for n, b in enumerate(blocks):
            b["n"] = n + 1
        result["context"] = [b["i"] for b in blocks if b["i"] >= 0]
        # What the budget does not cover, stated rather than implied: the instructions and the
        # question also occupy the window, and the answer needs room after them.
        overhead = (len(prompts.GENERATE_SYSTEM) + len(prompts.GENERATE_USER) + len(query)) // 4
        yield self._event("context", "Context assembly", t0, {
            "blocks": blocks, "tokens": tokens, "budget": c["token_budget"], "dropped": dropped,
            "prompt_overhead_tokens": overhead, "output_allowance_tokens": cfg["generate"]["max_tokens"],
            "estimated_total_tokens": tokens + overhead + cfg["generate"]["max_tokens"]}, result["context"])

        # 9. generate
        if cfg["generate"]["enabled"]:
            t0 = time.perf_counter()
            ctx = "\n\n".join(f"[{b['n']}] ({b['title']}{' > ' + b['section'] if b['section'] else ''})\n{b['text']}"
                              for b in blocks) or "(no passages survived retrieval and grading)"
            system = prompts.GENERATE_SYSTEM + ("\nThe retrieval grader found no supporting evidence. Say so." if abstain else "")
            messages = [{"role": "system", "content": system},
                        {"role": "user", "content": prompts.GENERATE_USER.format(context=ctx, query=query)}]
            answer = ""
            async for tok in chat_stream(M["generator"], messages, temperature=cfg["generate"]["temperature"],
                                         max_tokens=cfg["generate"]["max_tokens"], task="generate", rec=self.rec):
                answer += tok
                yield {"type": "token", "text": tok}
            answer = strip_think(answer)
            cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", answer) if 0 < int(n) <= len(blocks)})
            result["answer"] = answer
            result["context_text"] = ctx
            result["answer_abstained"] = answer_abstains(answer)
            result["abstained"] = result["answer_abstained"]  # what the user was told wins over the grader
            yield self._event("generate", "Generation", t0, {
                "answer": answer, "cited": cited, "uncited": [b["n"] for b in blocks if b["n"] not in cited],
                "model": M["generator"].get("model"), "messages": messages,
                "answer_abstained": result["answer_abstained"], "crag_abstained": result["crag_abstained"]})

            # 10. groundedness: check every claim of the answer against the passages it was given
            if cfg["verify"]["enabled"] and blocks and answer:
                t0 = time.perf_counter()
                out = await chat_json(M["router"], [{"role": "user", "content": prompts.VERIFY.format(context=ctx[:9000], answer=answer)}],
                                      task="verify", rec=self.rec, max_tokens=900, default={}) or {}
                claims = [{"claim": str(c_.get("claim", ""))[:300], "verdict": "supported" if str(c_.get("verdict", "")).lower().startswith("sup") else "unsupported",
                           "passage": c_.get("passage")} for c_ in (out.get("claims", []) if isinstance(out, dict) else []) if isinstance(c_, dict)]
                share = sum(c_["verdict"] == "supported" for c_ in claims) / len(claims) if claims else None
                result["groundedness"] = share
                yield self._event("verify", "Groundedness check", t0, {"claims": claims, "supported_share": share})

            if cfg["cache"]["enabled"] and self.store is not None and qvec is not None and answer \
                    and not result["abstained"]:
                self.store.x("INSERT INTO qcache (index_id, cfg_hash, query, embedding, answer, created) VALUES (?,?,?,?,?,?)",
                             (rt.id, self.cfg_hash, query, qvec.astype(np.float32).tobytes(), answer, time.time()))

        yield self._done(result, t_all)


class IncompatibleIndex(RuntimeError):
    """This index cannot be queried with the models currently configured."""


def controlled(cfg: dict | None) -> dict:
    """Pin the settings that would otherwise vary per question, for "change one thing" experiments.

    The semantic cache is bypassed so a repeat question cannot return someone else's timing, and
    adaptive routing is stopped from rewriting fusion weights, HyDE, decomposition, multi-query and
    graph retrieval per question — any of which can be the very knob under test.

    :param cfg: A partial pipeline config, as sent by the UI.
    :returns: A copy with those settings pinned; the caller's dict is untouched.
    """
    out = copy.deepcopy(cfg or {})
    out["cache"] = {**out.get("cache", {}), "enabled": False}
    out["route"] = {**out.get("route", {}), "apply": False}
    return out


async def execute(rt: Runtime, models: dict, query: str, cfg: dict | None = None, store=None,
                  control: bool = False, rules: dict | None = None) -> dict:
    """Run without streaming; returns {"result", "stages", "total_ms", "config"}.

    :param control: Apply :func:`controlled` first. Off by default, so an evaluation measures the
        pipeline as configured — routing included, which someone may be evaluating on purpose.
        Sweeps turn it on, because there the point is to vary one setting and nothing else.
    """
    # Checked here rather than only at the API boundary: a script, a test or a new endpoint reaching
    # the pipeline directly would otherwise get neighbours that look plausible and mean nothing.
    if (problems := compatibility(rt, models)):
        raise IncompatibleIndex(" ".join(problems))
    stages, final = [], {}
    cfg = controlled(cfg) if control else {**(cfg or {}), "cache": {"enabled": False}}
    pipe = Pipeline(rt, models, cfg, store=store, rules=rules)
    async for ev in pipe.run(query):
        if ev["type"] == "stage":
            stages.append(ev)
        elif ev["type"] == "done":
            final = ev
    # pipe.cfg is the merged config after routing may have rewritten it, which is what actually ran.
    return {"result": final.get("result", {}), "stages": stages, "total_ms": final.get("total_ms", 0),
            "config": pipe.cfg}
