"""Index build + in-memory runtime. An index = corpus snapshot x chunking config x embedding model,
so several indexes can sit side by side and be compared on the same questions."""
from __future__ import annotations
import asyncio, hashlib, json, re, time, traceback
import os
from collections import Counter, OrderedDict
from collections.abc import Iterable
import numpy as np
from . import graph as graphmod, normalize, prompts
from .bm25 import BM25
from .chunking import DEFAULT_CHUNKING, chunk_text
from .config import DATA_DIR
from .pipeline_bounds import MISSING, at as _at, number_error
from .providers import LLMCache, chat, chat_json, embed, fingerprint, strip_think
from .store import Store
from .vectors import DEFAULT_HNSW, VectorIndex, pca2d

DEFAULT_INDEX = {"chunking": DEFAULT_CHUNKING, "hnsw": DEFAULT_HNSW, "dedupe": {"enabled": True, "threshold": 0.5},
                 "graph": {"enabled": False, "communities": False}, "doc_metadata": {"enabled": False}, "doc_ids": None}


# Bounds for the build-time knobs, mirroring raglab.pipeline.BOUNDS for the query-time ones.
# Without these a negative chunk size or M=0 was accepted with HTTP 200 and then killed the build
# thread with a raw pybind11 TypeError, leaving an index stuck at status 'failed' with no
# explanation a reader could act on.
INDEX_BOUNDS: dict[str, tuple[float, float, bool]] = {
    "chunking.size": (20, 8000, True),
    "chunking.overlap": (0, 4000, True),
    "chunking.semantic.buffer": (0, 10, True),
    "chunking.semantic.percentile": (1, 99, True),
    "chunking.semantic.min_tokens": (10, 4000, True),
    "chunking.semantic.max_tokens": (20, 8000, True),
    "chunking.parent_child.parent_size": (50, 20000, True),
    "hnsw.M": (2, 128, True),
    "hnsw.ef_construction": (8, 4000, True),
    "dedupe.threshold": (0.0, 1.0, False),
}

INDEX_CHOICES = {"chunking.strategy": ("fixed", "sentence", "recursive", "markdown", "semantic")}


def validate_index(cfg: dict) -> list[dict]:
    """Check a merged index config before a build is started.

    :param cfg: A merged index config (see :func:`merged_config`).
    :returns: One ``{"field", "message"}`` per problem, empty when the build can run.
    """
    errors = []
    for path, (lo, hi, is_int) in INDEX_BOUNDS.items():
        value = _at(cfg, path)
        if value is MISSING:
            continue
        if (err := number_error(path, value, lo, hi, is_int)):
            errors.append(err)
    for path, allowed in INDEX_CHOICES.items():
        value = _at(cfg, path)
        if value is not MISSING and value not in allowed:
            errors.append({"field": path, "message": f"must be one of {', '.join(allowed)}, got {value!r}"})
    size, overlap = _at(cfg, "chunking.size"), _at(cfg, "chunking.overlap")
    if isinstance(size, (int, float)) and isinstance(overlap, (int, float)) and overlap >= size:
        # Every chunk would start inside the previous one's tail and the walk would not advance.
        errors.append({"field": "chunking.overlap", "message": f"must be smaller than chunking.size ({size}), got {overlap}"})
    doc_ids = cfg.get("doc_ids")
    if doc_ids is not None:
        if not isinstance(doc_ids, list) or any(not isinstance(d, str) for d in doc_ids):
            errors.append({"field": "doc_ids", "message": "must be a list of document ids, or null for the whole corpus"})
        elif not doc_ids:
            # Distinct from null on purpose. An empty selection is a question with an empty answer,
            # not an absent filter, and building the whole corpus from it is the wrong answer.
            errors.append({"field": "doc_ids", "message": "selects no documents. Choose a collection that has "
                                                          "documents in it, or leave it out to index the whole corpus."})
    lo_t, hi_t = _at(cfg, "chunking.semantic.min_tokens"), _at(cfg, "chunking.semantic.max_tokens")
    if isinstance(lo_t, (int, float)) and isinstance(hi_t, (int, float)) and lo_t >= hi_t:
        errors.append({"field": "chunking.semantic.min_tokens",
                       "message": f"must be smaller than max_tokens ({hi_t}), got {lo_t}"})
    return errors


def _entities_per_chunk(graph: dict) -> dict[int, list[str]]:
    """Which entities each chunk contributed, for reporting how many passages were too wide."""
    out: dict[int, list[str]] = {}
    for name, node in graph["nodes"].items():
        for c in node["chunks"]:
            out.setdefault(c, []).append(name)
    return out


def _shingles(text: str, n: int = 6) -> set[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {" ".join(words[i:i + n]) for i in range(max(1, len(words) - n + 1))}


def _version_key(doc: dict) -> tuple:
    """How recent a document claims to be, for choosing which of two near-duplicates to keep.

    Reads the metadata the corpora already carry: an explicit ``version``, then ``date``, then
    ``year``. A document nothing marks as superseded outranks one that is.

    :param doc: A document row with its parsed ``metadata``.
    :returns: A sort key, higher meaning newer.
    """
    meta = doc.get("metadata") or {}
    raw = str(meta.get("version", "") or "")
    version = tuple(int(n) for n in re.findall(r"\d+", raw)[:3]) or (0,)
    date = str(meta.get("date", "") or "")
    year = meta.get("year")
    return (0 if meta.get("superseded_by") else 1, version, date, int(year) if isinstance(year, int) else 0)


def dedupe(rows: list[dict], threshold: float) -> tuple[list[dict], list[dict]]:
    """Drop a chunk whose 6-word shingles overlap a chunk of ANOTHER document by Jaccard > threshold.

    Which copy survives is decided by :func:`_version_key`, not by ingestion order. Keeping the
    first-seen copy silently discarded revised policies that resemble the version they replace, and
    the freshness filter then removed the superseded original too, so the evidence disappeared
    entirely. Every removal records the document it lost to, so the trade is inspectable.

    :param rows: Chunk rows, each carrying its source ``doc``.
    :param threshold: Jaccard overlap above which two chunks count as near-duplicates.
    :returns: ``(kept, removed)``; ``removed`` entries name both sides and why one won.
    """
    order = sorted(range(len(rows)), key=lambda n: (_version_key(rows[n]["doc"]), n), reverse=True)
    kept: list[dict] = []
    sigs: list[set[str]] = []
    removed_at: dict[int, dict] = {}
    for n in order:  # newest first, so the copy that survives is the current one
        r = rows[n]
        sig = _shingles(r["text"])
        dup = next((k for k, s in zip(kept, sigs) if k["doc"]["id"] != r["doc"]["id"]
                    and len(sig & s) / max(len(sig | s), 1) > threshold), None)
        if dup is not None:
            removed_at[n] = {"title": r["doc"]["title"], "duplicate_of": dup["doc"]["title"], "text": r["text"][:160],
                             "kept_because": "newer version" if _version_key(dup["doc"]) > _version_key(r["doc"])
                             else "first of equal versions"}
        else:
            kept.append(r)
            sigs.append(sig)
    keep_ids = {id(k) for k in kept}
    return ([r for r in rows if id(r) in keep_ids],  # back into corpus order, which chunk ords follow
            [removed_at[n] for n in sorted(removed_at)])

JOBS: dict[str, dict] = {}
_RUNTIME: "OrderedDict[str, Runtime]" = OrderedDict()


def mark_stopped(store: Store, iid: str) -> None:
    """Fail an index whose build was stopped, whether it was running or still queued.

    :param store: Where the index lives.
    :param iid: The index id. One that has left 'building' keeps its status, so a stop that lands
        as a build completes cannot turn a good index into a failed one.
    """
    store.x("UPDATE indexes SET status='failed', stats=? WHERE id=? AND status='building'",
            (json.dumps({"error": "Stopped before this index finished building. Build it again when ready."}), iid))


def merged_config(cfg: dict | None) -> dict:
    cfg = cfg or {}
    out = json.loads(json.dumps(DEFAULT_INDEX))
    for k, v in cfg.items():
        out[k] = {**out[k], **v} if isinstance(out.get(k), dict) and isinstance(v, dict) else v
    return out


# Generic words that carry no information in an embedding model id, so they are dropped
# when shortening it for an index name.
_GENERIC_MODEL_WORDS = frozenset({"text", "embedding", "embeddings", "embed", "all", "general", "base", "model"})


def model_tag(model: str) -> str:
    """Shorten an embedding model id to one readable token for use in an index name.

    Drops any provider prefix and version tag, then the leading generic words:
    ``nomic-embed-text`` -> ``nomic``, ``bge-m3`` -> ``bge``, ``all-minilm`` -> ``minilm``.
    When only a version number survives, the dropped words are kept as initials so the
    result still reads like the model: ``text-embedding-3-large`` -> ``te3-large``.

    :param model: Embedding model id, possibly ``provider/name:tag``. Blank is tolerated.
    :returns: A lowercase token safe to use inside an index name; ``"unknown"`` if nothing usable.
    """
    ident = (model or "").split("/")[-1].split(":")[0].lower()
    parts = [p for p in re.split(r"[^a-z0-9]+", ident) if p]
    if not parts:
        return "unknown"
    lead = 0
    while lead < len(parts) and parts[lead] in _GENERIC_MODEL_WORDS:
        lead += 1
    rest = parts[lead:]
    if not rest:
        return parts[0]
    if rest[0].isdigit():
        initials = "".join(p[0] for p in parts[:lead])
        return f"{initials}{rest[0]}" + (f"-{rest[1]}" if len(rest) > 1 else "")
    return rest[0]


def name_parts(config: dict, embedding_model: str) -> list[str]:
    """The pieces of a generated index name, in order, for display and for :func:`suggest_name`.

    :param config: A merged index config (see :func:`merged_config`).
    :param embedding_model: Embedding model id used for the build.
    :returns: Ordered name components, e.g. ``["nomic", "recursive", "320x40", "pc"]``.
    """
    chunking = config.get("chunking") or {}
    strategy = str(chunking.get("strategy", "recursive"))
    parts = [model_tag(embedding_model), strategy]

    if strategy == "semantic":
        parts.append(f"p{(chunking.get('semantic') or {}).get('percentile', 90)}")
    else:
        parts.append(f"{chunking.get('size', 0)}x{chunking.get('overlap', 0)}")

    # Flags, in a fixed order so the same parameters always produce the same name.
    if (chunking.get("parent_child") or {}).get("enabled"):
        parts.append("pc")
    if (chunking.get("contextual") or {}).get("enabled"):
        parts.append("ctx")
    if (config.get("graph") or {}).get("enabled"):
        parts.append("graph")
        if (config.get("graph") or {}).get("communities"):
            parts.append("comm")
    if (config.get("doc_metadata") or {}).get("enabled"):
        parts.append("meta")
    if not (config.get("dedupe") or {}).get("enabled"):
        parts.append("nodedupe")          # dedupe is on by default, so only its absence is news
    doc_ids = config.get("doc_ids")
    if doc_ids:
        # Only a non-empty subset gets a flag. An empty list is refused by `validate_index`, so
        # there is no buildable index for "0d" to name.
        parts.append(f"{len(doc_ids)}d")  # built from a subset of the corpus
    return parts


def suggest_name(config: dict, embedding_model: str, existing: Iterable[str] = ()) -> str:
    """Build a short, self-describing name from the parameters that shape an index.

    Shape is ``<model>-<strategy>-<size>x<overlap>[-flags][-N]``::

        nomic-recursive-320x40
        nomic-fixed-426x40-pc
        nomic-semantic-p90-ctx-graph
        nomic-recursive-320x40-2        # same parameters built again

    Flags: ``pc`` parent-child, ``ctx`` contextual, ``graph``/``comm`` knowledge graph and
    community reports, ``meta`` LLM document metadata, ``nodedupe`` near-duplicate removal
    turned off, ``<n>d`` built from a subset of ``n`` documents.

    HNSW ``M`` and ``ef_construction`` are deliberately excluded: they can be rebuilt on an
    existing index from stored vectors, so a name carrying them would silently go stale.

    Mirrored in ``frontend/src/pages/Indexes.tsx`` (``suggestName``) to preview the name in
    the Name field; keep the two in step.

    :param config: A merged index config (see :func:`merged_config`).
    :param embedding_model: Embedding model id used for the build.
    :param existing: Names already in use; a ``-2``, ``-3``, ... suffix disambiguates a repeat.
    :returns: A unique, non-empty index name.
    """
    base = "-".join(name_parts(config, embedding_model))
    taken = set(existing)
    if base not in taken:
        return base
    n = 2
    while f"{base}-{n}" in taken:
        n += 1
    return f"{base}-{n}"


async def build(store: Store, iid: str, models: dict):
    job = JOBS[iid] = {"stage": "starting", "done": 0, "total": 0, "error": None}
    t_start = time.time()
    try:
        idx = store.index(iid)
        cfg = idx["config"]
        emb_cfg, router_cfg = models["embedding"], models["router"]
        cache = LLMCache(store)
        # `doc_ids` absent or null means the whole corpus; a list means exactly those documents,
        # including when it is empty. A truthiness check here read `[]` as "no filter" and built an
        # index over everything from a selection that had asked for nothing.
        docs = store.documents(with_text=True)
        doc_ids = cfg.get("doc_ids")
        if doc_ids is not None:
            wanted = set(doc_ids)
            docs = [d for d in docs if d["id"] in wanted]
            if not docs:
                raise ValueError("The selected documents are no longer in the corpus, so there is "
                                 "nothing to index. Choose a different collection.")
        if not docs:
            raise ValueError("No documents to index. Add documents to the corpus first.")

        async def embed_docs(texts):
            return await embed(emb_cfg, texts, "doc")

        # 1. optional LLM document metadata (feeds self-query filtering)
        if cfg["doc_metadata"]["enabled"]:
            job.update(stage="extracting document metadata", done=0, total=len(docs))
            for d in docs:
                out = await chat_json(router_cfg, [{"role": "user", "content": prompts.DOC_METADATA.format(
                    title=d["title"], document=d["text"][:6000])}], task="doc_metadata", default={}, cache=cache) or {}
                if isinstance(out, dict):
                    d["metadata"] = {**{k: v for k, v in out.items() if v not in (None, "", [])}, **d["metadata"]}
                job["done"] += 1

        # 2. chunk
        job.update(stage="chunking", done=0, total=len(docs))
        rows, parents = [], []
        for d in docs:
            res = await chunk_text(d["text"], cfg["chunking"], embed_docs)
            base = len(parents)
            for p in res["parents"]:
                parents.append({"doc": d, **p})
            for c in res["chunks"]:
                meta = {**d["metadata"], "title": d["title"]}
                if c["section"]:
                    meta["section"] = c["section"]
                rows.append({"doc": d, "start": c["start"], "end": c["end"], "meta": meta, "context": "",
                             "parent": base + c["parent"] if c["parent"] is not None else None,
                             "text": d["text"][c["start"]:c["end"]]})
            job["done"] += 1

        removed = []
        if cfg["dedupe"]["enabled"]:
            job.update(stage="removing near-duplicates", done=0, total=1)
            rows, removed = await asyncio.to_thread(dedupe, rows, float(cfg["dedupe"]["threshold"]))

        # 3. contextual retrieval: situate every chunk in its document before indexing
        if cfg["chunking"]["contextual"]["enabled"]:
            job.update(stage="writing chunk contexts", done=0, total=len(rows))
            sem = asyncio.Semaphore(4)

            async def ctx(r):
                async with sem:
                    out = await chat(router_cfg, [{"role": "user", "content": prompts.CONTEXTUALIZE.format(
                        title=r["doc"]["title"], document=r["doc"]["text"][:8000], chunk=r["text"])}],
                        task="context", max_tokens=160, cache=cache)
                r["context"] = strip_think(out)[:600]
                job["done"] += 1
            await asyncio.gather(*[ctx(r) for r in rows])

        def indexed_text(r):
            head = r["meta"].get("section") or r["doc"]["title"]
            return "\n".join(x for x in (head, r["context"], r["text"]) if x)

        # 4. embed
        job.update(stage="embedding chunks", done=0, total=len(rows))
        vecs = []
        for i in range(0, len(rows), 64):
            vecs.append(await embed_docs([indexed_text(r) for r in rows[i:i + 64]]))
            job["done"] = min(len(rows), i + 64)
        matrix = np.vstack(vecs)

        store.x("DELETE FROM chunks WHERE index_id=?", (iid,))
        store.xmany("INSERT INTO chunks VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    [(iid, i, r["doc"]["id"], "chunk", r["parent"], r["start"], r["end"], r["text"], r["context"],
                      json.dumps(r["meta"]), matrix[i].tobytes()) for i, r in enumerate(rows)])
        store.xmany("INSERT INTO chunks VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    [(iid, i, p["doc"]["id"], "parent", None, p["start"], p["end"],
                      p["doc"]["text"][p["start"]:p["end"]], "", "{}", None) for i, p in enumerate(parents)])

        # 5. HNSW
        job.update(stage="building HNSW graph", done=0, total=1)
        path = DATA_DIR / f"{iid}.hnsw"
        path.unlink(missing_ok=True)
        t0 = time.time()
        await asyncio.to_thread(VectorIndex, matrix, cfg["hnsw"], path)
        hnsw_s = time.time() - t0

        # 6. knowledge graph
        gstats = {}
        if cfg["graph"]["enabled"]:
            job.update(stage="extracting knowledge graph", done=0, total=len(rows))
            g = await graphmod.extract([r["text"] for r in rows], router_cfg,
                                       progress=lambda d, t: job.update(done=d, total=t), cache=cache)
            g["communities"] = []
            if cfg["graph"].get("communities") and g["nodes"]:
                job.update(stage="summarising graph communities", done=0, total=1)
                g["communities"] = await graphmod.summarise_communities(
                    g, [r["text"] for r in rows], router_cfg, progress=lambda d, t: job.update(done=d, total=t), cache=cache)
                if g["communities"]:
                    store.put_blob(f"{iid}:communities", await embed(emb_cfg, [c["summary"] for c in g["communities"]], "doc"))
            if g["nodes"]:
                job.update(stage="embedding graph entities", done=0, total=1)
                node_vecs = await embed(emb_cfg, [n["label"] for n in g["nodes"].values()], "query")
                store.put_blob(f"{iid}:nodes", node_vecs)
            store.x("UPDATE indexes SET graph=? WHERE id=?", (json.dumps(g), iid))
            wide = [c for c, members in _entities_per_chunk(g).items() if len(members) > graphmod.CO_OCCURRENCE_LIMIT]
            gstats = {"entities": len(g["nodes"]), "relations": len(g["edges"]), "communities": len(g["communities"]),
                      # Passages too wide to treat co-occurrence as a relation. Recorded so a graph
                      # that is smaller than the corpus suggests can be explained rather than guessed at.
                      "graph_wide_chunks": len(wide), "graph_cooccurrence_limit": graphmod.CO_OCCURRENCE_LIMIT}

        lens = [len(r["text"]) // 4 for r in rows]
        stats = {"documents": len(docs), "chunks": len(rows), "parents": len(parents), "dim": int(matrix.shape[1]),
                 "embedding_model": emb_cfg.get("model"), "embedding_fingerprint": fingerprint(emb_cfg),
                 # Prefixes sit outside the fingerprint so that adding them here does not invalidate
                 # every index built before this field existed; they are compared separately.
                 "embedding_query_prefix": emb_cfg.get("query_prefix", ""),
                 "embedding_doc_prefix": emb_cfg.get("doc_prefix", ""),
                 "tokens_mean": int(np.mean(lens)), "tokens_p95": int(np.percentile(lens, 95)),
                 "tokens_min": int(min(lens)), "tokens_max": int(max(lens)),
                 "hnsw_build_s": round(hnsw_s, 3), "build_s": round(time.time() - t_start, 1),
                 "deduped": len(removed), "deduped_examples": removed[:200], "deduped_truncated": max(0, len(removed) - 200), "llm_cache_hits": cache.hits,
                 "llm_cache_misses": cache.misses, **gstats}
        store.x("UPDATE indexes SET status='ready', stats=? WHERE id=?", (json.dumps(stats), iid))
        _RUNTIME.pop(iid, None)
        job.update(stage="ready", done=1, total=1)
    except asyncio.CancelledError:
        job.update(stage="cancelled", error="Stopped before the index finished building.")
        mark_stopped(store, iid)
        raise
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        job.update(stage="failed", error=str(e))
        store.x("UPDATE indexes SET status='failed', stats=? WHERE id=?", (json.dumps({"error": str(e)[:500]}), iid))


class Runtime:
    """Everything the query pipeline needs, loaded once per index."""

    def __init__(self, store: Store, iid: str):
        idx = store.index(iid)
        if not idx or idx["status"] != "ready":
            raise ValueError("Index is not ready")
        self.id, self.name, self.config, self.stats = iid, idx["name"], idx["config"], idx["stats"]
        self.created = idx["created"]
        rows = store.q("SELECT * FROM chunks WHERE index_id=? AND kind='chunk' ORDER BY ord", (iid,))
        self.chunks = [{"i": r["ord"], "doc_id": r["doc_id"], "text": r["text"], "context": r["context"],
                        "parent": r["parent"], "start": r["s0"], "end": r["s1"],
                        "meta": json.loads(r["metadata"])} for r in rows]
        self.parents = {r["ord"]: r for r in store.q(
            "SELECT ord,doc_id,s0,s1,text FROM chunks WHERE index_id=? AND kind='parent'", (iid,))}
        dim = self.stats["dim"]
        self.matrix = np.vstack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows]).reshape(-1, dim)
        self.vectors = VectorIndex(self.matrix, self.config["hnsw"], DATA_DIR / f"{iid}.hnsw")
        self.bm25 = BM25(["\n".join(x for x in (c["meta"].get("section", ""), c["context"], c["text"]) if x)
                          for c in self.chunks])
        self.graph = graphmod.GraphIndex(idx["graph"], store.get_blob(f"{iid}:nodes", dim)) \
            if idx["graph"] and idx["graph"]["nodes"] else None
        self.communities = (idx["graph"] or {}).get("communities") or []
        self.community_vecs = store.get_blob(f"{iid}:communities", dim) if self.communities else None
        self.superseded = np.fromiter((bool(c["meta"].get("superseded_by")) for c in self.chunks), dtype=bool, count=len(self.chunks))
        # An index holds its own copy of every chunk, so deleting a document from the corpus left
        # its text sitting in each index that had been built from it — still retrieved, still handed
        # to the model, while the Corpus page said it was gone. Chunks cannot be removed from an
        # index in place (ords are positional: the matrix, the BM25 index and the HNSW labels are
        # all aligned to them), so the document is excluded at query time instead and the exclusion
        # is reported wherever the index is described.
        live = {r["id"] for r in store.q("SELECT id FROM documents")}
        self.removed = np.fromiter((c["doc_id"] not in live for c in self.chunks), dtype=bool, count=len(self.chunks))
        self.removed_docs = sorted({c["doc_id"] for c, gone in zip(self.chunks, self.removed) if gone})
        self.vocab = normalize.vocabulary([c["text"] for c in self.chunks])
        self.pca_mean, self.pca_comps, self.pca_var = pca2d(self.matrix)
        self._umap = None
        self._revision: dict | None = None
        self.titles = list(dict.fromkeys(c["meta"].get("title", "") for c in self.chunks))

    def live_mask(self) -> "np.ndarray | None":
        """Chunks whose source document still exists, or None when every one of them does.

        None is the common case and means "no filtering needed", which keeps the retrievers on
        their unmasked fast path.
        """
        return ~self.removed if self.removed.any() else None

    @property
    def revision(self) -> dict:
        """What this index is made of, as two hashes that move independently.

        ``vectors`` covers the embedded matrix: it changes only when the corpus, chunking or
        embedding model changes. ``hnsw`` covers the approximate-search parameters, which
        :func:`rebuild_hnsw` can change in place under the same index id. Keeping them apart says
        precisely what a saved run has drifted from — a rebuild leaves exact-mode recall, MRR and
        the ranking score untouched and only invalidates the approximation numbers.

        Computed once per loaded runtime and cached; hashing the matrix is linear in its size.
        """
        if self._revision is None:
            vectors = hashlib.sha1(np.ascontiguousarray(self.matrix, dtype=np.float32).tobytes()).hexdigest()[:16]
            hnsw = hashlib.sha1(json.dumps(self.config.get("hnsw", {}), sort_keys=True).encode()).hexdigest()[:12]
            self._revision = {"vectors": vectors, "hnsw": hnsw, "chunks": len(self.chunks), "dim": int(self.matrix.shape[1])}
        return self._revision

    def project(self, vecs: np.ndarray, method: str = "pca") -> list[list[float]]:
        if method == "umap":
            reducer, _ = self.umap()
            return np.round(reducer.transform(vecs), 4).tolist()
        return np.round((vecs - self.pca_mean) @ self.pca_comps.T, 4).tolist()

    def umap(self):
        """Optional non-linear map (pip install -e '.[umap]'). Fitted once per index, then cached."""
        if self._umap is None:
            try:
                import umap  # noqa: PLC0415
            except ImportError as e:
                raise ValueError("UMAP needs `pip install -e '.[umap]'`. PCA works without it.") from e
            n = len(self.chunks)
            reducer = umap.UMAP(n_components=2, metric="cosine", n_neighbors=max(2, min(15, n - 1)),
                                min_dist=0.15, random_state=7)
            self._umap = (reducer, np.round(reducer.fit_transform(self.matrix), 4).tolist())
        return self._umap

    def schema(self) -> dict:
        """Filterable fields with example values, shown to the self-query model."""
        values: dict[str, Counter] = {}
        for c in self.chunks:
            for k, v in c["meta"].items():
                if k == "section":
                    continue
                for item in (v if isinstance(v, list) else [v]):
                    if isinstance(item, (str, int, float, bool)):
                        values.setdefault(k, Counter())[item] += 1
        return {k: [v for v, _ in cnt.most_common(12)] for k, cnt in values.items()}

    def allowed(self, filters: list[dict]) -> np.ndarray | None:
        if not filters:
            return None
        mask = np.ones(len(self.chunks), dtype=bool)
        for f in filters:
            mask &= np.fromiter((_match(c["meta"].get(f.get("field")), f.get("op", "eq"), f.get("value"))
                                 for c in self.chunks), dtype=bool, count=len(self.chunks))
        return mask


def _match(have, op: str, want) -> bool:
    if have is None:
        return False
    items = have if isinstance(have, list) else [have]
    lo = lambda v: v.lower() if isinstance(v, str) else v  # noqa: E731
    try:
        if op == "eq":
            return any(lo(i) == lo(want) for i in items)
        if op == "ne":
            return all(lo(i) != lo(want) for i in items)
        if op == "in":
            return any(lo(i) in [lo(w) for w in (want if isinstance(want, list) else [want])] for i in items)
        if op == "contains":
            return any(str(lo(want)) in str(lo(i)) for i in items)
        a, b = float(items[0]), float(want)
        return {"gt": a > b, "gte": a >= b, "lt": a < b, "lte": a <= b}[op]
    except (TypeError, ValueError, KeyError):
        return False


#: How many loaded indexes to keep in memory. Each holds its embedding matrix, a BM25 index, a PCA
#: fit and possibly a graph, so the cost is proportional to corpus size and unbounded before this:
#: a sweep across ten indexes loaded and kept all ten.
RUNTIME_CACHE = int(os.environ.get("RAGLAB_RUNTIME_CACHE", "3"))


def compatibility(rt: "Runtime", models: dict) -> list[str]:
    """Why this index cannot be queried with these models, or [] when it can.

    Shared by the API and by :func:`raglab.pipeline.execute`, so a script or a new endpoint cannot
    reach the pipeline without it. A different embedding model of the same width produces neighbours
    that look plausible and mean nothing, which is the failure this exists to make impossible.
    """
    cfg = models.get("embedding") or {}
    problems = []
    if rt.stats.get("embedding_fingerprint") != fingerprint(cfg):
        problems.append(f"This index was embedded with '{rt.stats.get('embedding_model')}', but the embedding role "
                        f"now points at '{cfg.get('model')}'. Switch back or rebuild the index.")
    # Prefixes sit outside the fingerprint, so indexes built before they were recorded have none and
    # are left alone rather than being declared stale.
    for key, field in (("embedding_query_prefix", "query_prefix"), ("embedding_doc_prefix", "doc_prefix")):
        was = rt.stats.get(key)
        if was is not None and was != cfg.get(field, ""):
            problems.append(f"This index was embedded with {field}={was!r}, but the embedding role now uses "
                            f"{cfg.get(field, '')!r}. Queries would land in a different part of the space.")
    return problems


def runtime(store: Store, iid: str) -> Runtime:
    """Load an index, reusing it when it is already in memory.

    Least-recently-used beyond :data:`RUNTIME_CACHE`. Evicting only drops the in-memory form; the
    next use rebuilds it from SQLite, which costs a load rather than a rebuild.

    A running job holds its runtime by reference, so eviction cannot pull it out from under one.
    The cost is that returning to an evicted index while a job still holds it builds a second copy,
    briefly doubling that index's memory. One batch job runs at a time, so the ceiling is
    ``RUNTIME_CACHE + 1`` loaded indexes rather than the bound alone.
    """
    if iid in _RUNTIME:
        _RUNTIME.move_to_end(iid)
        return _RUNTIME[iid]
    rt = Runtime(store, iid)
    _RUNTIME[iid] = rt
    while len(_RUNTIME) > max(1, RUNTIME_CACHE):
        _RUNTIME.popitem(last=False)
    return rt


def rebuild_hnsw(store: Store, iid: str, params: dict) -> dict:
    """Re-build only the HNSW graph from stored embeddings (no re-embedding)."""
    rt = runtime(store, iid)
    path = DATA_DIR / f"{iid}.hnsw"
    path.unlink(missing_ok=True)
    t0 = time.time()
    VectorIndex(rt.matrix, params, path)
    cfg = {**rt.config, "hnsw": {**rt.config["hnsw"], **params}}
    stats = {**rt.stats, "hnsw_build_s": round(time.time() - t0, 3)}
    store.x("UPDATE indexes SET config=?, stats=? WHERE id=?", (json.dumps(cfg), json.dumps(stats), iid))
    _RUNTIME.pop(iid, None)
    return stats
