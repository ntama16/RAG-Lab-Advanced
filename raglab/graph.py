"""Graph index: LLM-extracted entities and relations, embedding-based entity linking at query time,
and personalised PageRank to score chunks by graph proximity to the query's entities."""
from __future__ import annotations
import asyncio, re
import numpy as np
from . import prompts
from .providers import chat_json, embed, Recorder


#: A chunk naming more entities than this contributes no co-occurrence edges. The pairs grow with
#: the square of the members, and a passage listing thirty things is not asserting 870 relations.
CO_OCCURRENCE_LIMIT = 24


def norm(name: str) -> str:
    return re.sub(r"\s+", " ", str(name).strip().lower())


async def extract(chunks: list[str], router_cfg: dict, progress=None, concurrency: int = 4, cache=None) -> dict:
    """Returns {"nodes": {name: {type, chunks:[i], label}}, "edges": [{s,t,r,chunk}]}"""
    nodes: dict[str, dict] = {}
    edges: list[dict] = []
    sem = asyncio.Semaphore(concurrency)
    done = 0

    async def one(i: int, text: str):
        nonlocal done
        async with sem:
            out = await chat_json(router_cfg, [{"role": "user", "content": prompts.GRAPH_EXTRACT.format(passage=text[:3000])}],
                                  task="graph_extract", default={}, cache=cache) or {}
        def touch(raw, typ=""):
            key = norm(raw)
            if not key or len(key) > 80:
                return None
            node = nodes.setdefault(key, {"label": str(raw).strip(), "type": typ, "chunks": []})
            if i not in node["chunks"]:
                node["chunks"].append(i)
            return key
        for e in out.get("entities", []) if isinstance(out, dict) else []:
            if isinstance(e, dict) and e.get("name"):
                touch(e["name"], str(e.get("type", "")))
        for r in out.get("relations", []) if isinstance(out, dict) else []:
            if isinstance(r, dict) and r.get("source") and r.get("target"):
                s, t = touch(r["source"]), touch(r["target"])
                if s and t and s != t:
                    edges.append({"s": s, "t": t, "r": str(r.get("relation", "related_to"))[:60], "chunk": i})
        done += 1
        if progress:
            progress(done, len(chunks))

    await asyncio.gather(*[one(i, t) for i, t in enumerate(chunks)])
    return {"nodes": nodes, "edges": edges}


class GraphIndex:
    def __init__(self, data: dict, node_vecs: np.ndarray | None):
        self.names = list(data["nodes"].keys())
        self.nodes = data["nodes"]
        self.edges = data["edges"]
        self.pos = {n: i for i, n in enumerate(self.names)}
        self.vecs = node_vecs
        n = len(self.names)
        # The graph is sparse: entities connect to a handful of others, not to all of them. Held as
        # two dense n-by-n float32 matrices this cost 8 bytes per entity pair — about 1.6 GB at ten
        # thousand entities, allocated on every index load, for a matrix that is nearly all zeros.
        # Co-occurrence edges were also built by a Python double loop over each chunk's members.
        weights: dict[tuple[int, int], float] = {}
        for e in self.edges:
            a, b = self.pos[e["s"]], self.pos[e["t"]]
            weights[(a, b)] = weights.get((a, b), 0.0) + 1.0
            weights[(b, a)] = weights.get((b, a), 0.0) + 1.0
        # Entities that co-occur in a chunk are weakly connected even without an explicit relation.
        # A chunk naming many entities would contribute members^2 pairs, so the widest are skipped:
        # they say "these appeared in the same passage", which is not evidence of a relation.
        by_chunk: dict[int, list[int]] = {}
        for name, node in self.nodes.items():
            for c in node["chunks"]:
                by_chunk.setdefault(c, []).append(self.pos[name])
        self.wide_chunks = [c for c, members in by_chunk.items() if len(members) > CO_OCCURRENCE_LIMIT]
        for c, members in by_chunk.items():
            if len(members) > CO_OCCURRENCE_LIMIT:
                continue
            for a in members:
                for b in members:
                    if a != b:
                        weights.setdefault((a, b), 0.25)
        self.edge_count = len(weights)
        # Column-stochastic transition matrix in CSR form, so a walk step is one sparse matvec.
        rows = np.fromiter((a for a, _ in weights), dtype=np.int32, count=len(weights))
        cols = np.fromiter((b for _, b in weights), dtype=np.int32, count=len(weights))
        vals = np.fromiter(weights.values(), dtype=np.float32, count=len(weights))
        out_degree = np.zeros(n, dtype=np.float32)
        np.add.at(out_degree, cols, vals)  # column sums: mass leaving each node
        np.divide(vals, np.where(out_degree[cols] == 0, 1.0, out_degree[cols]), out=vals)
        self._rows, self._cols, self._vals, self._n = rows, cols, vals, n

    def _step(self, rank: np.ndarray) -> np.ndarray:
        """One walk step: the same product ``trans @ rank`` gave, without materialising ``trans``."""
        out = np.zeros(self._n, dtype=np.float32)
        if self._vals.size:
            np.add.at(out, self._rows, self._vals * rank[self._cols])
        return out

    async def link(self, query: str, router_cfg: dict, emb_cfg: dict, rec: Recorder, threshold: float = 0.6):
        out = await chat_json(router_cfg, [{"role": "user", "content": prompts.QUERY_ENTITIES.format(query=query)}],
                              task="query_entities", rec=rec, default={}) or {}
        mentions = [str(m) for m in (out.get("entities", []) if isinstance(out, dict) else []) if m][:8]
        links = []
        if not mentions or not self.names:
            return mentions, links
        vecs = await embed(emb_cfg, mentions, "query", rec) if self.vecs is not None else None
        for mi, m in enumerate(mentions):
            key = norm(m)
            if key in self.pos:
                links.append({"mention": m, "node": key, "score": 1.0, "how": "exact"})
                continue
            sub = [n for n in self.names if key in n or n in key]
            if sub:
                links.append({"mention": m, "node": min(sub, key=len), "score": 0.9, "how": "substring"})
                continue
            if vecs is not None:
                sims = self.vecs @ vecs[mi]
                j = int(np.argmax(sims))
                if sims[j] >= threshold:
                    links.append({"mention": m, "node": self.names[j], "score": float(sims[j]), "how": "embedding"})
        return mentions, links

    def search(self, links: list[dict], k: int, alpha: float = 0.25, allowed: np.ndarray | None = None):
        """Personalised PageRank seeded on linked entities; chunk score = sum of PPR mass of its entities."""
        if not links:
            return [], {"nodes": [], "edges": []}
        n = len(self.names)
        seed = np.zeros(n, dtype=np.float32)
        for l in links:
            seed[self.pos[l["node"]]] += l["score"]
        seed /= seed.sum()
        rank = seed.copy()
        for _ in range(30):
            rank = (1 - alpha) * self._step(rank) + alpha * seed
        scores: dict[int, float] = {}
        for name, node in self.nodes.items():
            for c in node["chunks"]:
                scores[c] = scores.get(c, 0.0) + float(rank[self.pos[name]]) / len(node["chunks"]) ** 0.5
        hits = sorted(scores.items(), key=lambda x: -x[1])
        if allowed is not None:
            hits = [(c, s) for c, s in hits if allowed[c]]
        top_nodes = [self.names[i] for i in np.argsort(-rank)[:14] if rank[i] > 0]
        # A silently smaller graph is the kind of thing that makes a result impossible to explain.
        skipped = ({"chunks": len(self.wide_chunks), "limit": CO_OCCURRENCE_LIMIT,
                    "note": f"{len(self.wide_chunks)} passage(s) named more than {CO_OCCURRENCE_LIMIT} entities, so "
                            f"their co-occurrence edges were left out: at that width 'appeared together' stops being "
                            f"evidence of a relation. Explicit relations from those passages are still in the graph."}
                   if self.wide_chunks else None)
        keep = set(top_nodes)
        seeds = {l["node"] for l in links}
        sub = {"nodes": [{"id": x, "label": self.nodes[x]["label"], "rank": round(float(rank[self.pos[x]]), 4),
                          "seed": x in seeds} for x in top_nodes],
               "edges": [{"s": a, "r": r, "t": b} for a, r, b in dict.fromkeys(
                   (e["s"], e["r"], e["t"]) for e in self.edges if e["s"] in keep and e["t"] in keep)][:30],
               "entities": len(self.names), "graph_edges": self.edge_count, "skipped": skipped}
        return hits[:k], sub


def communities(data: dict, min_size: int = 3, rounds: int = 12) -> list[list[str]]:
    """Weighted label propagation over the entity graph (relations count 1, chunk co-occurrence 0.25)."""
    names = sorted(data["nodes"])
    w: dict[str, dict[str, float]] = {n: {} for n in names}
    for e in data["edges"]:
        w[e["s"]][e["t"]] = w[e["s"]].get(e["t"], 0) + 1.0
        w[e["t"]][e["s"]] = w[e["t"]].get(e["s"], 0) + 1.0
    by_chunk: dict[int, list[str]] = {}
    for n, node in data["nodes"].items():
        for c in node["chunks"]:
            by_chunk.setdefault(c, []).append(n)
    for members in by_chunk.values():
        for a in members:
            for b in members:
                if a != b:
                    w[a].setdefault(b, 0.25)
    label = {n: n for n in names}
    for _ in range(rounds):
        changed = False
        for n in names:  # deterministic order
            tally: dict[str, float] = {}
            for m, wt in w[n].items():
                tally[label[m]] = tally.get(label[m], 0) + wt
            if tally:
                best = max(sorted(tally), key=lambda k: tally[k])
                if best != label[n]:
                    label[n], changed = best, True
        if not changed:
            break
    groups: dict[str, list[str]] = {}
    for n, l in label.items():
        groups.setdefault(l, []).append(n)
    return sorted((g for g in groups.values() if len(g) >= min_size), key=len, reverse=True)[:24]


async def summarise_communities(data: dict, chunks: list[str], router_cfg: dict, progress=None, cache=None) -> list[dict]:
    from .providers import chat, strip_think  # noqa: PLC0415
    out, groups = [], communities(data)
    for k, members in enumerate(groups):
        ms = set(members)
        rels = [f"- {e['s']} {e['r']} {e['t']}" for e in data["edges"] if e["s"] in ms and e["t"] in ms][:25]
        cids = sorted({c for m in members for c in data["nodes"][m]["chunks"]})
        counts = sorted(cids, key=lambda c: -sum(c in data["nodes"][m]["chunks"] for m in members))[:4]
        text = await chat(router_cfg, [{"role": "user", "content": prompts.COMMUNITY.format(
            entities=", ".join(data["nodes"][m]["label"] for m in members[:20]), relations="\n".join(rels) or "- (co-occurrence only)",
            excerpts="\n---\n".join(chunks[c][:500] for c in counts))}], task="community", max_tokens=350, cache=cache)
        text = strip_think(text)
        out.append({"id": k, "title": text.splitlines()[0].strip("# *")[:90] if text else f"Community {k}", "summary": text,
                    "entities": [data["nodes"][m]["label"] for m in members], "chunks": cids})
        if progress:
            progress(k + 1, len(groups))
    return out
