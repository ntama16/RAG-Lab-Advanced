"""Dense index: HNSW (hnswlib) for approximate search, plus exact search over the same matrix so the
lab can measure what the approximation costs."""
from __future__ import annotations
import time
from pathlib import Path
import hnswlib
import numpy as np

DEFAULT_HNSW = {"M": 16, "ef_construction": 200}


class VectorIndex:
    def __init__(self, matrix: np.ndarray, params: dict | None = None, path: Path | None = None):
        self.matrix = np.ascontiguousarray(matrix, dtype=np.float32)
        self.params = {**DEFAULT_HNSW, **(params or {})}
        n, dim = self.matrix.shape
        self.hnsw = hnswlib.Index(space="cosine", dim=dim)
        if path and path.exists():
            self.hnsw.load_index(str(path), max_elements=n)
        else:
            self.hnsw.init_index(max_elements=max(n, 1), M=int(self.params["M"]),
                                 ef_construction=int(self.params["ef_construction"]), random_seed=7)
            if n:
                self.hnsw.add_items(self.matrix, np.arange(n))
            if path:
                self.hnsw.save_index(str(path))

    def exact(self, q: np.ndarray, k: int, allowed: np.ndarray | None = None):
        sims = self.matrix @ q
        if allowed is not None:
            sims = np.where(allowed, sims, -2.0)
        top = np.argsort(-sims)[:k]
        return [(int(i), float(sims[i])) for i in top if sims[i] > -2.0]

    def approx(self, q: np.ndarray, k: int, ef_search: int = 64, allowed: np.ndarray | None = None):
        """Approximate search. Returns ``(hits, info)``; ``info`` reports what actually ran, since
        ``ef`` is raised to ``k`` when asked for less (HNSW cannot return k neighbours from a
        narrower search) and a graph that cannot reach k filtered neighbours falls back to exact."""
        n = self.matrix.shape[0]
        if n == 0:
            return [], {"ef_requested": int(ef_search), "ef_effective": int(ef_search), "exact_fallback": False}
        limit = int(allowed.sum()) if allowed is not None else n
        k = min(k, limit)
        info = {"ef_requested": int(ef_search), "ef_effective": max(int(ef_search), k), "exact_fallback": False}
        if k == 0:
            return [], info
        self.hnsw.set_ef(info["ef_effective"])
        flt = (lambda i: bool(allowed[i])) if allowed is not None else None
        try:
            labels, dists = self.hnsw.knn_query(q, k=k, filter=flt)
        except RuntimeError:  # graph could not reach k filtered neighbours — fall back to what exact gives
            info["exact_fallback"] = True
            return self.exact(q, k, allowed), info
        return [(int(i), float(1.0 - d)) for i, d in zip(labels[0], dists[0])], info

    def search(self, q: np.ndarray, k: int, mode: str = "hnsw", ef_search: int = 64,
               allowed: np.ndarray | None = None, compare: bool = False):
        """Returns ``(hits, recall, info)``. ``recall`` is None unless ``compare`` ran an exact
        search alongside; ``info["verify_ms"]`` is that comparison's cost, which is measurement
        overhead and not part of what the approximate search would take on its own."""
        if mode == "exact":
            t0 = time.perf_counter()
            hits = self.exact(q, k, allowed)
            return hits, None, {"ef_requested": None, "ef_effective": None, "exact_fallback": False,
                                "search_ms": round((time.perf_counter() - t0) * 1000, 3), "verify_ms": 0.0}
        t0 = time.perf_counter()
        hits, info = self.approx(q, k, ef_search, allowed)
        info["search_ms"] = round((time.perf_counter() - t0) * 1000, 3)
        recall, info["verify_ms"] = None, 0.0
        if compare:
            t1 = time.perf_counter()
            truth = {i for i, _ in self.exact(q, k, allowed)}
            recall = len(truth & {i for i, _ in hits}) / max(len(truth), 1)
            info["verify_ms"] = round((time.perf_counter() - t1) * 1000, 3)
        return hits, recall, info


def pca2d(matrix: np.ndarray):
    """Returns (mean, components[2,dim], explained variance share) for a stable 2-D projection."""
    mean = matrix.mean(axis=0)
    centered = matrix - mean
    if centered.shape[0] < 3:
        return mean, np.eye(2, matrix.shape[1], dtype=np.float32), 0.0
    _, sv, vt = np.linalg.svd(centered[:3000], full_matrices=False)
    var = sv ** 2
    return mean, vt[:2], float(var[:2].sum() / max(var.sum(), 1e-12))
