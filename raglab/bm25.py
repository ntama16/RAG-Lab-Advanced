"""Okapi BM25 over an inverted index. k1 and b are query-time knobs, so they can be tuned without re-indexing."""
from __future__ import annotations
import math, re
from collections import Counter, defaultdict
import numpy as np

_TOKEN = re.compile(r"[a-z0-9][a-z0-9_\-']*")
STOP = frozenset("""a an and are as at be but by for from has have if in into is it its of on or s so such t that the
their then there these they this to was were what when where which who will with would""".split())


def tokenize(text: str) -> list[str]:
    return [w.strip("-'") for w in _TOKEN.findall(text.lower()) if w not in STOP and len(w) > 1]


class BM25:
    def __init__(self, texts: list[str]):
        self.n = len(texts)
        self.doc_len = np.zeros(self.n, dtype=np.float32)
        postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for i, t in enumerate(texts):
            toks = tokenize(t)
            self.doc_len[i] = len(toks)
            for term, tf in Counter(toks).items():
                postings[term].append((i, tf))
        self.avgdl = float(self.doc_len.mean()) if self.n else 0.0
        self.postings = {t: (np.fromiter((d for d, _ in p), dtype=np.int32, count=len(p)),
                             np.fromiter((f for _, f in p), dtype=np.float32, count=len(p)))
                         for t, p in postings.items()}

    def search(self, query: str, k: int, k1: float = 1.5, b: float = 0.75, allowed: np.ndarray | None = None):
        """Returns ([(idx, score)], per-term idf) — idf is exposed so the UI can explain the ranking."""
        scores = np.zeros(self.n, dtype=np.float32)
        explain = {}
        for term in dict.fromkeys(tokenize(query)):
            if term not in self.postings:
                explain[term] = 0.0
                continue
            docs, tf = self.postings[term]
            idf = math.log(1 + (self.n - len(docs) + 0.5) / (len(docs) + 0.5))
            explain[term] = round(idf, 3)
            norm = k1 * (1 - b + b * self.doc_len[docs] / max(self.avgdl, 1e-9))
            scores[docs] += idf * tf * (k1 + 1) / (tf + norm)
        if allowed is not None:
            scores = np.where(allowed, scores, 0.0)
        top = np.argsort(-scores)[:k]
        return [(int(i), float(scores[i])) for i in top if scores[i] > 0], explain
