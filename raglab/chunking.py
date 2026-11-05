"""Chunking. Every strategy works on character spans so chunks always map back to the source text.

Strategies: fixed | sentence | recursive | markdown | semantic
Sizes are given in approximate tokens (1 token ~ 4 chars).
"""
from __future__ import annotations
import re
from typing import Awaitable, Callable
import numpy as np

Span = tuple[int, int]
CPT = 4  # chars per token, approximation used for every size knob

DEFAULT_CHUNKING = {
    "strategy": "recursive", "size": 320, "overlap": 40,
    "semantic": {"buffer": 1, "percentile": 90, "min_tokens": 60, "max_tokens": 600},
    "parent_child": {"enabled": False, "parent_size": 1200},
    "contextual": {"enabled": False},
}

_SENT_END = re.compile(r"(?<=[.!?])[\"')\]]*\s+(?=[A-Z0-9\"'(\[])|\n{2,}")
_HEADING = re.compile(r"^(#{1,6})\s+(.+)$", re.M)


def _trim(text: str, a: int, b: int) -> Span | None:
    while a < b and text[a].isspace():
        a += 1
    while b > a and text[b - 1].isspace():
        b -= 1
    return (a, b) if b > a else None


def _split(text: str, a: int, b: int, pattern: re.Pattern) -> list[Span]:
    out, cur = [], a
    for m in pattern.finditer(text, a, b):
        s = _trim(text, cur, m.start())
        if s:
            out.append(s)
        cur = m.end()
    s = _trim(text, cur, b)
    if s:
        out.append(s)
    return out


def sentences(text: str, a: int = 0, b: int | None = None) -> list[Span]:
    return _split(text, a, len(text) if b is None else b, _SENT_END)


def _words(text: str, a: int, b: int) -> list[Span]:
    return [(m.start(), m.end()) for m in re.compile(r"\S+").finditer(text, a, b)]


def _units(text: str, a: int, b: int, max_chars: int) -> list[Span]:
    """Paragraphs; oversized paragraphs fall back to sentences, oversized sentences to words."""
    out: list[Span] = []
    for p in _split(text, a, b, re.compile(r"\n\s*\n")):
        if p[1] - p[0] <= max_chars:
            out.append(p)
            continue
        for s in sentences(text, *p):
            out.extend([s] if s[1] - s[0] <= max_chars else _words(text, *s))
    return out


def _pack(units: list[Span], max_chars: int, overlap_chars: int) -> list[Span]:
    """Greedily pack consecutive units into windows <= max_chars, carrying trailing units as overlap."""
    chunks: list[Span] = []
    i = 0
    while i < len(units):
        j = i
        while j + 1 < len(units) and units[j + 1][1] - units[i][0] <= max_chars:
            j += 1
        chunks.append((units[i][0], units[j][1]))
        if j + 1 >= len(units):
            break
        nxt = j + 1
        while nxt - 1 > i and units[j][1] - units[nxt - 1][0] <= overlap_chars:
            nxt -= 1
        # carried-over units must leave room for at least one new unit, or the next chunk is pure repetition
        while nxt <= j and units[j + 1][1] - units[nxt][0] > max_chars:
            nxt += 1
        i = nxt
    return chunks


def _sections(text: str) -> list[tuple[Span, str]]:
    """Markdown sections with their heading path."""
    heads = list(_HEADING.finditer(text))
    if not heads:
        return [((0, len(text)), "")]
    out, path = [], []
    if heads[0].start() > 0 and text[:heads[0].start()].strip():
        out.append(((0, heads[0].start()), ""))
    for k, h in enumerate(heads):
        level = len(h.group(1))
        path = path[:level - 1] + [h.group(2).strip()]
        end = heads[k + 1].start() if k + 1 < len(heads) else len(text)
        out.append(((h.start(), end), " > ".join(path)))
    return out


EmbedFn = Callable[[list[str]], Awaitable[np.ndarray]]


async def semantic_breakpoints(text: str, a: int, b: int, cfg: dict, embed: EmbedFn):
    """Kamradt-style semantic chunking: embed a sliding window around each sentence, cut where the
    cosine distance between neighbours exceeds the chosen percentile."""
    sents = sentences(text, a, b)
    if len(sents) < 3:
        return sents, [], 0.0, [(a, b)] if _trim(text, a, b) else []
    buf = int(cfg.get("buffer", 1))
    windows = [text[sents[max(0, i - buf)][0]:sents[min(len(sents) - 1, i + buf)][1]] for i in range(len(sents))]
    vecs = await embed(windows)
    dist = (1.0 - np.sum(vecs[:-1] * vecs[1:], axis=1)).tolist()
    threshold = float(np.percentile(dist, float(cfg.get("percentile", 90))))
    groups, start = [], 0
    for i, d in enumerate(dist):
        if d > threshold:
            groups.append((start, i))
            start = i + 1
    groups.append((start, len(sents) - 1))
    min_c, max_c = int(cfg.get("min_tokens", 60)) * CPT, int(cfg.get("max_tokens", 600)) * CPT
    merged: list[Span] = []
    for g0, g1 in groups:
        span = (sents[g0][0], sents[g1][1])
        if merged and (merged[-1][1] - merged[-1][0] < min_c) and (span[1] - merged[-1][0] <= max_c):
            merged[-1] = (merged[-1][0], span[1])
        else:
            merged.append(span)
    final: list[Span] = []
    for s in merged:
        if s[1] - s[0] > max_c:
            final.extend(_pack(_units(text, s[0], s[1], max_c), max_c, 0))
        else:
            final.append(s)
    return sents, dist, threshold, final


async def chunk_text(text: str, cfg: dict, embed: EmbedFn | None = None) -> dict:
    """Returns {"chunks": [{start,end,section,parent}], "parents": [{start,end}], "debug": {...}}"""
    cfg = {**DEFAULT_CHUNKING, **(cfg or {})}
    strategy = cfg["strategy"]
    max_c, ov_c = int(cfg["size"]) * CPT, int(cfg["overlap"]) * CPT
    debug: dict = {}
    secs = _sections(text) if strategy == "markdown" else []

    async def chunk_region(a: int, b: int) -> list[tuple[Span, str]]:
        if strategy == "fixed":
            return [(s, "") for s in _pack(_words(text, a, b), max_c, ov_c)]
        if strategy == "sentence":
            units = []
            for s in sentences(text, a, b):
                units.extend([s] if s[1] - s[0] <= max_c else _words(text, *s))
            return [(s, "") for s in _pack(units, max_c, ov_c)]
        if strategy == "markdown":
            out = []
            for (sa, sb), path in secs:
                sa, sb = max(sa, a), min(sb, b)
                if sa < sb and _HEADING.sub("", text[sa:sb]).strip():  # skip heading-only sections
                    out.extend((s, path) for s in _pack(_units(text, sa, sb, max_c), max_c, ov_c))
            return out
        if strategy == "semantic":
            if embed is None:
                raise ValueError("semantic chunking needs an embedding model")
            sents, dist, thr, spans = await semantic_breakpoints(text, a, b, cfg["semantic"], embed)
            debug.setdefault("semantic", []).append(
                {"sentence_ends": [s[1] for s in sents], "distances": [round(d, 4) for d in dist],
                 "threshold": round(thr, 4)})
            return [(s, "") for s in spans]
        return [(s, "") for s in _pack(_units(text, a, b, max_c), max_c, ov_c)]  # recursive

    parents: list[Span] = []
    chunks: list[dict] = []
    pc = cfg.get("parent_child") or {}
    if pc.get("enabled"):
        p_max = int(pc.get("parent_size", 1200)) * CPT
        parents = _pack(_units(text, 0, len(text), p_max), p_max, 0)
        for pi, (pa, pb) in enumerate(parents):
            for (s, path) in await chunk_region(pa, pb):
                chunks.append({"start": s[0], "end": s[1], "section": path, "parent": pi})
    else:
        for (s, path) in await chunk_region(0, len(text)):
            chunks.append({"start": s[0], "end": s[1], "section": path, "parent": None})
    return {"chunks": chunks, "parents": [{"start": a, "end": b} for a, b in parents], "debug": debug}
