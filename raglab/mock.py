"""Deterministic stand-ins so the test-suite runs without any model server.

NOT a retrieval technique. Select provider "mock" (or RAGLAB_MOCK=1) only for tests and UI development.
"""
from __future__ import annotations
import hashlib, json, re
import numpy as np

DIM = 256
_WORD = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return [w for w in _WORD.findall(text.lower()) if len(w) > 2]


def embed(texts: list[str]) -> list[list[float]]:
    out = []
    for t in texts:
        v = np.zeros(DIM, dtype=np.float32)
        for w in _tokens(t):
            h = int(hashlib.md5(w.encode()).hexdigest(), 16)
            v[h % DIM] += 1.0 if (h >> 9) & 1 else -1.0
        out.append(v.tolist())
    return out


def rerank(query: str, docs: list[str]) -> list[float]:
    q = set(_tokens(query))
    return [len(q & set(_tokens(d))) / max(len(q), 1) for d in docs]


def _field(text: str, name: str) -> str:
    m = re.search(rf"^{name}:\s*(.+?)(?:\n\n|\Z)", text, re.S | re.M)
    return m.group(1).strip() if m else ""


def chat(task: str, messages: list[dict]) -> str:
    user = messages[-1]["content"]
    q = _field(user, "Query") or _field(user, "Question")
    if task == "route":
        multi = any(w in q.lower() for w in ("relationship", "connect", "between", "both"))
        chit = q.lower().strip(" !?.") in ("hi", "hello", "thanks", "thank you")
        return json.dumps({"intent": "chitchat" if chit else "multi_hop" if multi else "conceptual",
                           "needs_retrieval": not chit, "in_scope": True, "reason": "mock router"})
    if task == "self_query":
        return json.dumps({"query": q, "filters": []})
    if task == "multi_query":
        words = _tokens(q)
        return json.dumps({"queries": [" ".join(words), " ".join(reversed(words))]})
    if task == "hyde":
        return f"{q} This passage explains {q.lower()} in detail."
    if task == "grade":
        passage = _field(user, "Passage")
        overlap = len(set(_tokens(q)) & set(_tokens(passage)))
        grade = "relevant" if overlap >= 2 else "partial" if overlap == 1 else "irrelevant"
        return json.dumps({"grade": grade, "reason": f"{overlap} shared terms"})
    if task == "rewrite":
        return json.dumps({"query": " ".join(_tokens(q))})
    if task == "context":
        return "Context: part of the source document."
    if task in ("graph_extract", "query_entities"):
        text = _field(user, "Passage") or q
        ents = list(dict.fromkeys(re.findall(r"\b[A-Z][a-zA-Z]{2,}(?:\s[A-Z][a-zA-Z]+)*", text)))[:8]
        if task == "query_entities":
            return json.dumps({"entities": ents or _tokens(q)[:3]})
        rels = [{"source": a, "target": b, "relation": "mentioned_with"} for a, b in zip(ents, ents[1:])]
        return json.dumps({"entities": [{"name": e, "type": "concept"} for e in ents], "relations": rels})
    if task == "doc_metadata":
        return json.dumps({"topics": _tokens(user)[:3]})
    if task in ("evalgen", "evalgen_paraphrase"):
        passage = _field(user, "Passage")
        sent = re.split(r"(?<=[.!?])\s", passage)[0]
        return json.dumps({"question": f"What does the corpus say about {' '.join(_tokens(sent)[:5])}?",
                           "answer": sent, "evidence": sent})
    if task == "condense":
        hist = _field(user, "Conversation")
        subj = re.findall(r"\b[A-Z][a-zA-Z]{3,}", hist)
        return json.dumps({"query": q if not re.search(r"\b(it|that|they|this)\b", q.lower()) else f"{q} ({subj[-1] if subj else ''})"})
    if task == "decompose":
        parts = [p.strip() for p in re.split(r"\s+and\s+(?=(?:how|what|when|why|is|does|do|can)\b)", q.rstrip("?"), flags=re.I) if len(p.split()) >= 3]
        return json.dumps({"sub_questions": parts if len(parts) > 1 else [q]})
    if task == "step_back":
        return json.dumps({"query": "What is the general policy regarding " + " ".join(_tokens(q)[:4]) + "?"})
    if task == "refine":
        strips = re.findall(r"\[(\d+)\] ([^\[]+)", _field(user, "Passage"))
        keep = [int(n) for n, t in strips if set(_tokens(q)) & set(_tokens(t))]
        return json.dumps({"keep": keep})
    if task == "verify":
        return json.dumps({"claims": [{"claim": _field(user, "Answer")[:80], "verdict": "supported", "passage": 1}]})
    if task == "community":
        return "Cluster report\nThese entities appear together across the corpus."
    if task == "suggest_rules":
        return json.dumps({"abbreviations": {"rma": "return merchandise authorization RMA"}, "aliases": {}, "terminology": {}})
    if task == "evalgen_multi":
        a, b = _field(user, "Passage"), user.split("B: ")[-1]
        sa, sb = re.split(r"(?<=[.!?])\s", a.replace("A: ", ""))[0], re.split(r"(?<=[.!?])\s", b)[0]
        return json.dumps({"question": f"How do {' '.join(_tokens(sa)[:3])} relate to {' '.join(_tokens(sb)[:3])}?", "answer": sa + " " + sb, "evidence": [sa, sb]})
    if task == "evalgen_unanswerable":
        return json.dumps({"question": "What is the policy on interplanetary drone deliveries?"})
    if task == "judge":
        return json.dumps({"faithfulness": 1.0, "correctness": 1.0, "reason": "mock judge"})
    if task == "generate":
        return "According to the retrieved passages, the corpus addresses this directly [1]."
    return "ready"
