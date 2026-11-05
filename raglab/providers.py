"""Model access. Every call goes to a real model endpoint; `mock` exists only for tests/CI.

Chat/embedding providers
  ollama     native Ollama API (/api/embed, /api/chat)
  openai     any OpenAI-compatible server: OpenAI, OpenRouter, Groq, LM Studio, vLLM, llama.cpp, Together...
  anthropic  Anthropic Messages API (chat only)
Reranker providers
  cohere_compat  POST {base_url}/rerank  (Cohere, Jina, Voyage, vLLM, llama.cpp --reranking, infinity)
  tei            HuggingFace text-embeddings-inference /rerank
  cross_encoder  in-process sentence-transformers CrossEncoder (pip install -e '.[crossencoder]')
  llm            pointwise LLM relevance scoring through a chat endpoint (`via`: ollama|openai|anthropic)
"""
from __future__ import annotations
import asyncio, hashlib, json, re, time
from typing import AsyncIterator
import httpx
import numpy as np
from .config import api_key
from . import mock

_client: httpx.AsyncClient | None = None


def client() -> httpx.AsyncClient:
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(timeout=httpx.Timeout(180.0, connect=10.0))
    return _client


class ProviderError(RuntimeError):
    pass


class Recorder:
    """Collects every model call made during a pipeline stage so the UI can show prompts and outputs."""

    def __init__(self):
        self.calls: list[dict] = []

    def add(self, **kw):
        self.calls.append(kw)

    def drain(self) -> list[dict]:
        calls, self.calls = self.calls, []
        return calls


RETRY_STATUS = {408, 429, 500, 502, 503, 504}


async def _post(url: str, payload: dict, headers: dict | None = None, cfg: dict | None = None) -> dict:
    """POST with a per-role timeout and bounded retries on transport errors and retryable statuses."""
    cfg = cfg or {}
    retries, timeout = int(cfg.get("retries", 2)), float(cfg.get("timeout", 180))
    last = ""
    for attempt in range(retries + 1):
        try:
            r = await client().post(url, json=payload, headers=headers or {}, timeout=httpx.Timeout(timeout, connect=10.0))
            if r.status_code < 400:
                return r.json()
            last = f"{url} returned {r.status_code}: {r.text[:400]}"
            if r.status_code not in RETRY_STATUS:
                break
        except httpx.HTTPError as e:
            last = f"Cannot reach {url}: {e.__class__.__name__} {e}"
        if attempt < retries:
            await asyncio.sleep(0.8 * 2 ** attempt)
    raise ProviderError(last)


def _cost(cfg: dict, tin: int, tout: int) -> float:
    """price_in / price_out are USD per million tokens, set per role. Local models default to 0."""
    return round((tin * float(cfg.get("price_in", 0)) + tout * float(cfg.get("price_out", 0))) / 1e6, 6)


def _est(text) -> int:
    return max(1, len(text if isinstance(text, str) else "".join(m["content"] for m in text)) // 4)


class LLMCache:
    """Persistent cache for deterministic index-time calls (chunk contexts, graph extraction, doc metadata),
    keyed by task + model + prompt, so rebuilding an index does not pay for the same calls twice."""

    def __init__(self, store):
        self.store, self.hits, self.misses = store, 0, 0

    def key(self, cfg: dict, task: str, messages: list[dict]) -> str:
        return hashlib.sha1(f"{task}|{cfg.get('provider')}|{cfg.get('model')}|{json.dumps(messages)}".encode()).hexdigest()

    def get(self, key: str):
        rows = self.store.q("SELECT value FROM llmcache WHERE key=?", (key,))
        if rows:
            self.hits += 1
            return rows[0]["value"]
        self.misses += 1
        return None

    def put(self, key: str, value: str):
        self.store.x("INSERT OR REPLACE INTO llmcache VALUES (?,?)", (key, value))


def _auth(cfg: dict) -> dict:
    key = api_key(cfg)
    return {"Authorization": f"Bearer {key}"} if key else {}


def _base(cfg: dict) -> str:
    return (cfg.get("base_url") or "").rstrip("/")


# ---------------------------------------------------------------- embeddings
async def embed(cfg: dict, texts: list[str], kind: str = "doc", rec: Recorder | None = None) -> np.ndarray:
    """Returns L2-normalised float32 vectors, shape (len(texts), dim)."""
    if not texts:
        return np.zeros((0, 1), dtype=np.float32)
    prefix = cfg.get("query_prefix", "") if kind == "query" else cfg.get("doc_prefix", "")
    inputs = [prefix + (t if t.strip() else " ") for t in texts]
    t0 = time.perf_counter()
    provider = cfg.get("provider")
    bs = int(cfg.get("batch_size", 32))
    out: list[list[float]] = []
    for i in range(0, len(inputs), bs):
        batch = inputs[i:i + bs]
        if provider == "mock":
            out.extend(mock.embed(batch))
        elif provider == "ollama":
            data = await _post(f"{_base(cfg)}/api/embed", {"model": cfg["model"], "input": batch}, cfg=cfg)
            out.extend(data["embeddings"])
        elif provider == "openai":
            data = await _post(f"{_base(cfg)}/embeddings", {"model": cfg["model"], "input": batch}, _auth(cfg), cfg)
            out.extend(d["embedding"] for d in sorted(data["data"], key=lambda d: d["index"]))
        else:
            raise ProviderError(f"Unknown embedding provider '{provider}'")
    arr = np.asarray(out, dtype=np.float32)
    arr /= np.maximum(np.linalg.norm(arr, axis=1, keepdims=True), 1e-12)
    if rec is not None:
        tin = sum(_est(t) for t in inputs)
        rec.add(kind="embed", model=cfg.get("model"), n=len(texts), dim=int(arr.shape[1]), tokens_in=tin, tokens_out=0,
                cost=_cost(cfg, tin, 0), ms=round((time.perf_counter() - t0) * 1000))
    return arr


# ---------------------------------------------------------------- chat
_THINK = re.compile(r"<think>.*?</think>", re.S)


def strip_think(text: str) -> str:
    return _THINK.sub("", text).strip()


def extract_json(text: str):
    text = strip_think(text)
    text = re.sub(r"```(?:json)?", "", text)
    for opener, closer in (("{", "}"), ("[", "]")):
        a, b = text.find(opener), text.rfind(closer)
        if a != -1 and b > a:
            try:
                return json.loads(text[a:b + 1])
            except json.JSONDecodeError:
                continue
    return None


def _split_system(messages: list[dict]) -> tuple[str, list[dict]]:
    system = "\n".join(m["content"] for m in messages if m["role"] == "system")
    return system, [m for m in messages if m["role"] != "system"]


async def chat(cfg: dict, messages: list[dict], *, json_mode: bool = False, temperature: float = 0.0,
               max_tokens: int = 1024, task: str = "", rec: Recorder | None = None, cache: "LLMCache | None" = None) -> str:
    t0 = time.perf_counter()
    provider = cfg.get("via") or cfg.get("provider")
    ckey = cache.key(cfg, task, messages) if cache is not None and temperature == 0 else None
    if ckey and (hit := cache.get(ckey)) is not None:
        return hit
    tin = tout = None
    if provider == "mock":
        text = mock.chat(task, messages)
    elif provider == "ollama":
        payload = {"model": cfg["model"], "messages": messages, "stream": False,
                   "options": {"temperature": temperature, "num_predict": max_tokens}}
        if json_mode:
            payload["format"] = "json"
        data = await _post(f"{_base(cfg)}/api/chat", payload, cfg=cfg)
        text = data["message"]["content"]
        tin, tout = data.get("prompt_eval_count"), data.get("eval_count")
    elif provider == "openai":
        payload = {"model": cfg["model"], "messages": messages, "temperature": temperature,
                   "max_tokens": max_tokens}
        if json_mode and cfg.get("json_mode", True):
            payload["response_format"] = {"type": "json_object"}
        data = await _post(f"{_base(cfg)}/chat/completions", payload, _auth(cfg), cfg)
        text = data["choices"][0]["message"]["content"] or ""
        tin, tout = (data.get("usage") or {}).get("prompt_tokens"), (data.get("usage") or {}).get("completion_tokens")
    elif provider == "anthropic":
        system, msgs = _split_system(messages)
        payload = {"model": cfg["model"], "max_tokens": max_tokens, "temperature": temperature, "messages": msgs}
        if system:
            payload["system"] = system
        headers = {"x-api-key": api_key(cfg) or "", "anthropic-version": "2023-06-01"}
        data = await _post(f"{_base(cfg) or 'https://api.anthropic.com/v1'}/messages", payload, headers, cfg)
        text = "".join(b.get("text", "") for b in data["content"] if b["type"] == "text")
        tin, tout = (data.get("usage") or {}).get("input_tokens"), (data.get("usage") or {}).get("output_tokens")
    else:
        raise ProviderError(f"Unknown chat provider '{provider}'")
    estimated = tin is None or tout is None
    tin, tout = int(tin if tin is not None else _est(messages)), int(tout if tout is not None else _est(text))
    if ckey:
        cache.put(ckey, text)
    if rec is not None:
        rec.add(kind="chat", task=task, model=cfg.get("model"), messages=messages, response=text, tokens_in=tin,
                tokens_out=tout, estimated=estimated, cost=_cost(cfg, tin, tout), ms=round((time.perf_counter() - t0) * 1000))
    return text


async def chat_json(cfg: dict, messages: list[dict], *, task: str, rec: Recorder | None = None,
                    max_tokens: int = 1024, default=None, cache: "LLMCache | None" = None):
    """Chat call that must return JSON. One repair retry, then `default`."""
    text = await chat(cfg, messages, json_mode=True, task=task, rec=rec, max_tokens=max_tokens, cache=cache)
    parsed = extract_json(text)
    if parsed is None:
        retry = messages + [{"role": "assistant", "content": text},
                            {"role": "user", "content": "That was not valid JSON. Reply with only the JSON object."}]
        parsed = extract_json(await chat(cfg, retry, json_mode=True, task=task, rec=rec, max_tokens=max_tokens))
    return default if parsed is None else parsed


async def chat_stream(cfg: dict, messages: list[dict], *, temperature: float = 0.1, max_tokens: int = 1500,
                      task: str = "", rec: Recorder | None = None) -> AsyncIterator[str]:
    provider = cfg.get("provider")
    t0, out, usage = time.perf_counter(), [], {}

    def record():
        if rec is None:
            return
        text = "".join(out)
        tin, tout = usage.get("in"), usage.get("out")
        rec.add(kind="chat", task=task, model=cfg.get("model"), messages=messages, response=text,
                tokens_in=int(tin or _est(messages)), tokens_out=int(tout or _est(text)), estimated=not (tin and tout),
                cost=_cost(cfg, int(tin or _est(messages)), int(tout or _est(text))), ms=round((time.perf_counter() - t0) * 1000))

    if provider == "mock":
        for tok in re.findall(r"\S+\s*", mock.chat(task, messages)):
            await asyncio.sleep(0)
            out.append(tok)
            yield tok
        record()
        return
    if provider == "ollama":
        url, headers = f"{_base(cfg)}/api/chat", {}
        payload = {"model": cfg["model"], "messages": messages, "stream": True,
                   "options": {"temperature": temperature, "num_predict": max_tokens}}
    elif provider == "openai":
        url, headers = f"{_base(cfg)}/chat/completions", _auth(cfg)
        payload = {"model": cfg["model"], "messages": messages, "temperature": temperature,
                   "max_tokens": max_tokens, "stream": True}
        if cfg.get("stream_usage", True):  # set "stream_usage": false for servers that reject stream_options
            payload["stream_options"] = {"include_usage": True}
    elif provider == "anthropic":
        system, msgs = _split_system(messages)
        url = f"{_base(cfg) or 'https://api.anthropic.com/v1'}/messages"
        headers = {"x-api-key": api_key(cfg) or "", "anthropic-version": "2023-06-01"}
        payload = {"model": cfg["model"], "max_tokens": max_tokens, "temperature": temperature,
                   "messages": msgs, "stream": True}
        if system:
            payload["system"] = system
    else:
        raise ProviderError(f"Unknown chat provider '{provider}'")
    try:
        async with client().stream("POST", url, json=payload, headers=headers) as r:
            if r.status_code >= 400:
                body = (await r.aread()).decode(errors="replace")
                raise ProviderError(f"{url} returned {r.status_code}: {body[:400]}")
            async for line in r.aiter_lines():  # noqa: PLR1702
                line = line.strip()
                if not line:
                    continue
                if line.startswith("data:"):
                    line = line[5:].strip()
                if line == "[DONE]" or not line.startswith("{"):
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if provider == "ollama":
                    tok = obj.get("message", {}).get("content", "")
                    if obj.get("done"):
                        usage.update({"in": obj.get("prompt_eval_count"), "out": obj.get("eval_count")})
                elif provider == "openai":
                    ch = obj.get("choices") or [{}]
                    tok = (ch[0].get("delta") or {}).get("content") or ""
                    if obj.get("usage"):
                        usage.update({"in": obj["usage"].get("prompt_tokens"), "out": obj["usage"].get("completion_tokens")})
                else:
                    tok = obj.get("delta", {}).get("text", "") if obj.get("type") == "content_block_delta" else ""
                    if obj.get("type") == "message_start":
                        usage["in"] = obj.get("message", {}).get("usage", {}).get("input_tokens")
                    elif obj.get("type") == "message_delta":
                        usage["out"] = obj.get("usage", {}).get("output_tokens")
                if tok:
                    out.append(tok)
                    yield tok
        record()
    except httpx.HTTPError as e:
        raise ProviderError(f"Cannot reach {url}: {e.__class__.__name__} {e}") from e


# ---------------------------------------------------------------- rerank
_cross_encoders: dict = {}

RERANK_PROMPT = """Rate how well the passage answers the query.
Reply with JSON: {{"score": <integer 0-10>}}. 10 = directly and fully answers, 0 = unrelated.

Query: {query}

Passage:
{passage}"""


async def rerank(cfg: dict, query: str, docs: list[str], rec: Recorder | None = None) -> list[float]:
    """Relevance score per doc, same order as `docs`. Higher is better."""
    if not docs:
        return []
    t0 = time.perf_counter()
    provider = cfg.get("provider")
    if provider == "mock":
        scores = mock.rerank(query, docs)
    elif provider == "cohere_compat":
        data = await _post(f"{_base(cfg)}/rerank",
                           {"model": cfg["model"], "query": query, "documents": docs, "top_n": len(docs)}, _auth(cfg), cfg)
        scores = [0.0] * len(docs)
        for item in data.get("results") or data.get("data") or []:
            scores[item["index"]] = float(item.get("relevance_score", item.get("score", 0.0)))
    elif provider == "tei":
        data = await _post(f"{_base(cfg)}/rerank", {"query": query, "texts": docs, "truncate": True}, _auth(cfg), cfg)
        scores = [0.0] * len(docs)
        for item in data:
            scores[item["index"]] = float(item["score"])
    elif provider == "cross_encoder":
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as e:
            raise ProviderError("cross_encoder needs `pip install -e '.[crossencoder]'`") from e
        name = cfg.get("model") or "BAAI/bge-reranker-v2-m3"
        if name not in _cross_encoders:
            _cross_encoders[name] = await asyncio.to_thread(CrossEncoder, name)
        raw = await asyncio.to_thread(_cross_encoders[name].predict, [(query, d) for d in docs])
        scores = [float(s) for s in raw]
    elif provider == "llm":
        sem = asyncio.Semaphore(int(cfg.get("concurrency", 4)))
        inner = Recorder()

        async def one(d: str) -> float:
            async with sem:
                out = await chat_json(cfg, [{"role": "user", "content": RERANK_PROMPT.format(query=query, passage=d[:2500])}],
                                      task="rerank", max_tokens=30, default={"score": 0}, rec=inner)
                try:
                    return float(out.get("score", 0)) / 10.0
                except (TypeError, ValueError, AttributeError):
                    return 0.0
        scores = list(await asyncio.gather(*[one(d) for d in docs]))
    else:
        raise ProviderError(f"Unknown reranker provider '{provider}'")
    if rec is not None:
        if provider == "llm":
            tin, tout = sum(c["tokens_in"] for c in inner.calls), sum(c["tokens_out"] for c in inner.calls)
        else:
            tin, tout = _est(query) * len(docs) + sum(_est(d) for d in docs), 0
        rec.add(kind="rerank", model=cfg.get("model"), provider=provider, n=len(docs), tokens_in=tin, tokens_out=tout,
                cost=_cost(cfg, tin, tout), ms=round((time.perf_counter() - t0) * 1000))
    return scores


# ---------------------------------------------------------------- health
async def health(models: dict, roles: tuple[str, ...] = ("embedding", "reranker", "router", "generator")) -> dict:
    """Probe each role with a real call. :param roles: which roles to check; the default is all four."""
    out = {}

    async def check(role: str):
        cfg, t0 = models[role], time.perf_counter()
        try:
            if role == "embedding":
                v = await embed(cfg, ["health check"], "query")
                detail = f"dim {v.shape[1]}"
            elif role == "reranker":
                s = await rerank(cfg, "capital of France", ["Paris is the capital of France.", "Bananas are yellow."])
                detail = f"scores {s[0]:.2f} vs {s[1]:.2f}" + ("" if s[0] > s[1] else "  (unexpected order)")
            else:
                txt = await chat(cfg, [{"role": "user", "content": "Reply with the single word: ready"}],
                                 max_tokens=2000 if "think" in str(cfg.get("model")) else 20, task="health")
                detail = strip_think(txt)[:60]
            out[role] = {"ok": True, "ms": round((time.perf_counter() - t0) * 1000), "detail": detail}
        except Exception as e:  # noqa: BLE001 - surface everything to the UI
            out[role] = {"ok": False, "ms": round((time.perf_counter() - t0) * 1000), "detail": str(e)[:500]}

    await asyncio.gather(*[check(r) for r in roles if r in models])
    return out


def fingerprint(cfg: dict) -> str:
    return hashlib.sha1(f"{cfg.get('provider')}|{cfg.get('base_url')}|{cfg.get('model')}".encode()).hexdigest()[:10]
