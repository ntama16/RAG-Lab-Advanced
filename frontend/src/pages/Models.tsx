import { useEffect, useState } from "react";
import { api, Any } from "../api";

const ROLES: [string, string, string[]][] = [
  ["embedding", "Turns chunks and queries into vectors. Changing it invalidates existing indexes.", ["ollama", "openai", "mock"]],
  ["reranker", "Scores query-passage pairs jointly. Use a real cross-encoder or rerank API where you can; the LLM option is the fallback.",
    ["cohere_compat", "tei", "cross_encoder", "llm", "mock"]],
  ["router", "Small, fast model for routing, self-query, query rewriting, grading and graph extraction.", ["ollama", "openai", "anthropic", "mock"]],
  ["generator", "Writes HyDE passages, final answers, evaluation questions and judgements.", ["ollama", "openai", "anthropic", "mock"]],
];

const PRESETS: Record<string, Any> = {
  "cohere_compat": { base_url: "http://localhost:8080/v1", model: "bge-reranker-v2-m3" },
  "tei": { base_url: "http://localhost:8081", model: "" },
  "cross_encoder": { base_url: "", model: "BAAI/bge-reranker-v2-m3" },
  "ollama": { base_url: "http://localhost:11434" },
  "openai": { base_url: "https://api.openai.com/v1" },
  "anthropic": { base_url: "https://api.anthropic.com/v1" },
};

export default function Models({ onSaved }: { onSaved: (m: Any) => void }) {
  const [models, setModels] = useState<Any>(null);
  const [health, setHealth] = useState<Any>(null);
  const [busy, setBusy] = useState(false);
  const [ollama, setOllama] = useState<string[]>([]);
  const [msg, setMsg] = useState("");
  // Every hook has to run on every render, so this one stays above the `if (!models) return null`
  // below: declared after it, React saw five hooks on the first render and six on the next and
  // tore the tree down, which is a blank page rather than an error anybody can read.
  const [testing, setTesting] = useState("");

  useEffect(() => {
    api.get("/api/models").then((r) => setModels(r.models)).catch((e) => setMsg(String((e as Error).message)));
    api.get("/api/models/ollama").then((r) => setOllama(r.models)).catch(() => setOllama([]));
  }, []);
  // Never `return null`: with no message and no markup this page is indistinguishable from the
  // crash it actually had, and a failed fetch left it blank for good.
  if (!models) return <div className="panel">{msg
    ? <><h3>The model settings could not be loaded</h3><p className="err" style={{ marginTop: 8 }}>{msg}</p>
        <p className="small muted" style={{ marginTop: 8 }}>Check that the lab server is running, then reload.</p></>
    : <p className="muted">Loading the model settings…</p>}</div>;

  const set = (role: string, key: string, value: Any) => setModels({ ...models, [role]: { ...models[role], [key]: value } });
  // Testing one role before saving: a wrong endpoint used to be written to the config file first,
  // and the failure then arrived for all four at once.
  const test = async (role: string) => {
    setTesting(role);
    try { const r = await api.post("/api/models/test", { role, config: models[role] });
      setHealth((h: Any) => ({ ...h, [role]: r })); }
    catch (e) { setHealth((h: Any) => ({ ...h, [role]: { ok: false, ms: 0, detail: String((e as Error).message) } })); }
    setTesting("");
  };
  const save = async () => {
    setBusy(true); setMsg("");
    try {
      const r = await api.put("/api/models", { models });
      setModels(r.models); onSaved(r);
      setHealth(await api.get("/api/models/health"));
      setMsg(r.saved ? "Saved to raglab.config.json" : "Mock mode: these settings last until the server stops. raglab.config.json was not changed.");
    } catch (e) { setMsg(String(e)); }
    setBusy(false);
  };

  return (
    <>
      <div className="pagehead">
        <h2>Models</h2>
        <p>Four roles, each pointing at a local or cloud endpoint. “openai” means any OpenAI-compatible server: OpenAI, OpenRouter, Groq, LM Studio, vLLM, llama.cpp.</p>
      </div>
      <datalist id="ollama-models">{ollama.map((m) => <option key={m} value={m} />)}</datalist>
      <div className="grid2">
        {ROLES.map(([role, blurb, providers]) => {
          const c = models[role]; const h = health?.[role];
          return (
            <section className="panel stack" key={role}>
              <div className="row"><h3 className="grow" style={{ textTransform: "capitalize" }}>{role}</h3>
                <button className="link" disabled={!!testing} onClick={() => test(role)}>{testing === role ? "Testing…" : "Test"}</button>
                {h && <span className={`tag ${h.ok ? "ok" : "bad"}`}>{h.ok ? `ok ${h.ms} ms` : "failed"}</span>}</div>
              <p className="muted small">{blurb}</p>
              <div className="fields">
                <label className="f"><span>Provider</span>
                  <select value={c.provider} onChange={(e) => setModels({ ...models, [role]: { ...c, provider: e.target.value, ...(PRESETS[e.target.value] ?? {}) } })}>
                    {providers.map((p) => <option key={p}>{p}</option>)}</select></label>
                {c.provider === "llm" && <label className="f"><span>Through</span>
                  <select value={c.via ?? "ollama"} onChange={(e) => set(role, "via", e.target.value)}>
                    <option>ollama</option><option>openai</option><option>anthropic</option></select></label>}
                <label className="f"><span>Model</span>
                  <input type="text" list={c.provider === "ollama" || c.via === "ollama" ? "ollama-models" : undefined}
                    value={c.model ?? ""} onChange={(e) => set(role, "model", e.target.value)} /></label>
              </div>
              {c.provider !== "cross_encoder" && c.provider !== "mock" && <>
                <label className="f"><span>Base URL</span><input type="text" value={c.base_url ?? ""} onChange={(e) => set(role, "base_url", e.target.value)} /></label>
                <div className="fields">
                  <label className="f"><span>API key env var</span><input type="text" placeholder="OPENAI_API_KEY" value={c.api_key_env ?? ""} onChange={(e) => set(role, "api_key_env", e.target.value)} /></label>
                  <label className="f"><span>or API key (stored in local config file)</span><input type="password" value={c.api_key ?? ""} onChange={(e) => set(role, "api_key", e.target.value)} /></label>
                </div></>}
              {c.provider !== "mock" && <div className="fields">
                <label className="f"><span>$ per 1M input tokens</span><input type="number" step="0.01" value={c.price_in ?? 0} onChange={(e) => set(role, "price_in", Number(e.target.value))} /></label>
                <label className="f"><span>$ per 1M output tokens</span><input type="number" step="0.01" value={c.price_out ?? 0} onChange={(e) => set(role, "price_out", Number(e.target.value))} /></label>
                <label className="f"><span>Timeout (s)</span><input type="number" value={c.timeout ?? 180} onChange={(e) => set(role, "timeout", Number(e.target.value))} /></label>
                <label className="f"><span>Retries</span><input type="number" value={c.retries ?? 2} onChange={(e) => set(role, "retries", Number(e.target.value))} /></label></div>}
              {role === "embedding" && <div className="fields">
                <label className="f"><span>Query prefix</span><input type="text" value={c.query_prefix ?? ""} onChange={(e) => set(role, "query_prefix", e.target.value)} /></label>
                <label className="f"><span>Document prefix</span><input type="text" value={c.doc_prefix ?? ""} onChange={(e) => set(role, "doc_prefix", e.target.value)} /></label>
              </div>}
              {h && <div className={h.ok ? "note mono" : "err"}>{h.detail}</div>}
            </section>
          );
        })}
      </div>
      <div className="row" style={{ marginTop: 16 }}>
        <button className="btn" onClick={save} disabled={busy}>{busy ? "Checking…" : "Save and check all four"}</button>
        <span className="muted small">{msg}</span>
      </div>
      <div className="panel" style={{ marginTop: 18 }}>
        <h4>Running a real reranker locally</h4>
        <p className="small muted" style={{ marginTop: 6 }}>Ollama has no rerank endpoint. Three good options: (1) provider <code>cross_encoder</code> runs a sentence-transformers
          cross-encoder in this process (<code>pip install -e '.[crossencoder]'</code>); (2) <code>llama-server -m bge-reranker-v2-m3.gguf --reranking --port 8080</code> with provider
          <code> cohere_compat</code> and base URL <code>http://localhost:8080/v1</code>; (3) HuggingFace text-embeddings-inference with provider <code>tei</code>.
          Cohere, Jina and Voyage cloud rerankers also use <code>cohere_compat</code>.</p>
      </div>
    </>
  );
}
