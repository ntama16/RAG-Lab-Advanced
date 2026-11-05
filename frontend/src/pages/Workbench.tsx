import { createContext, ReactNode, useContext, useEffect, useMemo, useRef, useState } from "react";
import { api, Any, deepMerge, runFromTrace, streamQuery, usePersisted } from "../api";
import { CHANNEL, GraphView, Groups, Legend, makeGroups, RankFlow, ScoreBar, VectorMap, Waterfall } from "../viz";

/** Sample questions, tagged with the sample corpus they belong to. `corpus: ""` fits any index. */
const SAMPLES: { text: string; corpus: string }[] = [
  { text: "Can I send back a ZX-481 after three weeks?", corpus: "halcyon" },
  { text: "Is the Ridgeline battery covered by warranty?", corpus: "halcyon" },
  { text: "What restocking fee applies to an opened power bank and is it waived for DOA?", corpus: "halcyon" },
  { text: "What command warms the Redis cache after an outage, and how long does it take?", corpus: "kestrel" },
  { text: "Why did duplicate scans reach customers during INC-2291?", corpus: "kestrel" },
  { text: "How are the Resolver and the Notifier connected?", corpus: "kestrel" },
  { text: "In the runbooks, what should I do if Redis has not recovered after 15 minutes?", corpus: "kestrel" },
  { text: "How many days without a scan before a parcel is marked Lost?", corpus: "kestrel" },
  { text: "What is Northlight's policy on drone deliveries?", corpus: "" },
  { text: "hello", corpus: "" },
];

/* ---------- configuration column ---------- */
function Stage({ n, title, on, toggle, hint, children }: { n: number; title: string; on: boolean; toggle?: (v: boolean) => void; hint?: string; children?: ReactNode }) {
  return (
    <div className={`stage ${on ? "on" : ""}`} data-n={n}>
      {toggle ? <label className="check"><input type="checkbox" checked={on} onChange={(e) => toggle(e.target.checked)} />{title}</label>
        : <div style={{ fontWeight: 500, marginLeft: 4 }}>{title}</div>}
      {hint && <div className="hint" style={toggle ? undefined : { marginLeft: 0 }}>{hint}</div>}
      {on && children}
    </div>
  );
}

type Bounds = Record<string, { min: number; max: number; int: boolean }>;
/** Server-declared bounds for the pipeline's numeric settings, mirrored from raglab/pipeline.py BOUNDS. */
const BoundsCtx = createContext<Bounds>({});

/** Numeric field that refuses to emit a value its bound rejects, so a half-typed or cleared box
 *  cannot reach the pipeline as 0 or NaN. `bounds` comes from the server (GET /api/defaults). */
function Num({ label, value, set, step, path }: { label: string; value: number; set: (v: number) => void; step?: number; path?: string }) {
  const bounds = useContext(BoundsCtx);
  const b = path ? bounds[path] : undefined;
  const [raw, setRaw] = useState<string | null>(null);
  const shown = raw ?? String(value);
  const parsed = Number(shown);
  const bad = shown.trim() === "" || Number.isNaN(parsed) || (b && (parsed < b.min || parsed > b.max || (b.int && !Number.isInteger(parsed))));
  return <label className="f"><span>{label}</span>
    <input type="number" step={step} min={b?.min} max={b?.max} value={shown} aria-invalid={bad || undefined}
      onChange={(e) => { setRaw(e.target.value); const n = Number(e.target.value);
        if (e.target.value.trim() !== "" && !Number.isNaN(n) && (!b || (n >= b.min && n <= b.max && (!b.int || Number.isInteger(n))))) set(n); }}
      onBlur={() => setRaw(null)} />
    {bad && b && <span className="small bad">{b.int ? "whole number" : "number"} between {b.min} and {b.max}</span>}</label>;
}

function Presets({ p, set }: { p: Any; set: (p: Any) => void }) {
  const [list, setList] = useState<Any[]>([]); const [name, setName] = usePersisted("raglab.workbench.presetName", ""); const file = useRef<HTMLInputElement>(null);
  const load = () => api.get("/api/presets").then(setList); useEffect(() => { load(); }, []);
  const exportJson = () => { const a = document.createElement("a"); a.href = URL.createObjectURL(new Blob([JSON.stringify({ name: name || "pipeline", config: p }, null, 2)], { type: "application/json" })); a.download = `${name || "pipeline"}.raglab.json`; a.click(); };
  const importJson = async (f?: File) => { if (!f) return; try { const j = JSON.parse(await f.text()); const cfg = j.config ?? j; set(deepMerge(p, cfg)); if (j.name) { await api.put(`/api/presets/${encodeURIComponent(j.name)}`, { config: cfg }); load(); } } catch { alert("That file is not a pipeline preset."); } };
  return (
    <div className="stack" style={{ marginBottom: 14, paddingBottom: 12, borderBottom: "1px solid var(--rule)" }}>
      <label className="f"><span>Presets</span><select value="" onChange={(e) => { const x = list.find((l) => l.name === e.target.value); if (x) { set(deepMerge(p, x.config)); setName(x.name); } }}>
        <option value="">{list.length ? "Load a saved preset…" : "No saved presets yet"}</option>{list.map((l) => <option key={l.name}>{l.name}</option>)}</select></label>
      <div className="row"><input className="grow" type="text" placeholder="Name this configuration" value={name} onChange={(e) => setName(e.target.value)} />
        <button className="btn ghost" disabled={!name.trim()} onClick={async () => { await api.put(`/api/presets/${encodeURIComponent(name.trim())}`, { config: p }); load(); }}>Save</button></div>
      <div className="row small"><button className="link" onClick={exportJson}>Export JSON</button><button className="link" onClick={() => file.current?.click()}>Import JSON</button>
        {list.some((l) => l.name === name) && <button className="link" onClick={async () => { await api.del(`/api/presets/${encodeURIComponent(name)}`); setName(""); load(); }}>Delete “{name}”</button>}
        <input ref={file} type="file" accept=".json" hidden onChange={(e) => importJson(e.target.files?.[0])} /></div>
    </div>
  );
}

/** Named starting points from GET /api/defaults, each saying which model roles it needs and roughly
 *  how many requests one question costs — so a first run is a choice, not a surprise. */
function Starters({ presets, defaults, set }: { presets: Any[]; defaults: Any; set: (p: Any) => void }) {
  const [open, setOpen] = useState(false);
  if (!presets?.length) return null;
  // Applied against the defaults, never against whatever is currently on. Merging into the current
  // pipeline left self-query, graph retrieval, caching and community reports enabled underneath a
  // preset whose whole claim is that it needs two model roles and a handful of requests.
  return (
    <div className="stack" style={{ marginBottom: 12 }}>
      <div className="row"><b className="small grow">Start from</b><button className="link" onClick={() => setOpen(!open)}>{open ? "Hide" : "What do these cost?"}</button></div>
      <div className="row" style={{ gap: 6, flexWrap: "wrap" }}>
        {presets.map((s: Any) => <button key={s.name} className="btn ghost small" onClick={() => set(deepMerge(defaults, s.config))}>{s.name}</button>)}
      </div>
      {open && presets.map((s: Any) => <div key={s.name} className="note small" style={{ marginTop: 4 }}>
        <b>{s.name}</b> — {s.about}<div className="muted" style={{ marginTop: 3 }}>Needs: {s.roles.join(", ")}</div></div>)}
    </div>
  );
}

function Config({ p, set, reset, defaults, hasGraph, hasCommunities, presets, check }: { p: Any; set: (p: Any) => void; reset: () => void; defaults: Any; hasGraph: boolean; hasCommunities: boolean; presets: Any[]; check: Any }) {
  const up = (k: string, patch: Any) => set({ ...p, [k]: { ...p[k], ...patch } });
  const Sub = ({ k, label, children }: { k: string; label: ReactNode; children?: ReactNode }) => <><label className="check small" style={{ marginTop: 6 }}><input type="checkbox" checked={p[k].enabled} onChange={(e) => up(k, { enabled: e.target.checked })} />{label}</label>{p[k].enabled && children}</>;
  return (
    <div className="panel config">
      <div className="row" style={{ marginBottom: 12 }}><h3 className="grow">Pipeline</h3><button className="link" onClick={reset}>Reset</button></div>
      <Starters presets={presets} defaults={defaults} set={set} />
      <Presets p={p} set={set} />
      {!!check?.calls && <div className="note small" style={{ marginBottom: 10 }}>
        About <b>{check.calls.total}</b> model request{check.calls.total === 1 ? "" : "s"} per question:{" "}
        {(["embedding", "router", "reranker", "generator"] as const).filter((r) => check.calls[r]).map((r) => `${check.calls[r]} ${r}`).join(", ")}.
        {check.calls.reranker > 0 && <div className="muted" style={{ marginTop: 3 }}>{check.calls.note}</div>}</div>}
      {check?.errors?.map((e: Any) => <div className="err small" key={e.field} style={{ marginBottom: 6 }}><b>{e.field}</b> {e.message}</div>)}
      {check?.warnings?.map((w: string, n: number) => <div className="note small" key={n} style={{ marginBottom: 6 }}>{w}</div>)}
      <Stage n={1} title="Semantic cache" on={p.cache.enabled} toggle={(v) => up("cache", { enabled: v })} hint="A close-enough earlier question under identical settings returns instantly. Always off in evaluations.">
        <div className="fields"><Num path="cache.threshold" label="Cosine threshold" step={0.01} value={p.cache.threshold} set={(v) => up("cache", { threshold: v })} /></div></Stage>
      <Stage n={2} title="Query normalisation" on={p.normalize.enabled} toggle={(v) => up("normalize", { enabled: v })} hint="Aliases, acronyms, corpus terminology, durations, spelling. Rules live in the Normalisation lab." />
      <Stage n={3} title="Route and intent gate" on={p.route.enabled} toggle={(v) => up("route", { enabled: v })} hint="Router model classifies the query and can skip retrieval.">
        <label className="check small" style={{ marginTop: 6 }}><input type="checkbox" checked={p.route.apply} onChange={(e) => up("route", { apply: e.target.checked })} />Let the route change the settings below</label></Stage>
      <Stage n={4} title="Self-query metadata filters" on={p.self_query.enabled} toggle={(v) => up("self_query", { enabled: v })} hint="Pulls filters like doc_type = runbook out of the question.">
        <label className="check small" style={{ marginTop: 6 }}><input type="checkbox" checked={!!p.self_query.relax_empty} onChange={(e) => up("self_query", { relax_empty: e.target.checked })} />Relax filters that match nothing (otherwise: no matching evidence)</label></Stage>
      <Stage n={5} title="Query transformation" on>
        <Sub k="decompose" label="Decompose into sub-questions"><div className="fields"><Num path="decompose.n" label="At most" value={p.decompose.n} set={(v) => up("decompose", { n: v })} /></div></Sub>
        <Sub k="step_back" label="Step-back question" />
        <Sub k="multi_query" label="Multi-query variants"><div className="fields"><Num path="multi_query.n" label="Variants" value={p.multi_query.n} set={(v) => up("multi_query", { n: v })} /></div></Sub>
        <Sub k="hyde" label="HyDE (embed a hypothetical answer)" />
      </Stage>
      <Stage n={6} title="Retrieval" on>
        <Sub k="bm25" label={<span className="tag bm25">bm25</span>}><div className="fields"><Num path="bm25.k" label="k" value={p.bm25.k} set={(v) => up("bm25", { k: v })} /><Num path="bm25.k1" label="k1" step={0.1} value={p.bm25.k1} set={(v) => up("bm25", { k1: v })} /><Num path="bm25.b" label="b" step={0.05} value={p.bm25.b} set={(v) => up("bm25", { b: v })} /></div></Sub>
        <Sub k="dense" label={<span className="tag dense">dense</span>}><div className="fields"><Num path="dense.k" label="k" value={p.dense.k} set={(v) => up("dense", { k: v })} />
          <label className="f"><span>Search</span><select value={p.dense.mode} onChange={(e) => up("dense", { mode: e.target.value })}><option value="hnsw">HNSW</option><option value="exact">exact</option></select></label>
          {p.dense.mode === "hnsw" && <Num path="dense.ef_search" label="ef_search" value={p.dense.ef_search} set={(v) => up("dense", { ef_search: v })} />}</div></Sub>
        <Sub k="graph" label={<><span className="tag graph">graph</span>{!hasGraph && <span className="muted">index has no graph</span>}</>}><div className="fields"><Num path="graph.k" label="k" value={p.graph.k} set={(v) => up("graph", { k: v })} /><Num path="graph.alpha" label="PPR restart α" step={0.05} value={p.graph.alpha} set={(v) => up("graph", { alpha: v })} /></div></Sub>
        <Sub k="freshness" label="Freshness filter (drop superseded documents)" />
      </Stage>
      <Stage n={7} title="Fusion" on>
        <div className="fields">
          <label className="f"><span>Method</span><select value={p.fusion.method} onChange={(e) => up("fusion", { method: e.target.value })}><option value="rrf">reciprocal rank</option><option value="weighted">weighted scores</option></select></label>
          {p.fusion.method === "rrf" && <Num path="fusion.rrf_k" label="RRF k" value={p.fusion.rrf_k} set={(v) => up("fusion", { rrf_k: v })} />}
          <Num path="fusion.top_n" label="Keep top" value={p.fusion.top_n} set={(v) => up("fusion", { top_n: v })} />
          {["bm25", "dense", "graph"].map((r) => <Num key={r} path={`fusion.weights.${r}`} label={`w ${r}`} step={0.1} value={p.fusion.weights[r]} set={(v) => up("fusion", { weights: { ...p.fusion.weights, [r]: v } })} />)}
        </div></Stage>
      <Stage n={8} title="Rerank" on={p.rerank.enabled} toggle={(v) => up("rerank", { enabled: v })}>
        <div className="fields"><Num path="rerank.top_n" label="Keep top" value={p.rerank.top_n} set={(v) => up("rerank", { top_n: v })} /></div></Stage>
      <Stage n={9} title="Corrective RAG" on={p.crag.enabled} toggle={(v) => up("crag", { enabled: v })} hint="Grades each passage; rewrites and re-retrieves when evidence is weak; abstains when there is none.">
        <div className="fields"><Num path="crag.max_retries" label="Corrective rounds" value={p.crag.max_retries} set={(v) => up("crag", { max_retries: v })} /></div>
        <label className="check small" style={{ marginTop: 6 }}><input type="checkbox" checked={p.crag.refine} onChange={(e) => up("crag", { refine: e.target.checked })} />Knowledge refinement (keep only relevant strips)</label></Stage>
      <Stage n={10} title="Context assembly" on>
        <div className="fields"><Num path="context.max_chunks" label="Max passages" value={p.context.max_chunks} set={(v) => up("context", { max_chunks: v })} /><Num path="context.token_budget" label="Token budget" value={p.context.token_budget} set={(v) => up("context", { token_budget: v })} />
          <label className="f"><span>Order</span><select value={p.context.order} onChange={(e) => up("context", { order: e.target.value })}><option value="rank">by rank</option><option value="edges">best at both ends</option></select></label></div>
        <label className="check small" style={{ marginTop: 6 }}><input type="checkbox" checked={p.context.expand_parents} onChange={(e) => up("context", { expand_parents: e.target.checked })} />Expand to parent passage</label>
        <label className="check small" style={{ marginTop: 4 }}><input type="checkbox" checked={p.context.mmr.enabled} onChange={(e) => up("context", { mmr: { ...p.context.mmr, enabled: e.target.checked } })} />MMR diversity</label>
        {p.context.mmr.enabled && <div className="fields"><Num path="context.mmr.lambda" label="λ" step={0.05} value={p.context.mmr.lambda} set={(v) => up("context", { mmr: { ...p.context.mmr, lambda: v } })} /></div>}
        <Sub k="communities" label={<>Add graph community reports{!hasCommunities && <span className="muted">index has none</span>}</>}><div className="fields"><Num path="communities.k" label="Reports" value={p.communities.k} set={(v) => up("communities", { k: v })} /></div></Sub>
      </Stage>
      <Stage n={11} title="Generate cited answer" on={p.generate.enabled} toggle={(v) => up("generate", { enabled: v })}>
        <div className="fields"><Num path="generate.temperature" label="Temperature" step={0.1} value={p.generate.temperature} set={(v) => up("generate", { temperature: v })} /></div></Stage>
      <Stage n={12} title="Groundedness check" on={p.verify.enabled} toggle={(v) => up("verify", { enabled: v })} hint="Splits the answer into claims and checks each against the passages it was given." />
    </div>
  );
}

/* ---------- trace column ---------- */
type Chunks = Record<number, Any>;

function Hit({ i, rank, score, chunks, extra, max, color }: { i: number; rank: number; score?: number | string; chunks: Chunks; extra?: ReactNode; max?: number; color?: string }) {
  const [open, setOpen] = useState(false); const c = chunks[i];
  return (
    <div className={`hit ${open ? "open" : ""}`} onClick={() => setOpen(!open)} title={`chunk #${i}`}>
      <span className="r">{rank}</span>
      <div className="t"><b>{c?.meta?.section?.split(" > ").pop() || c?.meta?.title || `#${i}`}</b> {extra}<span className="snip">{c ? (open ? c.text : c.text.replace(/^#+ .*\n/, "").slice(0, 140)) : "(not loaded)"}</span></div>
      <span className="s">{typeof score === "number" ? score.toFixed(score < 1 ? 4 : 2) : score}</span>
      {typeof score === "number" && max !== undefined && <ScoreBar value={score} max={max} color={color} />}
    </div>
  );
}

type Ctx = { chunks: Chunks; detail: Any; groups: Groups };
const flowLabel = (c: Any, i: number) => c ? `${(c.meta.title ?? "").slice(0, 16)} › ${(c.meta.section?.split(" > ").pop() ?? "").slice(0, 16)}` : `#${i}`;

function Body({ ev, ctx }: { ev: Any; ctx: Ctx }) {
  const d = ev.data; const { chunks, detail, groups } = ctx;
  const colorOf = (i: number) => groups.color(groups.of(chunks[i]?.doc_id ?? ""));
  switch (ev.stage) {
    case "cache": return <div className="row"><span className={`tag ${d.hit ? "ok" : ""}`}>{d.hit ? "hit" : "miss"}</span>
      <span>{d.entries ? <>Closest cached question at cosine <b className="mono">{d.similarity}</b> (threshold {d.threshold}){d.matched ? <>: “{d.matched}”</> : null}</> : "The cache is empty for this index and these settings."}</span></div>;
    case "condense": return <table><tbody><tr><td style={{ width: 110 }}><span className="tag">follow-up</span></td><td>{d.original}</td></tr><tr><td><span className="tag ok">standalone</span></td><td><b>{d.standalone}</b></td></tr></tbody></table>;
    case "normalize": return <div className="stack"><div>Search text: <b>{d.normalized}</b></div>
      {d.steps.length ? <table><tbody>{d.steps.map((st: Any, n: number) => <tr key={n}><td style={{ width: 100 }}><span className="tag ok">{st.rule}</span></td><td className="mono small" style={{ width: "34%" }}>{st.from} → {st.to}</td><td className="small muted">{st.note}</td></tr>)}</tbody></table> : <span className="muted">No rule applied: the query already uses the corpus's wording.</span>}
      {!!d.protected?.length && <div className="row small"><span className="muted">Protected identifiers</span>{d.protected.map((t: string) => <span className="tag" key={t}>{t}</span>)}</div>}</div>;
    case "verify": { const n = d.claims.length; const ok = d.claims.filter((c: Any) => c.verdict === "supported").length;
      return <div className="stack">{n ? <><div className="row small"><span>{ok} of {n} claims supported</span><div className="stackbar grow">{d.claims.map((c: Any, k: number) => <i key={k} style={{ flex: 1, background: c.verdict === "supported" ? "var(--ok)" : "var(--bad)" }} />)}</div></div>
        <table><tbody>{d.claims.map((c: Any, k: number) => <tr key={k}><td style={{ width: 110 }}><span className={`tag ${c.verdict === "supported" ? "ok" : "bad"}`}>{c.verdict}</span></td><td>{c.claim}</td><td className="num">{c.passage ? `[${c.passage}]` : ""}</td></tr>)}</tbody></table></> : <span className="muted">The checker returned no claims.</span>}</div>; }
    case "route": return <div className="stack">
      <div className="row"><span className="tag ok">{d.intent}</span><span className="tag">{d.needs_retrieval === false ? "no retrieval" : "retrieve"}</span>{d.in_scope === false && <span className="tag bad">likely out of scope</span>}<span>{d.reason}</span></div>
      {!!d.applied?.length && <div className="note">Applied: {d.applied.join(", ")}</div>}</div>;
    case "self_query": return <div className="stack">
      <div>Search text: <b>{d.search_query}</b></div>
      <div className="row">{d.filters.length ? d.filters.map((f: Any, n: number) => <span className="tag ok" key={n}>{f.field} {f.op} {JSON.stringify(f.value)}</span>) : <span className="muted">No filters extracted.</span>}
        {d.rejected.map((f: Any, n: number) => <span className="tag bad" key={n}>rejected: {JSON.stringify(f)}</span>)}</div>
      <div className="small muted">{d.matching_chunks} of {d.total_chunks} chunks searchable. {d.note}</div></div>;
    case "transform": return <table><tbody>{d.queries.map((q: Any, n: number) => <tr key={n}><td style={{ width: 110 }}><span className="tag">{q.label}</span></td><td>{q.text}</td></tr>)}</tbody></table>;
    case "retrieve": {
      const denseLists = d.lists.filter((l: Any) => l.retriever === "dense");
      const denseHits = new Set<number>(denseLists.flatMap((l: Any) => l.hits.slice(0, 10).map((h: Any) => h.i)));
      const links = (denseLists[0]?.hits ?? []).slice(0, 6).map((h: Any) => h.i);
      return <div className="stack">
        {!!d.filtered_out?.length && <div className="note"><b>Freshness filter</b> removed {d.filtered_out.length} candidate{d.filtered_out.length > 1 ? "s" : ""}:
          {d.filtered_out.slice(0, 5).map((f: Any) => <div key={f.i} className="small mono">#{f.i} {chunks[f.i]?.meta?.title ?? ""} — {f.reason}</div>)}</div>}
        {d.hnsw_recall_vs_exact !== undefined && <div className="note">HNSW recall against exact search for this query: <b className="mono">{(d.hnsw_recall_vs_exact * 100).toFixed(1)}%</b></div>}
        <div className="lists">{d.lists.map((l: Any, n: number) => <div className={`list ${l.retriever}`} key={n}>
          <div className="row" style={{ marginBottom: 4 }}><span className={`tag ${l.retriever}`}>{l.retriever}</span><span className="small muted">{l.query}, {l.hits.length} hits</span></div>
          {l.idf && <div className="small muted" style={{ marginBottom: 4 }}>idf {Object.entries(l.idf).map(([t, v]) => `${t} ${v}`).join("  ")}</div>}
          {l.hits.slice(0, 6).map((h: Any) => <Hit key={h.i} i={h.i} rank={h.rank} score={h.score} chunks={chunks} max={l.hits[0].score} color={CHANNEL[l.retriever]} />)}
          {!l.hits.length && <div className="small muted">Nothing matched.</div>}</div>)}</div>
        {d.graph && (d.graph.skipped ? <div className="note">{d.graph.skipped}</div> : <div>
          <h4>Graph walk</h4>
          {d.graph.subgraph.skipped && <div className="note small">{d.graph.subgraph.skipped.note}</div>}
          <div className="row" style={{ margin: "6px 0" }}>{d.graph.links.map((l: Any, n: number) => <span className="tag graph" key={n}>“{l.mention}” → {l.node} ({l.how} {l.score.toFixed(2)})</span>)}
            {!d.graph.links.length && <span className="muted small">No query entity matched a graph node ({d.graph.mentions.join(", ") || "none found"}).</span>}</div>
          {!!d.graph.subgraph.nodes.length && <><GraphView sub={d.graph.subgraph} /><div className="small muted" style={{ marginTop: 4 }}>Filled nodes are the entities linked from the query. Node size is personalised PageRank mass after the walk.
            {d.graph.subgraph.entities !== undefined && <> Graph: {d.graph.subgraph.entities} entities, {d.graph.subgraph.graph_edges} edges.</>}</div></>}
          {!!d.graph.subgraph.edges.length && <table style={{ marginTop: 6, maxWidth: 640 }}><tbody>{d.graph.subgraph.edges.slice(0, 10).map((e: Any, n: number) => <tr key={n}><td>{e.s}</td><td className="muted">{e.r}</td><td>{e.t}</td></tr>)}</tbody></table>}</div>)}
        {d.projection && detail?.projection?.length > 2 && <div><h4 style={{ marginBottom: 6 }}>Where the query landed</h4>
          <VectorMap points={detail.projection} groups={groups} hits={denseHits} rings={new Set<number>(links)} height={380}
            queries={d.projection.map((q: Any, n: number) => ({ ...q, label: q.kind === "original" ? ev.query ?? q.label : q.label, links: n === 0 ? links : undefined }))}
            tip={(i) => ({ title: detail.chunks[i]?.title ?? `#${i}`, sub: detail.chunks[i]?.section?.split(" > ").pop(), text: detail.chunks[i]?.text ?? "" })} />
          <div style={{ marginTop: 8 }}><Legend groups={groups} /></div>
          <div className="small muted" style={{ marginTop: 4 }}>PCA, {Math.round((detail.pca_variance ?? 0) * 100)}% of variance kept. The star is the query, dashed links join it to its six nearest chunks by cosine, small diamonds are query variants and the HyDE passage. Bright markers are the top dense hits.</div></div>}
      </div>;
    }
    case "fuse": {
      const found = new Set<string>(ev.lists?.map((l: Any) => l.retriever) ?? Object.keys(d.fused[0]?.parts ?? {})); const names = ["bm25", "dense", "graph"].filter((r) => found.has(r));
      const best = (r: string) => { const m = new Map<number, number>(); (ev.lists ?? []).filter((l: Any) => l.retriever === r).forEach((l: Any) => l.hits.forEach((h: Any) => m.set(h.i, Math.min(m.get(h.i) ?? 1e9, h.rank)))); return m; };
      const maps = names.map(best); const top = d.fused.slice(0, 10); const fr = new Map<number, number>(d.fused.map((h: Any) => [h.i, h.rank]));
      return <div className="stack">
        <div className="note mono">{ev.title.includes("rrf") ? `score = Σ weight / (${ev.rrfK ?? 60} + rank) over ${names.join(", ")} lists` : "score = Σ weight × min-max(score) per retriever"}&emsp;weights {names.map((r) => `${r} ${d.weights[r]}`).join("  ")}</div>
        {!!ev.lists && <RankFlow items={top.map((h: Any) => ({ id: h.i, label: flowLabel(chunks[h.i], h.i), color: colorOf(h.i) }))}
          columns={[...names.map((r, k) => ({ label: `${r} rank`, color: CHANNEL[r], rank: (id: number) => maps[k].get(id) ?? null })), { label: "Fused rank", rank: (id: number) => fr.get(id) ?? null }]}
          note="Follow a passage's line: where it sat in each list (best rank across query variants) and where fusion placed it. Dashed means a retriever never returned it. Hover a line to isolate it." />}
        <details><summary className="small muted">Fused list with per-retriever contributions</summary>{d.fused.slice(0, 12).map((h: Any) => <Hit key={h.i} i={h.i} rank={h.rank} score={h.score} chunks={chunks} max={d.fused[0].score} color={colorOf(h.i)}
          extra={<>{Object.entries(h.parts).map(([r, v]) => <span key={r} className={`tag ${r}`} style={{ marginRight: 3 }}>{r} {(v as number).toFixed(3)}</span>)}</>} />)}</details></div>;
    }
    case "rerank": { const lo = Math.min(...d.reranked.map((h: Any) => h.score), 0); const hi = Math.max(...d.reranked.map((h: Any) => h.score)) - lo;
      return <div className="stack"><div className="small muted">{d.provider} / {d.model}</div>
        <RankFlow items={d.reranked.map((h: Any) => ({ id: h.i, label: flowLabel(chunks[h.i], h.i), color: colorOf(h.i) }))}
          columns={[{ label: "Fused rank", rank: (id) => d.reranked.find((h: Any) => h.i === id)?.from_rank ?? null }, { label: "Reranked", color: "var(--ok)", rank: (id) => d.reranked.find((h: Any) => h.i === id)?.rank ?? null }]} />
        <div>{d.reranked.map((h: Any) => { const mv = h.from_rank - h.rank; return <Hit key={h.i} i={h.i} rank={h.rank} score={h.score} chunks={chunks} max={hi || 1} color="var(--ok)"
          extra={<span className={`move ${mv > 0 ? "up" : mv < 0 ? "down" : ""}`}>{mv > 0 ? `▲${mv}` : mv < 0 ? `▼${-mv}` : "="} from {h.from_rank}</span>} />; })}</div></div>; }
    case "crag": return <div className="stack">
      <div className="row"><span className={`tag ${d.verdict === "correct" ? "ok" : d.verdict === "incorrect" ? "bad" : "partial"}`}>{d.verdict}</span><span>{d.action}</span></div>
      {(() => { const all = d.rounds.flatMap((r: Any) => r.grades); const cnt = (g: string) => all.filter((x: Any) => x.grade === g).length; const col: Record<string, string> = { relevant: "var(--ok)", partial: "var(--bm25)", irrelevant: "var(--bad)" };
        return <div><div className="stackbar">{all.map((g: Any, k: number) => <i key={k} title={`#${g.i}: ${g.grade}`} style={{ flex: 1, background: col[g.grade] }} />)}</div>
          <div className="legend small" style={{ marginTop: 5 }}>{Object.keys(col).map((g) => <span key={g}><svg width="10" height="10"><rect width="10" height="10" rx="2" fill={col[g]} /></svg>{g} <b className="mono">{cnt(g)}</b></span>)}</div></div>; })()}
      {!!d.refined?.length && <div><h4>Knowledge refinement</h4>{d.refined.map((r: Any) => <div key={r.i} className="row small" style={{ marginTop: 4 }}><span className="mono muted" style={{ width: 170 }}>#{r.i} kept {r.kept.length} of {r.strips} strips</span>
        <div className="stackbar grow" style={{ height: 10 }}>{Array.from({ length: r.strips }, (_, k) => <i key={k} style={{ flex: 1, background: r.kept.includes(k + 1) ? "var(--ok)" : "var(--rule)" }} />)}</div><span className="mono muted">{r.tokens_before} → {r.tokens_after} tok</span></div>)}</div>}
      {d.rounds.map((r: Any, n: number) => <div key={n}><h4>{n === 0 ? "Grading" : `Corrective round ${n}`} <span className="muted" style={{ fontWeight: 400 }}>“{r.query}”</span></h4>
        {r.grades.map((g: Any, k: number) => <Hit key={g.i} i={g.i} rank={k + 1} chunks={chunks} score="" extra={<><span className={`tag ${g.grade}`}>{g.grade}</span> <span className="small muted">{g.reason}</span></>} />)}
        {!r.grades.length && <div className="small muted">No new passages found.</div>}</div>)}</div>;
    case "context": return <div id="ctx-blocks"><div className="row small muted" style={{ marginBottom: 6 }}><span>{d.tokens} of {d.budget} tokens</span>
      {d.prompt_overhead_tokens !== undefined && <span className="muted">+ {d.prompt_overhead_tokens} instructions, {d.output_allowance_tokens} reserved for the answer</span>}</div><div className="stackbar" style={{ marginBottom: 8 }}>{d.blocks.map((b: Any) => <i key={b.n} title={`[${b.n}] ${b.title}: ${b.tokens} tokens`} style={{ width: `${(100 * b.tokens) / d.budget}%`, background: b.i < 0 ? "var(--graph)" : colorOf(b.i) }} />)}</div>
      {d.blocks.map((b: Any) => <div className="chunkbox" key={b.n} id={`ctx-${b.n}`} style={{ borderLeftColor: b.i < 0 ? "var(--graph)" : colorOf(b.i) }}><span className="tag">[{b.n}]</span> <b>{b.title}</b> <span className="muted">{b.section}</span> <span className="tag">{b.tokens} tok</span> {b.expanded && <span className="tag ok">parent</span>} {b.refined && <span className="tag ok">refined</span>} {b.overflow && <span className="tag bad">over budget</span>} {b.community !== undefined && <span className="tag graph">community report {b.similarity}</span>}
        <div className="small">{b.text.slice(0, 260)}{b.text.length > 260 ? "…" : ""}</div></div>)}
      {!d.blocks.length && <div className="note">No passages survived. The generator will be told there is no evidence.</div>}</div>;
    case "generate": return <div className="stack">{d.cited && <div className="row small"><span className="muted">Cited</span>{d.cited.map((n: number) => <span className="tag ok" key={n}>[{n}]</span>)}
      {!!d.uncited?.length && <><span className="muted">supplied but unused</span>{d.uncited.map((n: number) => <span className="tag" key={n}>[{n}]</span>)}</>}</div>}
      {d.messages && <details><summary className="small muted">Full prompt sent to {d.model}</summary><pre>{d.messages.map((m: Any) => `## ${m.role}\n${m.content}`).join("\n\n")}</pre></details>}</div>;
    default: return <pre>{JSON.stringify(d, null, 2)}</pre>;
  }
}

function Event({ ev, n, ctx }: { ev: Any; n: number; ctx: Ctx }) {
  const [open, setOpen] = useState(true);
  const chats = ev.calls.filter((c: Any) => c.kind === "chat");
  return (
    <section className="ev">
      <header onClick={() => setOpen(!open)}><span className="tag">{n}</span><h3>{ev.title}</h3>
        {ev.calls.length > 0 && <span className="small muted">{ev.calls.length} model call{ev.calls.length > 1 ? "s" : ""}{ev.usage ? `, ${ev.usage.tokens_in}↑ ${ev.usage.tokens_out}↓ tok` : ""}{ev.usage?.cost ? `, $${ev.usage.cost.toFixed(4)}` : ""}</span>}<span className="ms">{ev.ms} ms</span></header>
      {open && <div className="body"><Body ev={ev} ctx={ctx} />
        {chats.length > 0 && <details style={{ marginTop: 10 }}><summary className="small muted">Prompts and raw outputs ({chats.length})</summary>
          {chats.slice(0, 12).map((c: Any, k: number) => <pre key={k} style={{ marginTop: 6 }}>{`[${c.task}] ${c.model}  ${c.ms} ms  ${c.tokens_in}↑ ${c.tokens_out}↓ tokens${c.estimated ? " (estimated)" : ""}\n\n${c.messages[c.messages.length - 1].content}\n\n>>> ${c.response}`}</pre>)}</details>}</div>}
    </section>
  );
}

/** The answer, with every [n] a control that reveals the passage it cites. */
function Answer({ text, onCite }: { text: string; onCite?: (n: number) => void }) {
  const parts = text.replace(/<think>[\s\S]*?(<\/think>|$)/g, "").split(/(\[\d+\])/g);
  return <div className="answer">{parts.map((p, i) => {
    const m = /^\[(\d+)\]$/.exec(p);
    if (!m) return p;
    const n = Number(m[1]);
    return onCite
      ? <sup key={i}><button className="cite" onClick={() => onCite(n)} title={`Show passage ${n}`}
                             aria-label={`Show passage ${n}`}>{p}</button></sup>
      : <sup key={i}>{p}</sup>;
  })}</div>;
}

export default function Workbench({ pipeline, setPipeline, defaults, bounds, presets, goto }: { pipeline: Any; setPipeline: (p: Any) => void; defaults: Any; bounds: Bounds; presets: Any[]; goto: (t: string) => void }) {
  const [indexes, setIndexes] = useState<Any[]>([]); const [indexId, setIndexId] = useState(localStorage.getItem("raglab.index") ?? "");
  const [detail, setDetail] = useState<Any>(null); const [query, setQuery] = usePersisted("raglab.workbench.query", SAMPLES[0].text);
  const [events, setEvents] = useState<Any[]>([]); const [chunks, setChunks] = useState<Chunks>({});
  const [answer, setAnswer] = useState(""); const [running, setRunning] = useState(false); const [err, setErr] = useState(""); const [total, setTotal] = useState<number | null>(null);
  const abort = useRef<AbortController | null>(null); const [ranQuery, setRanQuery] = useState("");
  const [usage, setUsage] = useState<Any>(null);
  // Every query is already stored server-side as a trace. Keeping its id (not the trace itself,
  // which carries whole passages and prompts) lets the last run come back when the tab does, and
  // costs a few bytes of localStorage instead of megabytes.
  const [traceId, setTraceId] = usePersisted<string>("raglab.workbench.trace", "");
  const [history_, setHistory_] = useState<Any[]>([]);
  const [reopening, setReopening] = useState(false);

  /** Rebuild the view from a stored trace. Shared with Compare, which restores two of them. */
  const openTrace = async (tid: string, { remember = true } = {}) => {
    setReopening(true); setErr("");
    try {
      const run = await runFromTrace(tid);
      setEvents(run.events); setChunks(run.chunks as Chunks); setAnswer(run.answer);
      setTotal(run.total); setUsage(run.usage);
      const q = run.events.find((e: Any) => e.stage === "retrieve")?.query
        ?? (await api.get(`/api/traces/${tid}`)).query;
      setRanQuery(q); setQuery(q);
      if (remember) setTraceId(tid);
    } catch (e) { setErr(`That run could not be reopened: ${(e as Error).message}`); }
    setReopening(false);
  };

  const loadHistory = () => { if (indexId) api.get(`/api/traces?index_id=${indexId}`).then(setHistory_).catch(() => setHistory_([])); };
  useEffect(loadHistory, [indexId]); // eslint-disable-line
  // Restore the last run once, on the first mount that has an index.
  const restored = useRef(false);
  useEffect(() => { if (restored.current || !traceId || !indexId || events.length) return;
    restored.current = true; openTrace(traceId, { remember: false }); }, [traceId, indexId]); // eslint-disable-line
  const [follow, setFollow] = usePersisted("raglab.workbench.follow", false);
  // History has to outlive the tab too: with follow-up on, losing it silently changes what the
  // next question means.
  const [history, setHistory] = usePersisted<{ q: string; a: string }[]>("raglab.workbench.history", []);
  const [wf, setWf] = usePersisted<"ms" | "tokens" | "cost">("raglab.workbench.waterfall", "ms");
  const groups = useMemo(() => makeGroups(detail?.docs ?? [], detail?.docs?.some((x: Any) => x.meta.doc_type) ? "doc_type" : "document"), [detail]);
  // Field errors, inactive-control warnings and the per-question request count, from the server so
  // the rules live in one place. Debounced, because the settings change on every keystroke.
  const [check, setCheck] = useState<Any>(null);
  useEffect(() => { const t = setTimeout(() => { api.post("/api/pipeline/check", { config: pipeline })
    .then(setCheck).catch(() => setCheck(null)); }, 250); return () => clearTimeout(t); }, [pipeline]);
  const blocked = !!check?.errors?.length;
  // Sample questions follow the corpus the selected index was built from; the full list mixes two
  // corpora, so half of them used to miss whichever index was open.
  const samples = useMemo(() => {
    const corpora = new Set<string>((detail?.docs ?? []).map((d: Any) => d.meta?.corpus).filter(Boolean));
    const fit = SAMPLES.filter((q) => !q.corpus || corpora.has(q.corpus)).map((q) => q.text);
    return fit.length ? fit : SAMPLES.map((q) => q.text);
  }, [detail]);

  useEffect(() => { api.get("/api/indexes").then((ix: Any[]) => { const ready = ix.filter((i) => i.status === "ready"); setIndexes(ready);
    if (!ready.some((i) => i.id === indexId) && ready[0]) setIndexId(ready[0].id); }); }, []); // eslint-disable-line
  useEffect(() => { if (!indexId) return; localStorage.setItem("raglab.index", indexId); setDetail(null);
    api.get(`/api/indexes/${indexId}`).then(setDetail).catch(() => undefined); }, [indexId]);

  const run = async () => {
    abort.current?.abort(); abort.current = new AbortController();
    setRanQuery(query); setEvents([]); setChunks({}); setAnswer(""); setErr(""); setTotal(null); setUsage(null); setRunning(true);
    try {
      let final = "";
      await streamQuery({ index_id: indexId, query, config: pipeline, history: follow ? history : [] }, (ev) => {
        if (ev.type === "stage") { setEvents((e) => [...e, ev]); setChunks((c) => ({ ...c, ...ev.chunks })); if (ev.data.answer) { setAnswer(ev.data.answer); final = ev.data.answer; } }
        else if (ev.type === "token") { final += ev.text; setAnswer((a) => a + ev.text); }
        else if (ev.type === "done") { setTotal(ev.total_ms); setUsage(ev.usage); if (ev.trace_id) { setTraceId(ev.trace_id); loadHistory(); }
          if (follow) setHistory((h) => [...h, { q: query, a: final.replace(/<think>[\s\S]*?<\/think>/g, "") }].slice(-6)); }
        else if (ev.type === "error") setErr(ev.message);
      }, abort.current.signal);
    } catch (e) { if ((e as Error).name !== "AbortError") setErr(String((e as Error).message ?? e)); }
    setRunning(false);
  };
  // The abort controller already existed; nothing exposed it, so a slow run could only be waited out.
  const stop = () => { abort.current?.abort(); setRunning(false); };
  /** Scroll to the passage a citation names and flash it, so [2] leads somewhere. */
  const showPassage = (n: number) => {
    const el = document.getElementById(`ctx-${n}`);
    if (!el) return;
    el.scrollIntoView({ behavior: "smooth", block: "center" });
    el.classList.add("flash");
    setTimeout(() => el.classList.remove("flash"), 1600);
  };

  if (!indexes.length) return <div className="panel"><h3>No index to query yet</h3>
    <p className="muted" style={{ marginTop: 6 }}>Check your models, load documents, then build an index.</p>
    <div className="row" style={{ marginTop: 10 }}><button className="btn" onClick={() => goto("Models")}>Set up models</button><button className="btn ghost" onClick={() => goto("Corpus")}>Add documents</button><button className="btn ghost" onClick={() => goto("Chunking")}>Build an index</button></div></div>;

  return (
    <BoundsCtx.Provider value={bounds ?? {}}>
    <div className="split">
      <Config p={pipeline} set={setPipeline} reset={() => setPipeline(defaults)} defaults={defaults} hasGraph={!!detail?.graph}
          hasCommunities={!!detail?.communities?.length} presets={presets} check={check} />
      <div>
        <div className="panel stack">
          <div className="row">
            <select style={{ width: 210 }} value={indexId} onChange={(e) => setIndexId(e.target.value)} aria-label="Index">{indexes.map((i) => <option key={i.id} value={i.id}>{i.name} ({i.stats.chunks} chunks)</option>)}</select>
            <input className="grow" type="text" value={query} onChange={(e) => setQuery(e.target.value)} onKeyDown={(e) => e.key === "Enter" && !running && run()} aria-label="Query" />
            {running ? <button className="btn ghost" onClick={stop}>Stop</button>
              : <button className="btn" onClick={run} disabled={!query.trim() || blocked} title={blocked ? "Fix the highlighted settings first" : undefined}>Run pipeline</button>}
          </div>
          <div className="row small"><label className="check small"><input type="checkbox" checked={follow} onChange={(e) => { setFollow(e.target.checked); if (!e.target.checked) setHistory([]); }} />Follow-up mode: keep the conversation and rewrite each question to stand alone</label>
            {follow && history.length > 0 && <><span className="tag">{history.length} turn{history.length > 1 ? "s" : ""} kept</span><button className="link" onClick={() => setHistory([])}>Clear</button></>}</div>
          <div className="row small" style={{ gap: 6 }}><span className="muted">Try</span>{samples.map((s) => <button key={s} className="link" onClick={() => setQuery(s)}>{s.length > 44 ? s.slice(0, 42) + "…" : s}</button>)}</div>
          {history_.length > 0 && <details>
            <summary className="small muted">Earlier runs on this index ({history_.length})</summary>
            <div className="stack" style={{ marginTop: 6 }}>{history_.slice(0, 15).map((h: Any) => <div className="row small" key={h.id}>
              <span className="grow" style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}
                    title={h.query}>{h.id === traceId && <span className="tag ok">shown</span>} {h.query}</span>
              <span className="muted mono">{new Date(h.created * 1000).toLocaleTimeString()}</span>
              <button className="link" disabled={reopening} onClick={() => openTrace(h.id)}>Reopen</button>
              <button className="link" onClick={() => setQuery(h.query)}>Ask again</button></div>)}</div>
            <p className="small muted" style={{ marginTop: 6 }}>Reopen shows the run exactly as it was traced. Ask again
              puts the question back in the box so you can run it under the settings you have now.</p></details>}
        </div>
        {err && <div className="err" style={{ marginTop: 10 }}>{err}</div>}
        {(answer || running) && <div className="panel" style={{ marginTop: 10 }}><div className="row"><h3 className="grow">Answer</h3>{total !== null && <span className="ms mono small muted">{(total / 1000).toFixed(1)} s end to end</span>}</div>
          <div style={{ marginTop: 8 }}>{answer ? <Answer text={answer} onCite={showPassage} /> : <span className="muted">Working through the pipeline…</span>}</div></div>}
        {total !== null && events.length > 1 && <div className="panel" style={{ marginTop: 10 }}><div className="row" style={{ marginBottom: 8 }}><h4 className="grow">Where the {wf === "ms" ? "time" : wf === "tokens" ? "tokens" : "money"} went</h4>
          <span className="mono small muted">{total} ms, {usage?.tokens_in ?? 0}↑ {usage?.tokens_out ?? 0}↓ tokens{usage?.cost ? `, $${usage.cost.toFixed(4)}` : ""}</span>
          <div className="seg">{(["ms", "tokens", "cost"] as const).map((m) => <button key={m} aria-pressed={wf === m} onClick={() => setWf(m)}>{m === "ms" ? "time" : m}</button>)}</div></div><Waterfall events={events} metric={wf} /></div>}
        {events.map((ev, n) => <Event key={n} n={n + 1} ctx={{ chunks, detail, groups }}
          ev={ev.stage === "fuse" ? { ...ev, lists: events.find((e) => e.stage === "retrieve")?.data.lists, rrfK: pipeline.fusion.rrf_k } : ev.stage === "retrieve" ? { ...ev, query: ranQuery } : ev} />)}
      </div>
    </div>
    </BoundsCtx.Provider>
  );
}
