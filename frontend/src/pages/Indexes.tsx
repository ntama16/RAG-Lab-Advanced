import { useEffect, useMemo, useRef, useState } from "react";
import { api, Any, deepMerge, usePersisted } from "../api";
import { Legend, makeGroups, Section, Tiles } from "../viz";

/* Generic words that carry no information in an embedding model id.
   Mirrors _GENERIC_MODEL_WORDS in raglab/indexer.py. */
const GENERIC_MODEL_WORDS = new Set(["text", "embedding", "embeddings", "embed", "all", "general", "base", "model"]);

/**
 * Shorten an embedding model id to one readable token for use in an index name.
 * Mirror of `model_tag` in raglab/indexer.py — the backend is authoritative, this only
 * previews the name in the Name field, so keep the two in step.
 *
 * @param model Embedding model id, possibly `provider/name:tag`.
 * @returns A lowercase token, e.g. `nomic-embed-text` -> `nomic`, `text-embedding-3-large` -> `te3-large`.
 */
function modelTag(model: string): string {
  const ident = (model || "").split("/").slice(-1)[0].split(":")[0].toLowerCase();
  const parts = ident.split(/[^a-z0-9]+/).filter(Boolean);
  if (!parts.length) return "unknown";
  let lead = 0;
  while (lead < parts.length && GENERIC_MODEL_WORDS.has(parts[lead])) lead++;
  const rest = parts.slice(lead);
  if (!rest.length) return parts[0];
  if (/^\d+$/.test(rest[0])) return parts.slice(0, lead).map((w) => w[0]).join("") + rest[0] + (rest[1] ? `-${rest[1]}` : "");
  return rest[0];
}

/**
 * Build a short, self-describing index name from the parameters that shape an index:
 * `<model>-<strategy>-<size>x<overlap>[-flags][-N]`, e.g. `nomic-fixed-426x40-pc`.
 * Mirror of `suggest_name` in raglab/indexer.py — keep the two in step.
 *
 * @param cfg Index config as held by this form.
 * @param embeddingModel Embedding model id the build will use.
 * @param existing Names already in use, so a repeat gets a `-2`, `-3`, ... suffix.
 * @returns A unique, non-empty index name.
 */
function suggestName(cfg: Any, embeddingModel: string, existing: string[]): string {
  const c = cfg?.chunking ?? {};
  const parts = [modelTag(embeddingModel), String(c.strategy ?? "recursive")];
  parts.push(c.strategy === "semantic" ? `p${c.semantic?.percentile ?? 90}` : `${c.size ?? 0}x${c.overlap ?? 0}`);
  if (c.parent_child?.enabled) parts.push("pc");
  if (c.contextual?.enabled) parts.push("ctx");
  if (cfg?.graph?.enabled) { parts.push("graph"); if (cfg.graph.communities) parts.push("comm"); }
  if (cfg?.doc_metadata?.enabled) parts.push("meta");
  if (!cfg?.dedupe?.enabled) parts.push("nodedupe");
  if (cfg?.doc_ids?.length) parts.push(`${cfg.doc_ids.length}d`);
  const base = parts.join("-");
  const taken = new Set(existing);
  if (!taken.has(base)) return base;
  let n = 2;
  while (taken.has(`${base}-${n}`)) n++;
  return `${base}-${n}`;
}

/** What each flag in a generated name means, for the hint under the Name field. */
const NAME_FLAGS: [string, string][] = [
  ["pc", "parent-child"], ["ctx", "contextual"], ["graph", "knowledge graph"], ["comm", "communities"],
  ["meta", "doc metadata"], ["nodedupe", "dedupe off"], ["<n>d", "document subset"],
];

/** Format an epoch-seconds timestamp for display; `full` spells out the date and time. */
function when(epoch: number | undefined, full: boolean): string {
  if (!epoch) return "unknown";
  const d = new Date(epoch * 1000);
  return full ? d.toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" })
              : d.toLocaleString(undefined, { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" });
}

/**
 * The parameters an index was actually built with, read back from its stored config and stats.
 * @param detail An index detail record from `GET /api/indexes/:id`.
 * @returns Label/value pairs in display order.
 */
function buildParams(detail: Any): [string, string][] {
  const cfg = detail.config ?? {}, c = cfg.chunking ?? {}, s = detail.stats ?? {};
  const onOff = (on: unknown, extra = "") => (on ? (extra ? `on, ${extra}` : "on") : "off");
  // An index built by an older version can lack a config key added since, so never print "undefined".
  const val = (v: unknown) => (v === undefined || v === null || v === "" ? "unknown" : String(v));
  const chunkRows: [string, string][] = c.strategy === "semantic"
    ? [["Breakpoint", c.semantic?.percentile == null ? "unknown" : `p${c.semantic.percentile}, buffer ${val(c.semantic?.buffer)}`],
       ["Chunk bounds", `${val(c.semantic?.min_tokens)}–${val(c.semantic?.max_tokens)} tok`]]
    : [["Size / overlap", `${val(c.size)} / ${val(c.overlap)} tok`]];
  return [
    ["Built", when(detail.created, true)],
    ["Build time", s.build_s != null ? `${s.build_s}s` : "unknown"],
    ["Embedding model", `${val(s.embedding_model)}${s.dim ? `, ${s.dim} dim` : ""}`],
    ["Documents", val(s.documents)],
    ["Chunking", val(c.strategy)],
    ...chunkRows,
    ["Parent-child", onOff(c.parent_child?.enabled, c.parent_child?.enabled ? `${val(c.parent_child.parent_size)} tok parents` : "")],
    ["Contextual", onOff(c.contextual?.enabled)],
    ["Near-duplicates", onOff(cfg.dedupe?.enabled, cfg.dedupe?.enabled ? `Jaccard > ${val(cfg.dedupe.threshold)}` : "")],
    ["Knowledge graph", onOff(cfg.graph?.enabled, cfg.graph?.enabled && cfg.graph.communities ? "+ communities" : "")],
    ["LLM doc metadata", onOff(cfg.doc_metadata?.enabled)],
    ["HNSW M", val(cfg.hnsw?.M)],
    ["ef_construction", val(cfg.hnsw?.ef_construction)],
  ];
}

function Spark({ dbg }: { dbg: Any }) {
  const d: number[] = dbg.distances; if (!d.length) return null;
  const max = Math.max(...d, dbg.threshold) * 1.1 || 1; const w = 600; const h = 54;
  const x = (i: number) => (i / Math.max(d.length - 1, 1)) * w; const y = (v: number) => h - (v / max) * h;
  return (
    <svg className="spark" viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none" role="img" aria-label="Cosine distance between neighbouring sentences">
      <line x1="0" x2={w} y1={y(dbg.threshold)} y2={y(dbg.threshold)} stroke="var(--bad)" strokeDasharray="4 3" strokeWidth="1" />
      <polyline fill="none" stroke="var(--dense)" strokeWidth="1.5" points={d.map((v, i) => `${x(i)},${y(v)}`).join(" ")} />
      {d.map((v, i) => v > dbg.threshold && <circle key={i} cx={x(i)} cy={y(v)} r="3" fill="var(--bad)" />)}
    </svg>
  );
}

/** Source text on the left, the chunks it became on the right. Hover either side to find the other. */
function InsideChunk({ preview }: { preview: Any }) {
  const [hot, setHot] = useState<number | null>(null); const { text, chunks } = preview;
  const ov = (k: number) => (k > 0 && chunks[k - 1].end > chunks[k].start ? chunks[k - 1].end - chunks[k].start : 0);
  const cuts = useMemo(() => Array.from(new Set<number>([0, text.length, ...chunks.flatMap((c: Any) => [c.start, c.end])])).sort((a, b) => a - b), [text, chunks]);
  const withOv = chunks.filter((_: Any, k: number) => ov(k) > 0).length; const avg = Math.round(chunks.reduce((a: number, c: Any) => a + c.end - c.start, 0) / Math.max(chunks.length, 1));
  return (
    <div className="inside">
      <div><div className="small muted" style={{ marginBottom: 6 }}>Source text. Blue marks the hovered chunk, amber marks text it shares with the previous chunk.</div>
        <div className="srctext">{cuts.slice(0, -1).map((a, n) => { const b = cuts[n + 1]; const c = hot !== null ? chunks[hot] : null; const inside = c && a >= c.start && b <= c.end;
          const shared = inside && hot! > 0 && b <= chunks[hot! - 1].end; return <span key={a} className={shared ? "ov" : inside ? "in" : undefined}>{text.slice(a, b)}</span>; })}</div></div>
      <div><div className="small muted" style={{ marginBottom: 6 }}>{chunks.length} chunks, avg {avg} chars, {withOv} with overlap{preview.parents.length ? `, ${preview.parents.length} parents` : ""}</div>
        <div style={{ maxHeight: 520, overflow: "auto" }}>{chunks.map((c: Any, k: number) => <div key={k} className={`ccard ${hot === k ? "hot" : ""}`} onPointerEnter={() => setHot(k)} onPointerLeave={() => setHot(null)}>
          <header><span>#{k}{c.section ? `  ${c.section}` : ""}{c.parent !== null ? `  parent ${c.parent}` : ""}</span><span>{c.end - c.start} chars, ~{Math.round((c.end - c.start) / 4)} tok</span></header>
          {ov(k) > 0 && <span className="ov">{text.slice(c.start, c.start + ov(k))}</span>}{text.slice(c.start + ov(k), c.end)}</div>)}</div></div>
    </div>
  );
}

function ChunkMap({ detail }: { detail: Any }) {
  const [field, setField] = useState(detail.docs.some((d: Any) => d.meta.doc_type) ? "doc_type" : "document"); const [hot, setHot] = useState<Any>(null);
  const groups = useMemo(() => makeGroups(detail.docs, field), [detail, field]);
  const fields = ["document", ...Array.from(new Set<string>(detail.docs.flatMap((d: Any) => Object.keys(d.meta))))];
  const toks: number[] = detail.chunks.map((c: Any) => c.tokens); const max = Math.max(...toks, 1); const bins = 16;
  const hist = Array.from({ length: bins }, (_, b) => toks.filter((t) => Math.min(bins - 1, Math.floor((t / (max + 1)) * bins)) === b).length); const hmax = Math.max(...hist, 1);
  return (
    <div className="stack">
      <div className="row"><label className="row small muted" style={{ gap: 6 }}>Colour by<select style={{ width: 130 }} value={field} onChange={(e) => setField(e.target.value)}>{fields.map((f) => <option key={f}>{f}</option>)}</select></label><span className="grow" /><Legend groups={groups} /></div>
      <div className="cmap">{detail.docs.map((d: Any) => { const cs = detail.chunks.filter((c: Any) => c.doc_id === d.id); const col = groups.color(groups.of(d.id));
        return <div key={d.id}><div className="t"><svg width="9" height="9"><rect width="9" height="9" rx="2" fill={col} /></svg>{d.title}</div>
          <div className="segs">{cs.map((c: Any) => <i key={c.i} className={hot?.i === c.i ? "hot" : ""} style={{ flex: c.tokens, background: col }} onPointerEnter={() => setHot(c)} />)}</div></div>; })}</div>
      <div className="note" style={{ minHeight: 64 }}>{hot ? <><b>#{hot.i} {hot.title}</b> <span className="muted">{hot.section}</span> <span className="tag">{hot.tokens} tok</span>
        {hot.context && <div className="small" style={{ color: "var(--graph)" }}>{hot.context}</div>}<div className="small">{hot.text.slice(0, 300)}{hot.text.length > 300 ? "…" : ""}</div></>
        : <span className="muted">Each bar is a document, each block a chunk sized by its token count. Hover a block to read it.</span>}</div>
      <div><div className="small muted" style={{ marginBottom: 4 }}>Chunk size distribution (tokens), 0 to {max}</div><div className="hist">{hist.map((h, b) => <i key={b} title={`${Math.round((b * max) / bins)} to ${Math.round(((b + 1) * max) / bins)} tokens: ${h} chunks`} style={{ height: `${(100 * h) / hmax}%` }} />)}</div></div>
    </div>
  );
}

export default function Indexes({ defaults }: { defaults: Any }) {
  const [indexes, setIndexes] = useState<Any[]>([]); const [docs, setDocs] = useState<Any[]>([]);
  // Collections come from the server so this page and the Corpus page cannot disagree about what
  // exists. A dataset with nothing in it is listed too — it is selectable, reports 0, and says so.
  const [collections, setCollections] = useState<Any[]>([]);
  const [docsLoaded, setDocsLoaded] = useState(false);
  const [typedName, setTypedName] = usePersisted<string | null>("raglab.chunking.name", null);
  // Deep-merged onto the server defaults so a config saved before a new option existed still
  // loads, picking up the default for anything it does not mention.
  const [cfg, setCfg] = usePersisted<Any>("raglab.chunking.config", defaults,
                                          (stored, init) => (stored ? deepMerge(init, stored) : init));
  const [embModel, setEmbModel] = useState("");
  const [preview, setPreview] = useState<Any>(null);
  const [detail, setDetail] = useState<Any>(null); const [err, setErr] = useState(""); const [hnsw, setHnsw] = useState({ M: 16, ef_construction: 200 });
  const [corpus, setCorpus] = usePersisted("raglab.chunking.corpus", "all");
  const [previewDoc, setPreviewDoc] = usePersisted("raglab.chunking.previewDoc", "");
  const detailRef = useRef<HTMLDivElement>(null); const openRowRef = useRef<HTMLDivElement>(null);
  const railRef = useRef<HTMLDivElement>(null);
  // Bumped on every Inspect so re-opening the same index scrolls to it again.
  const [opened, setOpened] = useState(0);

  const refresh = () => api.get("/api/indexes").then(setIndexes);
  useEffect(() => {
    refresh();
    api.get("/api/documents").then((d: Any[]) => {
      setDocs(d);
      // Keep the remembered document only while it still exists.
      setPreviewDoc((cur) => (d.some((x) => x.id === cur) ? cur : d[0]?.id ?? ""));
      setDocsLoaded(true);
    }).catch((e) => setErr(`Could not read the corpus: ${e}`));
    // A remembered collection survives as long as the collection does, even with nothing in it —
    // an empty dataset you created on purpose should still be the selection when you come back.
    api.get("/api/collections").then((cs: Any[]) => {
      setCollections(cs);
      setCorpus((cur) => (cur === "all" || cs.some((c) => c.key === cur) ? cur : "all"));
    }).catch((e) => setErr(String(e)));
    // Only used to preview the generated name; if it fails the placeholder degrades to
    // "unknown-…" and the server still names the index correctly, so this must not block the page.
    api.get("/api/models").then((m) => setEmbModel(m?.models?.embedding?.model ?? ""))
      .catch((e) => { console.warn("Could not read the embedding model for the index-name preview:", e); setEmbModel(""); });
  }, []);
  useEffect(() => {
    if (!indexes.some((i) => i.status === "building")) return;
    const t = setInterval(refresh, 900); return () => clearInterval(t);
  }, [indexes]);

  // What the index will be called if the Name field is left blank. The backend generates the
  // real name the same way (indexer.suggest_name); this is the preview shown in the field.
  const suggested = useMemo(() => suggestName(cfg, embModel, indexes.map((i) => i.name)),
                            [cfg, embModel, indexes]);
  // Real, editable text rather than a placeholder: the generated name is there to be typed
  // over or added to. It tracks the form until the first keystroke, then leaves the user alone.
  const name = typedName ?? suggested;

  // doc_ids follows the corpus choice rather than being stored alongside it, so a remembered
  // filter cannot go stale and silently build an index from documents that no longer exist.
  useEffect(() => {
    // Until the corpus has actually arrived, leave doc_ids alone. Recomputing it from an empty
    // list would write `[]` for any collection, which reads on screen as "this collection is
    // empty" when the truth is that the request failed or has not landed yet.
    if (!docsLoaded) return;
    const ids = corpus === "all" ? null
      // Case-insensitive, matching `collection_of` on the server: the count in the dropdown is
      // grouped that way, so selecting a collection has to gather the same documents.
      : docs.filter((d) => ((String(d.metadata.corpus ?? "").trim() || "uploaded").toLowerCase()
                            === corpus.toLowerCase())).map((d) => d.id);
    setCfg((c: Any) => (JSON.stringify(c.doc_ids) === JSON.stringify(ids) ? c : { ...c, doc_ids: ids }));
  }, [docs, docsLoaded, corpus, setCfg]);

  // Opening a detail is the user's own action, so move the page to it rather than leaving them
  // to find it. Keyed on `opened` alone: `detail` also changes identity when the HNSW rebuild
  // writes new stats, and that must not drag the page away from the control just used.
  useEffect(() => {
    if (!opened) return;
    // Reveal the open row by scrolling the rail itself. scrollIntoView would scroll every
    // ancestor including the page, cancelling the smooth page scroll below.
    const row = openRowRef.current, rail = railRef.current;
    if (row && rail) {
      const r = row.getBoundingClientRect(), box = rail.getBoundingClientRect();
      if (r.top < box.top) rail.scrollTop -= box.top - r.top;
      else if (r.bottom > box.bottom) rail.scrollTop += r.bottom - box.bottom;
    }
    // Instant, not smooth: smooth scrolling is a no-op wherever the browser or OS has it
    // turned off, which would make this silently do nothing. A jump is also what someone
    // with reduced-motion set wants, so there is nothing to branch on.
    detailRef.current?.scrollIntoView({ block: "start" });
  }, [opened]);

  // An empty selection is not "no filter": `doc_ids: []` asks for nothing, and both this button
  // and `indexer.validate_index` refuse it rather than quietly indexing the whole corpus.
  const selectionEmpty = Array.isArray(cfg.doc_ids) && cfg.doc_ids.length === 0;

  const ch = cfg.chunking;
  const setCh = (patch: Any) => setCfg({ ...cfg, chunking: { ...ch, ...patch } });
  const guard = async (fn: () => Promise<unknown>) => { setErr(""); try { await fn(); } catch (e) { setErr(String(e)); } };
  const num = (v: string) => Number(v) || 0;

  return (
    <>
      <div className="pagehead"><h2>Chunking and indexes</h2>
        <p>An index is the corpus, chunked one way and embedded with one model. Build several and compare them on the same questions.</p></div>
      {err && <div className="err" style={{ marginBottom: 12 }}>{err}</div>}
      <div className="split">
        <div className="panel stack">
          <h3>Build an index</h3>
          <label className="f"><span>Name</span><input type="text" value={name} placeholder={suggested} onChange={(e) => setTypedName(e.target.value)} /></label>
          <p className="small muted" style={{ marginTop: -4 }}>Generated from the settings &mdash; model, chunking,
            size<span aria-hidden>×</span>overlap, then a flag per option &mdash; and follows them until you edit it.
            Clear it to go back to the generated name.</p>
          <p className="small muted" style={{ marginTop: -6 }}>{NAME_FLAGS.map(([flag, meaning], i) =>
            <span key={flag}>{i > 0 ? " · " : ""}<code>{flag}</code> {meaning}</span>)}</p>
          <div className="fields">
            <label className="f"><span>Chunking</span><select value={ch.strategy} onChange={(e) => setCh({ strategy: e.target.value })}>
              {["recursive", "markdown", "sentence", "fixed", "semantic"].map((s) => <option key={s}>{s}</option>)}</select></label>
            {ch.strategy !== "semantic" ? <>
              <label className="f"><span>Size (tokens)</span><input type="number" value={ch.size} onChange={(e) => setCh({ size: num(e.target.value) })} /></label>
              <label className="f"><span>Overlap</span><input type="number" value={ch.overlap} onChange={(e) => setCh({ overlap: num(e.target.value) })} /></label></> : <>
              <label className="f"><span>Breakpoint percentile</span><input type="number" value={ch.semantic.percentile} onChange={(e) => setCh({ semantic: { ...ch.semantic, percentile: num(e.target.value) } })} /></label>
              <label className="f"><span>Sentence buffer</span><input type="number" value={ch.semantic.buffer} onChange={(e) => setCh({ semantic: { ...ch.semantic, buffer: num(e.target.value) } })} /></label>
              <label className="f"><span>Min tokens</span><input type="number" value={ch.semantic.min_tokens} onChange={(e) => setCh({ semantic: { ...ch.semantic, min_tokens: num(e.target.value) } })} /></label>
              <label className="f"><span>Max tokens</span><input type="number" value={ch.semantic.max_tokens} onChange={(e) => setCh({ semantic: { ...ch.semantic, max_tokens: num(e.target.value) } })} /></label></>}
          </div>
          {ch.strategy === "semantic" && <p className="small muted">Embeds a window around every sentence and cuts where neighbouring windows drift apart. Costs one embedding call per sentence.</p>}
          <label className="check"><input type="checkbox" checked={ch.parent_child.enabled} onChange={(e) => setCh({ parent_child: { ...ch.parent_child, enabled: e.target.checked } })} />Parent-child (retrieve small, read big)</label>
          {ch.parent_child.enabled && <label className="f"><span>Parent size (tokens)</span><input type="number" value={ch.parent_child.parent_size} onChange={(e) => setCh({ parent_child: { ...ch.parent_child, parent_size: num(e.target.value) } })} /></label>}
          <label className="check"><input type="checkbox" checked={ch.contextual.enabled} onChange={(e) => setCh({ contextual: { enabled: e.target.checked } })} />Contextual retrieval (LLM situates each chunk)</label>
          <label className="check"><input type="checkbox" checked={cfg.dedupe.enabled} onChange={(e) => setCfg({ ...cfg, dedupe: { ...cfg.dedupe, enabled: e.target.checked } })} />Near-duplicate removal (6-word shingles, Jaccard &gt; {cfg.dedupe.threshold})</label>
          <label className="check"><input type="checkbox" checked={cfg.graph.enabled} onChange={(e) => setCfg({ ...cfg, graph: { ...cfg.graph, enabled: e.target.checked } })} />Knowledge graph (LLM entity and relation extraction)</label>
          {cfg.graph.enabled && <label className="check" style={{ marginLeft: 23 }}><input type="checkbox" checked={!!cfg.graph.communities} onChange={(e) => setCfg({ ...cfg, graph: { ...cfg.graph, communities: e.target.checked } })} />Community reports (for whole-corpus questions)</label>}
          <label className="f"><span>Documents</span><select value={corpus} onChange={(e) => setCorpus(e.target.value)}>
            <option value="all">All {docs.length} documents</option>
            {collections.map((c: Any) => <option key={c.key} value={c.key}>{c.label} only ({c.docs})</option>)}</select></label>
          <label className="check"><input type="checkbox" checked={cfg.doc_metadata.enabled} onChange={(e) => setCfg({ ...cfg, doc_metadata: { enabled: e.target.checked } })} />LLM document metadata (doc type, topics, year)</label>
          {(ch.contextual.enabled || cfg.graph.enabled) && <p className="small muted">These make one router-model call per chunk. On a local 8B model expect a few seconds per chunk. Results are cached by prompt, so rebuilding with the same chunks is free.</p>}
          <div className="fields">
            <label className="f"><span>HNSW M</span><input type="number" value={cfg.hnsw.M} onChange={(e) => setCfg({ ...cfg, hnsw: { ...cfg.hnsw, M: num(e.target.value) } })} /></label>
            <label className="f"><span>ef_construction</span><input type="number" value={cfg.hnsw.ef_construction} onChange={(e) => setCfg({ ...cfg, hnsw: { ...cfg.hnsw, ef_construction: num(e.target.value) } })} /></label>
          </div>
          <div className="row">
            <button className="btn" disabled={!docs.length || selectionEmpty} onClick={() => guard(async () => { await api.post("/api/indexes", { name: name.trim(), config: cfg }); setTypedName(null); refresh(); })}>Build index</button>
            <select style={{ width: 130 }} value={previewDoc} onChange={(e) => setPreviewDoc(e.target.value)} aria-label="Document to preview">{docs.map((d) => <option key={d.id} value={d.id}>{d.title}</option>)}</select>
            <button className="btn ghost" disabled={!previewDoc} onClick={() => guard(async () => setPreview(await api.post("/api/chunk-preview", { doc_id: previewDoc, chunking: ch })))}>Preview chunks</button>
          </div>
          {!docs.length && <p className="small muted">Add documents on the Corpus tab first.</p>}
          {!!docs.length && selectionEmpty && <p className="small muted">
            <b>{collections.find((c: Any) => c.key === corpus)?.label ?? corpus}</b> has no documents in it yet, so
            there is nothing to index. Upload into it on the Corpus tab, or choose another collection.</p>}
          <p className="small muted">An index is a snapshot. Documents added after it is built are not in it
            until you build again — existing indexes are never rewritten underneath you.</p>
        </div>

        <div>
          {preview && <div style={{ marginBottom: 14 }}><Section title="Inside a chunk" aside={<button className="link" onClick={() => setPreview(null)}>Close preview</button>}>
            {preview.debug.semantic?.map((d: Any, i: number) => <div key={i} style={{ marginBottom: 12 }}><Spark dbg={d} />
              <p className="small muted">Cosine distance between neighbouring sentence windows. Cuts fall where the line crosses the dashed threshold ({d.threshold}).</p></div>)}
            <InsideChunk preview={preview} /></Section></div>}

          <div className="panel">
            <div className="row"><h3 className="grow">Built indexes</h3>
              {indexes.length > 0 && <span className="ixcount">{indexes.length}</span>}</div>
            {!indexes.length && <p className="muted" style={{ marginTop: 8 }}>No indexes yet. Build one with the form on the left.</p>}
            <div className="ixlist" ref={railRef}>
            {indexes.map((ix) => { const s = ix.stats; const c = ix.config.chunking; const open = detail?.id === ix.id; return (
              <div key={ix.id} ref={open ? openRowRef : undefined} className={`ixrow ${open ? "open" : ""}`} aria-current={open ? "true" : undefined}>
                <div className="row"><h4 className="grow">{ix.name} <span className="tag">{ix.id}</span></h4>
                  <span className={`tag ${ix.status === "ready" ? "ok" : ix.status === "failed" ? "bad" : ""}`}>{ix.status}</span>
                  {ix.status === "ready" && <button className="link" onClick={() => guard(async () => { setDetail(await api.get(`/api/indexes/${ix.id}`)); setHnsw(ix.config.hnsw); setOpened((n) => n + 1); })}>Inspect</button>}
                  <button className="link" onClick={() => guard(async () => { await api.del(`/api/indexes/${ix.id}`); if (detail?.id === ix.id) setDetail(null); refresh(); })}>Delete</button></div>
                <div className="row small" style={{ marginTop: 4, gap: 5 }}>
                  <span className="tag">{c.strategy}{c.strategy !== "semantic" ? ` ${c.size}/${c.overlap}` : ` p${c.semantic.percentile}`}</span>
                  {c.parent_child.enabled && <span className="tag">parent-child</span>}{c.contextual.enabled && <span className="tag">contextual</span>}
                  {ix.config.graph.enabled && <span className="tag graph">graph{ix.config.graph.communities ? " + communities" : ""}</span>}
                  {s.chunks && <span className="muted">{s.chunks} chunks, mean {s.tokens_mean} tok, dim {s.dim}, {s.embedding_model}, built {when(ix.created, false)} in {s.build_s}s{s.entities ? `, ${s.entities} entities / ${s.relations} relations` : ""}</span>}
                </div>
                {ix.status === "building" && ix.job && <><div className="row small" style={{ marginTop: 4 }}>
                    <span className="muted grow">{ix.job.stage} {ix.job.total ? `${ix.job.done}/${ix.job.total}` : ""}</span>
                    <button className="link" onClick={() => api.post(`/api/indexes/${ix.id}/stop`).then(() => refresh())}>Stop</button></div>
                  <div className="progress"><i style={{ width: `${ix.job.total ? (100 * ix.job.done) / ix.job.total : 5}%` }} /></div></>}
                {ix.status === "failed" && <div className="err" style={{ marginTop: 6 }}>{s.error}</div>}
              </div>); })}
            </div>
          </div>

          {detail && <div className="panel stack" ref={detailRef} style={{ scrollMarginTop: 78 }}>
            <div className="row"><h3 className="grow">{detail.name} <span className="tag">{detail.id}</span></h3><button className="link" onClick={() => setDetail(null)}>Close</button></div>
            <Tiles items={[[detail.stats.chunks, `chunks, ${detail.config.chunking.strategy}`], [detail.stats.documents, "documents"], [detail.stats.tokens_mean, "avg tokens / chunk"], [detail.stats.tokens_p95, "p95 tokens"],
              ...(detail.stats.parents ? [[detail.stats.parents, "parent passages"] as [number, string]] : []), ...(detail.stats.deduped ? [[detail.stats.deduped, "near-duplicates removed"] as [number, string]] : []), ...(detail.stats.entities ? [[detail.stats.entities, "graph entities"] as [number, string], [detail.stats.relations, "relations"] as [number, string]] : [])]} />
            <div><h4 style={{ marginBottom: 8 }}>Built with</h4>
              <div className="params">{buildParams(detail).map(([label, value]) => <div key={label}><span>{label}</span><b>{value}</b></div>)}</div></div>
            <div><h4 style={{ marginBottom: 8 }}>Chunk map</h4><ChunkMap detail={detail} /></div>
            {(detail.stats.deduped > 0 || detail.stats.llm_cache_hits > 0) && <div className="note small">
              {detail.stats.deduped > 0 && <div><b>{detail.stats.deduped} near-duplicate chunk{detail.stats.deduped > 1 ? "s" : ""} removed.</b> {detail.stats.deduped_examples.slice(0, 3).map((d: Any, n: number) => <span key={n}> “{d.text.slice(0, 70)}…” from <i>{d.title}</i> duplicated <i>{d.duplicate_of}</i>.</span>)}</div>}
              {detail.stats.llm_cache_hits > 0 && <div>{detail.stats.llm_cache_hits} index-time model calls served from cache, {detail.stats.llm_cache_misses} made fresh.</div>}</div>}
            {detail.communities?.length > 0 && <div><h4>Community reports</h4>{detail.communities.slice(0, 8).map((c: Any) => <details key={c.id} style={{ marginTop: 6 }}><summary><b>{c.title}</b> <span className="muted small">{c.entities.length} entities</span></summary>
              <p className="small" style={{ marginTop: 4, whiteSpace: "pre-wrap" }}>{c.summary}</p><div className="row" style={{ gap: 4 }}>{c.entities.slice(0, 14).map((e: string) => <span className="tag graph" key={e}>{e}</span>)}</div></details>)}</div>}
            <div><h4>Rebuild HNSW from stored vectors</h4>
              <div className="row" style={{ marginTop: 6 }}>
                <label className="f" style={{ width: 90 }}><span>M</span><input type="number" value={hnsw.M} onChange={(e) => setHnsw({ ...hnsw, M: num(e.target.value) })} /></label>
                <label className="f" style={{ width: 130 }}><span>ef_construction</span><input type="number" value={hnsw.ef_construction} onChange={(e) => setHnsw({ ...hnsw, ef_construction: num(e.target.value) })} /></label>
                <button className="btn ghost" style={{ alignSelf: "end" }} onClick={() => guard(async () => { const st = await api.post(`/api/indexes/${detail.id}/hnsw`, hnsw); setDetail({ ...detail, stats: st }); refresh(); })}>Rebuild</button>
                <span className="small muted" style={{ alignSelf: "end" }}>last graph build {detail.stats.hnsw_build_s}s. No re-embedding. Recall against exact search shows in every Workbench trace.</span></div></div>
            <div><h4>Filterable metadata</h4><div className="row" style={{ marginTop: 6, gap: 5 }}>
              {Object.entries(detail.schema).map(([k, v]) => <span className="tag" key={k}>{k}: {(v as Any[]).slice(0, 5).join(", ")}</span>)}</div></div>
            {detail.graph && <div><h4>Most connected entities</h4><div className="row" style={{ marginTop: 6, gap: 5 }}>
              {detail.graph.nodes.slice(0, 40).map((n: Any) => <span className="tag graph" key={n.id}>{n.label} ({n.chunks})</span>)}</div>
              <table style={{ marginTop: 8 }}><tbody>{detail.graph.edges.slice(0, 25).map((e: Any, i: number) => <tr key={i}><td>{e.s}</td><td className="muted">{e.r}</td><td>{e.t}</td></tr>)}</tbody></table></div>}
          </div>}
        </div>
      </div>
    </>
  );
}
