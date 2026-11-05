import { useEffect, useMemo, useState } from "react";
import { api, Any , usePersisted } from "../api";
import { Legend, makeGroups, ScoreBar, VectorMap, MapQuery } from "../viz";

export default function VectorSpace() {
  const [indexes, setIndexes] = useState<Any[]>([]); const [indexId, setIndexId] = useState(localStorage.getItem("raglab.index") ?? "");
  const [detail, setDetail] = useState<Any>(null); const [umap, setUmap] = useState<Any[] | null>(null);
  const [method, setMethod] = usePersisted<"pca" | "umap">("raglab.vectorspace.method", "pca");
  const [field, setField] = usePersisted("raglab.vectorspace.field", "document");
  const [k, setK] = usePersisted("raglab.vectorspace.k", 6);
  const [labels, setLabels] = usePersisted("raglab.vectorspace.labels", false);
  const [pinned, setPinned] = useState<number | null>(null); const [near, setNear] = useState<Any[]>([]);
  const [text, setText] = useState(""); const [probe, setProbe] = useState<MapQuery | null>(null); const [err, setErr] = useState(""); const [busy, setBusy] = useState("");
  // Chunks fetched individually because they sit beyond the detail page, keyed by index.
  const [extra, setExtra] = useState<Record<number, Any>>({});

  /** Fetch any of these chunks the detail response did not carry. */
  const ensureChunks = async (want: number[]) => {
    if (!detail) return;
    const missing = want.filter((i) => !detail.chunks[i] && !extra[i]);
    if (!missing.length) return;
    try {
      const r = await api.get(`/api/indexes/${indexId}/chunks?ids=${missing.join(",")}`);
      setExtra((cur) => r.chunks.reduce((acc: Any, c: Any) => ({ ...acc, [c.i]: c }), { ...cur }));
    } catch (e) { setErr(String((e as Error).message)); }
  };

  useEffect(() => { api.get("/api/indexes").then((ix: Any[]) => { const r = ix.filter((i) => i.status === "ready"); setIndexes(r); if (!r.some((i) => i.id === indexId) && r[0]) setIndexId(r[0].id); }); }, []); // eslint-disable-line
  useEffect(() => { if (!indexId) return; setDetail(null); setUmap(null); setMethod("pca"); setPinned(null); setProbe(null); setNear([]); setExtra({});
    api.get(`/api/indexes/${indexId}`).then((d) => { setDetail(d); setField(d.docs.some((x: Any) => x.meta.doc_type) ? "doc_type" : "document"); }).catch((e) => setErr(String(e))); }, [indexId]);

  const groups = useMemo(() => makeGroups(detail?.docs ?? [], field), [detail, field]);
  const fields = useMemo(() => ["document", ...Array.from(new Set<string>((detail?.docs ?? []).flatMap((d: Any) => Object.keys(d.meta))))], [detail]);
  // The remembered colour-by field may not exist on a different index; fall back rather than
  // leaving the select showing a value that is not in its own list.
  useEffect(() => { if (detail && !fields.includes(field)) setField("document"); }, [detail, fields, field, setField]);
  const points = method === "umap" && umap ? umap : detail?.projection ?? [];

  const switchMethod = async (m: "pca" | "umap") => {
    setErr(""); setProbe(null);
    if (m === "umap" && !umap) { setBusy("umap"); try { setUmap((await api.get(`/api/indexes/${indexId}/umap`)).projection); } catch (e) { setErr(String((e as Error).message)); setBusy(""); return; } setBusy(""); }
    setMethod(m);
  };
  const pin = async (i: number | null) => { setPinned(i); setProbe(null); if (i === null) { setNear([]); return; }
    const hits = (await api.post(`/api/indexes/${indexId}/probe`, { i, k })).neighbours;
    await ensureChunks([i, ...hits.map((h: Any) => h.i)]); setNear(hits); };
  const drop = async () => { setErr(""); setBusy("probe"); setPinned(null);
    try { const r = await api.post(`/api/indexes/${indexId}/probe`, { text, k, method });
      await ensureChunks(r.neighbours.map((n: Any) => n.i)); setNear(r.neighbours);
      setProbe({ label: text, kind: "probe", xy: r.xy, links: r.neighbours.map((n: Any) => n.i) }); } catch (e) { setErr(String((e as Error).message)); } setBusy(""); };
  useEffect(() => { if (pinned !== null) pin(pinned); else if (probe) drop(); }, [k]); // eslint-disable-line

  if (!indexes.length) return <div className="panel"><h3>No index to map yet</h3><p className="muted" style={{ marginTop: 6 }}>Build an index on the Chunking tab, then come back to see its vector space.</p></div>;
  // /api/indexes/{id} carries a page of chunks, but an exact-search probe ranges over the whole
  // matrix, so a neighbour past that page has no local record. Those are now fetched by id rather
  // than shown as "not loaded".
  const chunk = (i: number): Any => detail.chunks[i] ?? extra[i] ?? null;
  const s = detail?.stats; const nearSet = new Set<number>(near.map((n) => n.i));
  const truncated = (s?.chunks ?? 0) > (detail?.chunks?.length ?? 0);

  return (
    <>
      <div className="pagehead"><h2>Vector space</h2>
        <p>Every chunk of the index, embedded by your real embedding model and projected to two dimensions. Drop a phrase in to see where a query lands, or pin a chunk to list its true nearest neighbours by cosine.</p></div>
      {err && <div className="err" style={{ marginBottom: 12 }}>{err}</div>}
      <div className="split" style={{ gridTemplateColumns: "minmax(0, 1fr) 340px" }}>
        <div className="panel">
          <div className="row" style={{ marginBottom: 10 }}>
            <div className="seg">{(["pca", "umap"] as const).map((m) => <button key={m} aria-pressed={method === m} onClick={() => switchMethod(m)}>{busy === "umap" && m === "umap" ? "fitting…" : m.toUpperCase()}</button>)}</div>
            <label className="row small muted" style={{ gap: 6 }}>Colour by<select style={{ width: 130 }} value={field} onChange={(e) => setField(e.target.value)}>{fields.map((f) => <option key={f}>{f}</option>)}</select></label>
            <span className="grow" />
            {s && <><span className="tag">{s.chunks} chunks</span><span className="tag">{s.dim}d → 2d</span>{method === "pca" && detail && <span className="tag">{Math.round(detail.pca_variance * 100)}% variance kept</span>}</>}
          </div>
          {detail && <VectorMap points={points} groups={groups} height={560} pinned={pinned} onPin={pin} queries={probe ? [probe] : []}
            hits={near.length ? new Set<number>([...nearSet, ...(pinned !== null ? [pinned] : [])]) : undefined} rings={nearSet} labelled={labels ? nearSet : undefined}
            tip={(i) => ({ title: chunk(i)?.title ?? `#${i}`, sub: chunk(i)?.section?.split(" > ").pop(), text: chunk(i)?.text ?? "(not loaded)" })} />}
          <div style={{ marginTop: 10 }}><Legend groups={groups} /></div>
          <p className="small muted" style={{ marginTop: 8 }}>{method === "pca" ? "PCA is linear: directions are faithful, but with only part of the variance kept, clusters can overlap." : "UMAP preserves neighbourhoods, not distances: trust who sits next to whom, not how far apart clusters are."}</p>
        </div>
        <div className="stack">
          <div className="panel stack">
            <label className="f"><span>Index</span><select value={indexId} onChange={(e) => { setIndexId(e.target.value); localStorage.setItem("raglab.index", e.target.value); }}>{indexes.map((i) => <option key={i.id} value={i.id}>{i.name}</option>)}</select></label>
            {s && <div className="small muted">Embedded with <code>{s.embedding_model}</code>, {s.dim} dimensions.</div>}
          </div>
          <div className="panel stack">
            <h4>Probe the space</h4>
            <div className="row"><input className="grow" type="text" value={text} placeholder="e.g. cache is down at peak time" onChange={(e) => setText(e.target.value)} onKeyDown={(e) => e.key === "Enter" && text.trim() && drop()} />
              <button className="btn" disabled={!text.trim() || busy === "probe"} onClick={drop}>Drop in</button></div>
            <p className="small muted">Embedded with the query prefix, so it lands where a question would.</p>
            <label className="check small"><input type="checkbox" checked={labels} onChange={(e) => setLabels(e.target.checked)} />Label neighbours on the map</label>
            <label className="f"><span>Neighbours: {k}</span><input type="range" min={1} max={20} value={k} onChange={(e) => setK(Number(e.target.value))} style={{ width: "100%", accentColor: "var(--bm25)" }} /></label>
          </div>
          <div className="panel">
            <h4>{pinned !== null ? `Neighbours of #${pinned}` : probe ? "Nearest to the probe" : "Neighbours"}</h4>
            {pinned !== null && (chunk(pinned)
              ? <div className="chunkbox" style={{ marginBottom: 8 }}><b>{chunk(pinned).title}</b> <span className="muted">{chunk(pinned).section}</span><div className="small">{chunk(pinned).text.slice(0, 240)}…</div></div>
              : <div className="chunkbox" style={{ marginBottom: 8 }}><b>#{pinned}</b> <span className="muted">not loaded</span></div>)}
            {!near.length && <p className="small muted" style={{ marginTop: 6 }}>Hover to inspect a chunk. Click one to pin it and list its neighbours.</p>}
            {near.map((n, r) => { const c = chunk(n.i); return <div className="hit" key={n.i} onClick={() => pin(n.i)}><span className="r">{r + 1}</span>
              <div className="t"><b>{c ? (c.section?.split(" > ").pop() || c.title) : `#${n.i}`}</b><span className="snip">{c ? c.title : "fetching…"}</span></div><span className="s">{n.score.toFixed(3)}</span>
              <ScoreBar value={n.score} max={near[0].score} color={groups.color(groups.of(c?.doc_id ?? ""))} /></div>; })}
            {truncated && <p className="small muted" style={{ marginTop: 8 }}>This index has {s.chunks} chunks; the map draws
              the first {detail.chunks.length}. Neighbours past that point are searched over in full and fetched on demand,
              so the list above is complete even where the map is not.</p>}
          </div>
        </div>
      </div>
    </>
  );
}
