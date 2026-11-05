import { useEffect, useMemo, useRef, useState } from "react";
import { api, Any, deepMerge, runCollect, runFromTrace, usePersisted } from "../api";
import type { Run } from "../api";
import { makeGroups, RankFlow, STAGE_COLOR } from "../viz";

const rankIn = (list: number[], id: number) => (list.includes(id) ? list.indexOf(id) + 1 : null);
const flat = (o: Any, pre = ""): Record<string, Any> => Object.entries(o ?? {}).reduce((a, [k, v]) => (v && typeof v === "object" && !Array.isArray(v) ? { ...a, ...flat(v, `${pre}${k}.`) } : { ...a, [`${pre}${k}`]: v }), {});

export default function Compare({ pipeline }: { pipeline: Any }) {
  const [indexes, setIndexes] = useState<Any[]>([]); const [presets, setPresets] = useState<Any[]>([]); const [detail, setDetail] = useState<Any>(null);
  // Comparing two different indexes is the point of this page, so both sides are remembered
  // rather than being reset to one shared index every time the tab is re-entered.
  const [ia, setIa] = usePersisted("raglab.compare.indexA", "");
  const [ib, setIb] = usePersisted("raglab.compare.indexB", "");
  // null means "never chosen", which still gets the old default of the first saved preset.
  const [paStored, setPa] = usePersisted<string | null>("raglab.compare.presetA", null);
  const [pb, setPb] = usePersisted("raglab.compare.presetB", "__current");
  const pa = paStored ?? "__current";
  const [query, setQuery] = usePersisted("raglab.compare.query", "Can I send back a ZX-481 after three weeks?"); const [A, setA] = useState<Run | null>(null); const [B, setB] = useState<Run | null>(null); const [busy, setBusy] = useState(false); const [err, setErr] = useState("");
  // On by default: with the cache live, an identical B can return A's answer instantly and the
  // timing column then compares a run against a lookup. Routing is pinned for the same reason.
  const [control, setControl] = usePersisted("raglab.compare.control", true);
  // The configuration the results on screen came from, captured at run time. Persisted with the two
  // trace ids so the comparison survives leaving the tab, which unmounts this page: remembering the
  // controls and forgetting the results left the form describing a comparison that was gone.
  const [ran, setRan] = usePersisted<Any>("raglab.compare.ran", null);
  const [restoring, setRestoring] = useState(false);
  const restored = useRef(false);
  useEffect(() => {
    if (restored.current || !ran?.traceA || !ran?.traceB || A || B) return;
    restored.current = true; setRestoring(true);
    Promise.all([runFromTrace(ran.traceA), runFromTrace(ran.traceB)])
      .then(([a, b]) => { setA(a); setB(b); })
      // A trace can be gone if its index was deleted; drop the stale record rather than nagging.
      .catch(() => setRan(null))
      .finally(() => setRestoring(false));
  }, [ran]); // eslint-disable-line

  useEffect(() => {
    api.get("/api/indexes").then((ix: Any[]) => {
      const r = ix.filter((i) => i.status === "ready"); setIndexes(r);
      const ids = r.map((i) => i.id);
      const shared = localStorage.getItem("raglab.index") ?? "";
      const fallback = ids.includes(shared) ? shared : r[0]?.id ?? "";
      // Only fill in a side whose remembered index has since been deleted.
      setIa((cur) => (ids.includes(cur) ? cur : fallback));
      setIb((cur) => (ids.includes(cur) ? cur : fallback));
    });
    api.get("/api/presets").then((p: Any[]) => {
      setPresets(p);
      const known = (v: string) => v === "__current" || p.some((x) => x.name === v);
      setPa((cur) => (cur !== null && known(cur) ? cur : p[0]?.name ?? "__current"));
      setPb((cur) => (known(cur) ? cur : "__current"));
    });
  }, []); // eslint-disable-line
  useEffect(() => { if (ib) api.get(`/api/indexes/${ib}`).then(setDetail).catch(() => undefined); }, [ib]);
  const cfgOf = (name: string) => name === "__current" ? pipeline : deepMerge(pipeline, presets.find((p) => p.name === name)?.config ?? {});
  const groups = useMemo(() => makeGroups(detail?.docs ?? [], "document"), [detail]);

  const run = async () => { setBusy(true); setErr(""); setA(null); setB(null);
    // Freeze what is about to run. The controls stay editable while the results are on screen, and
    // reading the diff back off them described a comparison that was never performed.
    const frozen = { a: cfgOf(pa), b: cfgOf(pb), ia, ib, query, control, pa, pb };
    try { const a = await runCollect({ index_id: ia, query, config: frozen.a, control }); setA(a);
      const b = await runCollect({ index_id: ib, query, config: frozen.b, control }); setB(b);
      setRan({ ...frozen, traceA: a.traceId, traceB: b.traceId });
      if (a.error || b.error) setErr(a.error || b.error); }
    catch (e) { setErr(String((e as Error).message)); } setBusy(false); };

  /** Settings that differ between two configs, flattened to dotted paths. */
  const diffOf = (ca: Any, cb: Any) => { const fa = flat(ca), fb = flat(cb);
    return Object.keys({ ...fa, ...fb }).filter((k) => JSON.stringify(fa[k]) !== JSON.stringify(fb[k])).map((k) => [k, fa[k], fb[k]]); };
  // Before a run, the diff previews what the controls would compare; after one, it describes what
  // they actually did.
  const diffs = useMemo(() => ran ? diffOf(ran.a, ran.b) : diffOf(cfgOf(pa), cfgOf(pb)), [ran, pa, pb, presets, pipeline]); // eslint-disable-line
  // Does the form still describe the results on screen?
  const stale = useMemo(() => {
    if (!ran || !A || !B) return null;
    const changed = [ran.ia !== ia && "index A", ran.ib !== ib && "index B", ran.pa !== pa && "preset A",
      ran.pb !== pb && "preset B", ran.query !== query && "the question", ran.control !== control && "controlled mode",
      JSON.stringify(cfgOf(ran.pa)) !== JSON.stringify(ran.a) && "the Workbench settings"].filter(Boolean);
    return changed.length ? changed.join(", ") : null;
  }, [ran, A, B, ia, ib, pa, pb, query, control, pipeline, presets]); // eslint-disable-line
  const stage = (r: Run | null, s: string) => r?.events.find((e) => e.stage === s);
  const sameIndex = ran ? ran.ia === ran.ib : ia === ib;
  const ctx = (r: Run | null): number[] => stage(r, "context")?.data.blocks.filter((b: Any) => b.i >= 0).map((b: Any) => b.i) ?? [];
  const label = (r: Run | null, i: number) => { const c = r?.chunks[i]; return c ? `${(c.meta.title ?? "").slice(0, 18)} › ${(c.meta.section?.split(" > ").pop() ?? "").slice(0, 16)}` : `#${i}`; };
  const stagesAll = Array.from(new Set([...(A?.events ?? []), ...(B?.events ?? [])].map((e) => e.stage)));
  const maxMs = Math.max(...[...(A?.events ?? []), ...(B?.events ?? [])].map((e) => e.ms), 1);
  const Sel = ({ v, set }: { v: string; set: (s: string) => void }) => <select style={{ width: 170 }} value={v} onChange={(e) => set(e.target.value)}><option value="__current">Workbench settings</option>{presets.map((p) => <option key={p.name} value={p.name}>{p.name}</option>)}</select>;
  const Ix = ({ v, set }: { v: string; set: (s: string) => void }) => <select style={{ width: 170 }} value={v} onChange={(e) => set(e.target.value)}>{indexes.map((i) => <option key={i.id} value={i.id}>{i.name}</option>)}</select>;

  if (!indexes.length) return <div className="panel"><h3>Nothing to compare yet</h3><p className="muted" style={{ marginTop: 6 }}>Build an index, then save a preset or two in the Workbench.</p></div>;
  return (
    <>
      <div className="pagehead"><h2>Compare two runs</h2><p>One question, two configurations, side by side: what changed in the settings, where each spent its time and tokens, which passages reached the model, and the two answers. Save configurations as presets in the Workbench to pick them here.</p></div>
      {err && <div className="err" style={{ marginBottom: 12 }}>{err}</div>}
      {restoring && <div className="note small" style={{ marginBottom: 12 }}>Bringing back the last comparison…</div>}
      <div className="panel stack">
        <div className="row"><span className="tag" style={{ color: "var(--muted)", borderColor: "currentColor" }}>A</span><Sel v={pa} set={setPa} /><span className="small muted">on</span><Ix v={ia} set={setIa} />
          <span className="tag" style={{ color: "var(--bm25)", borderColor: "currentColor", marginLeft: 14 }}>B</span><Sel v={pb} set={setPb} /><span className="small muted">on</span><Ix v={ib} set={setIb} /></div>
        <div className="row"><input className="grow" type="text" value={query} onChange={(e) => setQuery(e.target.value)} onKeyDown={(e) => e.key === "Enter" && !busy && run()} aria-label="Question" /><button className="btn" disabled={busy || !query.trim()} onClick={run}>{busy ? "Running both…" : "Run both"}</button></div>
        <div className="row small"><label className="check small"><input type="checkbox" checked={control} onChange={(e) => setControl(e.target.checked)} />Controlled run: bypass the semantic cache and stop the router changing settings per question</label></div>
        {stale && <div className="note small">Showing the run that finished earlier. Since then you changed {stale} — run again to compare what the form now describes.</div>}
        <div className="row small" style={{ gap: 5 }}><span className="muted">{diffs.length ? (ran ? "Settings that differed" : "Settings that differ") : sameIndex ? "A and B are identical. Pick a preset or a second index." : "Same settings, different index."}</span>
          {diffs.slice(0, 14).map(([k, a, b]) => <span className="tag" key={k}>{k}: {JSON.stringify(a)} → <b style={{ color: "var(--bm25)" }}>{JSON.stringify(b)}</b></span>)}</div>
      </div>
      {A && B && <>
        <div className="grid2" style={{ marginTop: 14, gridTemplateColumns: "1fr 1fr" }}>{([["A", A, "var(--muted)"], ["B", B, "var(--bm25)"]] as [string, Run, string][]).map(([n, r, col]) => <div className="panel" key={n} style={{ marginTop: 0 }}>
          <div className="row"><span className="tag" style={{ color: col, borderColor: "currentColor" }}>{n}</span><span className="grow" /><span className="mono small muted">{(r.total / 1000).toFixed(1)} s, {(r.usage.tokens_in ?? 0) + (r.usage.tokens_out ?? 0)} tok{r.usage.cost ? `, $${r.usage.cost.toFixed(4)}` : ""}</span></div>
          <div className="answer" style={{ fontSize: 14.5, marginTop: 8 }}>{r.answer.replace(/<think>[\s\S]*?<\/think>/g, "") || <span className="muted">No answer (generation off).</span>}</div>
          {stage(r, "crag") && <div className="row small" style={{ marginTop: 8 }}><span className="muted">CRAG</span><span className={`tag ${stage(r, "crag")!.data.verdict === "correct" ? "ok" : "bad"}`}>{stage(r, "crag")!.data.verdict}</span></div>}
          {stage(r, "verify")?.data.supported_share != null && <div className="row small" style={{ marginTop: 4 }}><span className="muted">Grounded claims</span><span className="tag ok">{Math.round(stage(r, "verify")!.data.supported_share * 100)}%</span></div>}</div>)}</div>
        <div className="panel"><h3 style={{ marginBottom: 10 }}>What reached the model</h3>
          {sameIndex ? <RankFlow items={Array.from(new Set([...ctx(A), ...ctx(B)])).map((i) => ({ id: i, label: label(B.chunks[i] ? B : A, i), color: groups.color(groups.of((B.chunks[i] ?? A.chunks[i])?.doc_id ?? "")) }))}
            columns={[{ label: "A context", color: "var(--muted)", rank: (id: number) => rankIn(ctx(A), id) }, { label: "B context", color: "var(--bm25)", rank: (id: number) => rankIn(ctx(B), id) }]}
            note="Dashed lines are passages only one configuration delivered. Those are the ones to read when the answers differ." />
            : <div className="grid2" style={{ gridTemplateColumns: "1fr 1fr" }}>{[A, B].map((r, n) => <div key={n}>{stage(r, "context")?.data.blocks.map((b: Any) => <div className="chunkbox" key={b.n}><span className="tag">[{b.n}]</span> <b>{b.title}</b> <span className="muted">{b.section}</span><div className="small">{b.text.slice(0, 200)}…</div></div>)}</div>)}</div>}
          {!sameIndex && <p className="small muted" style={{ marginTop: 6 }}>Different indexes have different chunks, so passages are listed rather than linked.</p>}</div>
        <div className="panel"><h3 style={{ marginBottom: 10 }}>Stage by stage</h3>
          <table><thead><tr><th>Stage</th><th style={{ width: "26%" }}>Time, A over B</th><th className="num">Tokens A</th><th className="num">Tokens B</th><th>What differed</th></tr></thead><tbody>
            {stagesAll.map((s) => { const a = stage(A, s), b = stage(B, s); const tok = (e?: Any) => e ? (e.usage?.tokens_in ?? 0) + (e.usage?.tokens_out ?? 0) : null;
              const note = s === "retrieve" ? `${a?.data.lists.length ?? 0} vs ${b?.data.lists.length ?? 0} ranked lists${b?.data.filtered_out?.length || a?.data.filtered_out?.length ? `, freshness removed ${a?.data.filtered_out?.length ?? 0} vs ${b?.data.filtered_out?.length ?? 0}` : ""}`
                : s === "transform" ? `${a?.data.queries.length ?? 1} vs ${b?.data.queries.length ?? 1} queries` : s === "normalize" ? `${a?.data.steps.length ?? 0} vs ${b?.data.steps.length ?? 0} rules fired`
                : s === "crag" ? `${a?.data.verdict ?? "off"} vs ${b?.data.verdict ?? "off"}, kept ${a?.data.kept.length ?? "–"} vs ${b?.data.kept.length ?? "–"}` : s === "context" ? `${a?.data.tokens ?? 0} vs ${b?.data.tokens ?? 0} context tokens`
                : s === "route" ? `${a?.data.intent ?? "off"} vs ${b?.data.intent ?? "off"}` : s === "rerank" ? (sameIndex ? `top passage ${a?.data.reranked[0]?.i === b?.data.reranked[0]?.i ? "unchanged" : "changed"}` : "") : !a || !b ? `only in ${a ? "A" : "B"}` : "";
              return <tr key={s}><td><svg width="10" height="10" style={{ marginRight: 6 }}><rect width="10" height="10" rx="2" fill={STAGE_COLOR[s] ?? "var(--muted)"} /></svg>{s.replace("_", " ")}</td>
                <td><div className="funnel" style={{ gridTemplateColumns: "1fr" }}><div className="pair"><div><i style={{ width: `${((a?.ms ?? 0) / maxMs) * 80}%`, background: "var(--muted)" }} /><span>{a ? `${a.ms} ms` : "off"}</span></div><div><i style={{ width: `${((b?.ms ?? 0) / maxMs) * 80}%`, background: "var(--bm25)" }} /><span>{b ? `${b.ms} ms` : "off"}</span></div></div></div></td>
                <td className="num">{tok(a) ?? "–"}</td><td className="num">{tok(b) ?? "–"}</td><td className="small muted">{note}</td></tr>; })}</tbody></table></div>
      </>}
    </>
  );
}
