import { Fragment, useEffect, useState } from "react";
import { api, Any, deepMerge, usePersisted } from "../api";
import { CHANNEL } from "../viz";

import { DIAGNOSTIC, METRIC_LABELS, STAGES, STAGE_HELP, WHOLE_PROMPT, cutoffLabelFor, isLegacy, stageLabel } from "../stages";

const A_COLOR = "var(--muted)", B_COLOR = "var(--bm25)";
// Names come from METRIC_LABELS so this screen and Sweeps cannot call one number two things.
const METRICS: [string, string][] = [
  ["recall", "Share of the expected documents or evidence quotes found. The floor: if evidence is not retrieved, nothing downstream can fix it."],
  ["mrr", "Mean reciprocal rank of the first passage holding the evidence. Rewards putting it first, where the model reads most carefully."],
  // Not textbook nDCG: the ideal is the evidence-bearing positions packed at the top, times
  // coverage. Saying so beats letting a familiar name imply a definition this does not use.
  ["ndcg", "Position quality times coverage, between 0 and 1. Position quality discounts evidence found further down the list; coverage is the share of expected evidence found at all. A single expected quote at rank r scores 1 / log2(r + 1)."]];

function MetricCard({ label, help, a, b }: { label: string; help: string; a?: number; b?: number }) {
  const pct = (v?: number) => v === undefined ? "–" : `${Math.round(v * 100)}%`; const d = a !== undefined && b !== undefined ? Math.round((b - a) * 100) : null;
  return <div className="panel metric" style={{ marginTop: 0 }}><div className="lab">{label}</div>
    <div className="vals"><span className="a">{pct(a)}</span><span className="b">{pct(b)}</span>{d !== null && <span className={`d ${d > 0 ? "up" : d < 0 ? "down" : ""}`}>{d > 0 ? "+" : ""}{d} pts</span>}</div>
    <div className="track"><i style={{ width: `${(a ?? 0) * 100}%`, background: A_COLOR }} /></div><div className="track"><i style={{ width: `${(b ?? 0) * 100}%`, background: B_COLOR }} /></div>
    <p className="small muted" style={{ marginTop: 10 }}>{help}</p></div>;
}

function Rank({ r }: { r: number | null | undefined }) { return <span className={`rk ${r ? (r <= 3 ? "good" : "") : "miss"}`}>{r ?? "miss"}</span>; }

/** A delivered passage, shown in full on demand. Truncating under a heading that says "full trace"
 *  hides exactly the text you opened the trace to read. */
function Passage({ text }: { text: string }) {
  const [open, setOpen] = useState(false);
  const long = text.length > 300;
  return <div className="small">{open || !long ? text : `${text.slice(0, 300)}…`}
    {long && <> <button className="link" onClick={() => setOpen(!open)}>{open ? "show less" : `show all ${text.length} characters`}</button></>}</div>;
}

/** Where a question's expected evidence stood at each stage, and where it stopped being there. */
function EvidenceTrail({ row, stages }: { row: Any; stages: string[] }) {
  const seen = stages.filter((s) => s in (row.ranks ?? {}));
  if (!seen.length) return null;
  const lost = seen.find((s, i) => row.ranks[s] == null && i > 0 && row.ranks[seen[i - 1]] != null);
  return <div className="small" style={{ marginTop: 4 }}>
    {seen.map((s) => <span key={s} className={`tag ${row.ranks[s] ? "ok" : "bad"}`} style={{ margin: "0 3px 3px 0" }}
      title={STAGE_HELP[s]}>{s} {row.ranks[s] ?? "gone"}</span>)}
    {lost ? <div className="muted" style={{ marginTop: 3 }}>Expected evidence was still there going into <b>{lost}</b>, and not after it.</div>
      : row.ranks[seen[0]] == null ? <div className="muted" style={{ marginTop: 3 }}>Retrieval never found it, so nothing downstream could.</div>
      : <div className="muted" style={{ marginTop: 3 }}>Survived every stage.</div>}</div>;
}

export default function Evaluate({ pipeline, defaults }: { pipeline: Any; defaults: Any }) {
  const [samples, setSamples] = useState<Any[]>([]);
  // A saved preset can be picked here, so changing what is measured no longer means walking to the
  // Workbench and back with the results left behind.
  const [presets, setPresets] = useState<Any[]>([]);
  const [preset, setPreset] = usePersisted("raglab.evaluate.preset", "__current");
  const [control, setControl] = usePersisted("raglab.evaluate.control", false);
  const [k, setK] = usePersisted("raglab.evaluate.k", 5);
  const [mix, setMix] = usePersisted<Record<string, number>>("raglab.evaluate.mix",
    { simple: 40, paraphrased: 30, multi: 20, unanswerable: 10 });
  // A and B name runs that may have been deleted, so they are not remembered across tabs.
  const [aId, setAId] = useState(""); const [bId, setBId] = useState("");
  const [stage, setStage] = usePersisted("raglab.evaluate.stage", "context");
  const [indexes, setIndexes] = useState<Any[]>([]); const [indexId, setIndexId] = usePersisted("raglab.evaluate.index", "");
  const [sets, setSets] = useState<Any[]>([]); const [setId, setSetId] = usePersisted("raglab.evaluate.set", "");
  const [runs, setRuns] = useState<Any[]>([]); const [jobs, setJobs] = useState<Any>({});
  const [n, setN] = usePersisted("raglab.evaluate.n", 12);
  const [label, setLabel] = usePersisted("raglab.evaluate.label", "baseline");
  const [judge, setJudge] = usePersisted("raglab.evaluate.judge", false);
  const [withGen, setWithGen] = usePersisted("raglab.evaluate.withGeneration", false);
  const [paste, setPaste] = usePersisted("raglab.evaluate.paste", "");
  const [busy, setBusy] = useState(""); const [err, setErr] = useState(""); const [open, setOpen] = useState<Any>(null);

  const loadSets = () => api.get("/api/evalsets").then((s: Any[]) => { setSets(s); if (!s.some((x) => x.id === setId) && s[0]) setSetId(s[0].id); });
  const cfgOf = (name: string) => name === "__current" ? pipeline : deepMerge(pipeline, presets.find((p) => p.name === name)?.config ?? {});
  const loadRuns = () => setId && api.get(`/api/evalruns?evalset_id=${setId}`).then((r) => { setRuns(r.runs); setJobs(r.jobs); });
  useEffect(() => { api.get("/api/indexes").then((ix: Any[]) => { const r = ix.filter((i) => i.status === "ready"); setIndexes(r);
      // Keep the remembered index; only pick one when it is missing or has been deleted.
      setIndexId((cur) => (r.some((i) => i.id === cur) ? cur : r[0]?.id ?? "")); }); loadSets(); api.get("/api/samples").then(setSamples); api.get("/api/presets").then(setPresets); }, []); // eslint-disable-line
  useEffect(() => { loadRuns(); }, [setId]); // eslint-disable-line
  const active = Object.values(jobs).some((j: Any) => j.status === "running");
  useEffect(() => { if (!active) return; const t = setInterval(loadRuns, 1000); return () => clearInterval(t); }, [active, setId]); // eslint-disable-line

  const guard = async (what: string, fn: () => Promise<unknown>) => { setErr(""); setBusy(what); try { await fn(); } catch (e) { setErr(String(e)); } setBusy(""); };
  const pct = (v?: number | null) => v == null ? "–" : `${Math.round(v * 100)}%`;
  // Every evaluated question now stores its trace, so a failure can be opened where it happened
  // instead of being re-run by hand in the Workbench.
  const [trace, setTrace] = useState("");
  const [traceData, setTraceData] = useState<Any>(null);
  useEffect(() => { if (!trace) { setTraceData(null); return; }
    api.get(`/api/traces/${trace}`).then(setTraceData).catch((e) => setErr(String(e.message))); }, [trace]);
  const current = sets.find((s) => s.id === setId);
  // A checkpointed run is saved after every question, so a row can exist with no metrics yet.
  // Those are progress, not results: they are listed, but never compared.
  const done = runs.filter((r) => !r.metrics?.partial && r.metrics?.stages);
  const name = (id: string) => indexes.find((i) => i.id === id)?.name ?? id;

  return (
    <>
      <div className="pagehead"><h2>Evaluation</h2>
        <p>Gold labels never depend on chunking: either the documents a careful human would cite, or verbatim evidence quotes. So one question set scores any index. Every metric is reported after each stage, which shows where the pipeline wins the evidence and where it loses it.</p></div>
      {err && <div className="err" style={{ marginBottom: 12 }}>{err}</div>}
      <div className="split">
        <div className="stack">
          <div className="panel stack"><h3>Question sets</h3>
            <div className="fields"><label className="f"><span>Source index</span><select value={indexId} onChange={(e) => setIndexId(e.target.value)}>{indexes.map((i) => <option key={i.id} value={i.id}>{i.name}</option>)}</select></label>
              <label className="f"><span>Questions</span><input type="number" value={n} onChange={(e) => setN(Number(e.target.value))} /></label></div>
            <button className="btn" disabled={!indexId || !!busy} onClick={() => guard("gen", async () => { const r = await api.post("/api/evalsets/generate", { index_id: indexId, n, name: `generated from ${name(indexId)}`, mix }); await loadSets(); setSetId(r.id); })}>{busy === "gen" ? "Writing questions…" : "Generate with the generator model"}</button>
            <div><div className="small muted" style={{ marginBottom: 4 }}>Question mix (%). “Simple” questions are written from the chunk that answers them, so they are easy; the others are the real test.</div>
              <div className="fields">{Object.keys(mix).map((t) => <label className="f" key={t}><span>{t}</span><input type="number" value={mix[t]} onChange={(e) => setMix({ ...mix, [t]: Number(e.target.value) })} /></label>)}</div></div>
            {samples.filter((sm) => sm.golden).map((sm) => <button key={sm.name} className="btn ghost" onClick={() => guard("golden", async () => { const r = await api.post(`/api/evalsets/golden?name=${sm.name}`); await loadSets(); setSetId(r.id); })}>Load the hand-written {sm.name} golden set</button>)}
            <details><summary className="small muted">Or paste your own as JSON</summary>
              <textarea style={{ marginTop: 6 }} value={paste} onChange={(e) => setPaste(e.target.value)} placeholder={'[{"question": "...", "relevant_docs": ["DOC-KEY"], "probes": "the trap this question sets"}]\nor [{"question": "...", "answer": "...", "evidence": "verbatim quote"}]'} />
              <button className="btn ghost" style={{ marginTop: 6 }} disabled={!paste.trim()} onClick={() => guard("paste", async () => { const r = await api.post("/api/evalsets", { name: "pasted", items: JSON.parse(paste) }); setPaste(""); await loadSets(); setSetId(r.id); })}>Save question set</button></details>
            {sets.map((s) => <label className="check small" key={s.id}><input type="radio" name="set" checked={s.id === setId} onChange={() => setSetId(s.id)} />{s.name} ({s.items.length})
              <button className="link" onClick={() => guard("del", async () => { await api.del(`/api/evalsets/${s.id}`); await loadSets(); })}>Delete</button></label>)}
          </div>
          <div className="panel stack"><h3>Run</h3>
            <p className="small muted">Runs a saved preset, or whatever the Workbench currently has. Results stay on this page while you change what is measured.</p>
            <label className="f"><span>Settings</span><select value={preset} onChange={(e) => setPreset(e.target.value)}>
              <option value="__current">Workbench settings</option>{presets.map((p) => <option key={p.name} value={p.name}>{p.name}</option>)}</select></label>
            <div className="fields"><label className="f"><span>Index</span><select value={indexId} onChange={(e) => setIndexId(e.target.value)}>{indexes.map((i) => <option key={i.id} value={i.id}>{i.name}</option>)}</select></label>
              <label className="f"><span>Label</span><input type="text" value={label} onChange={(e) => setLabel(e.target.value)} /></label>
              <label className="f"><span>Score the top k</span><input type="number" value={k} onChange={(e) => setK(Number(e.target.value))} /></label></div>
            <label className="check small"><input type="checkbox" checked={withGen} onChange={(e) => { setWithGen(e.target.checked); if (!e.target.checked) setJudge(false); }} />Generate answers too (slower)</label>
            <label className="check small"><input type="checkbox" checked={judge} disabled={!withGen} onChange={(e) => setJudge(e.target.checked)} />LLM judge: faithfulness and correctness</label>
            <label className="check small"><input type="checkbox" checked={control} onChange={(e) => setControl(e.target.checked)} />Controlled run: bypass the cache and stop the router changing settings per question</label>
            {!withGen && <div className="note small">With generation off, abstention is measured from the retrieval grader alone — no answer is written, so there is nothing to decline. Turn generation on to score what the model actually says.</div>}
            <button className="btn" disabled={!setId || !indexId || active} onClick={() => guard("run", async () => { const base = cfgOf(preset); await api.post("/api/evalruns", { evalset_id: setId, index_id: indexId, label, judge, k, control,
              config: { ...base, generate: { ...base.generate, enabled: withGen } } }); await loadRuns(); })}>{active ? "Running…" : "Run evaluation"}</button>
            <button className="btn ghost" disabled={!setId || !indexId || active} onClick={() => guard("run", async () => { await api.post("/api/evalruns", { evalset_id: setId, index_id: indexId, label: "lexical baseline", judge: false, k,
              config: { ...defaults, route: { enabled: false, apply: false }, multi_query: { enabled: false, n: 0 }, hyde: { enabled: false }, normalize: { enabled: false }, freshness: { enabled: false }, dense: { ...defaults.dense, enabled: false }, graph: { ...defaults.graph, enabled: false },
                rerank: { ...defaults.rerank, enabled: false }, crag: { ...defaults.crag, enabled: false }, generate: { ...defaults.generate, enabled: false } } }); await loadRuns(); })}>Run the frozen lexical baseline (BM25 only)</button>
            {Object.entries(jobs).filter(([, j]: Any) => j.status !== "done").map(([id, j]: Any) => <div key={id}>{j.status === "failed" || j.status === "cancelled" ? <div className={j.status === "cancelled" ? "note" : "err"}>{j.error}</div> :
              <><div className="row small"><span className="muted grow">{j.done}/{j.total} questions{j.partial ? `, ${j.partial} saved` : ""}</span>
                <button className="link" onClick={() => api.post(`/api/evalruns/${id}/stop`).then(() => { loadRuns(); })}>Stop</button></div>
              <div className="progress"><i style={{ width: `${(100 * j.done) / Math.max(j.total, 1)}%` }} /></div></>}</div>)}
          </div>
        </div>
        <div>
          {done.length > 0 && (() => { const A = done.find((r) => r.id === aId) ?? done.find((r) => r.label === "lexical baseline") ?? done[0]; const B = done.find((r) => r.id === bId) ?? done[done.length - 1];
            const same = A.id === B.id; const st = (r: Any, s: string) => r.metrics.stages[s]; const stages = STAGES.filter((s) => st(A, s) || st(B, s));
            // A run saved before the metric definitions changed keeps its own labels; today's
            // footnotes would assert things about it that are not true.
            const legacyBoth = isLegacy(A.metrics) && isLegacy(B.metrics);
            const legacyEither = isLegacy(B.metrics) || (!same && isLegacy(A.metrics));
            // Which stages a run measured varies, so fall back when the remembered one is absent.
            const shown = stages.includes(stage) ? stage : stages[stages.length - 1] ?? "";
            return <><div className="panel"><div className="row">
              <label className="row small" style={{ gap: 6 }}><span className="tag" style={{ color: A_COLOR, borderColor: "currentColor" }}>A</span><select style={{ width: 190 }} value={A.id} onChange={(e) => setAId(e.target.value)}>{done.map((r) => <option key={r.id} value={r.id}>{r.label} on {name(r.index_id)}</option>)}</select></label>
              <label className="row small" style={{ gap: 6 }}><span className="tag" style={{ color: B_COLOR, borderColor: "currentColor" }}>B</span><select style={{ width: 190 }} value={B.id} onChange={(e) => setBId(e.target.value)}>{done.map((r) => <option key={r.id} value={r.id}>{r.label} on {name(r.index_id)}</option>)}</select></label>
              <label className="row small muted" style={{ gap: 6 }}>measured at<select style={{ width: 180 }} value={shown} onChange={(e) => setStage(e.target.value)}>{stages.map((s) => <option key={s} value={s}>{legacyEither ? s : stageLabel(s)}</option>)}</select></label>
              <span className="grow" /><span className="tag">{B.metrics.n} questions</span><span className="tag">top {B.metrics.k}</span><span className="tag">{B.metrics.latency_ms} ms</span><span className="tag">{B.metrics.tokens ?? 0} tok</span>{B.metrics.cost > 0 && <span className="tag">${B.metrics.cost.toFixed(4)} / question</span>}</div>
              {same && <p className="small muted" style={{ marginTop: 8 }}>Only one run so far. Run the lexical baseline, or change the Workbench settings and run again, to get a comparison.</p>}
              {legacyEither && <div className="err" style={{ marginTop: 8 }}>
                {[isLegacy(A.metrics) && !same ? `A (${A.label})` : "", isLegacy(B.metrics) ? `B (${B.label})` : ""].filter(Boolean).join(" and ")} {" "}
                {legacyBoth || same ? "was" : "were"} measured before the current definitions. Back then <code>context</code> recall was cut at the
                top k rather than covering the whole prompt, and the abstention rate averaged questions that nothing had decided.
                Those numbers are shown as recorded and are not comparable with a current run — re-run to compare like with like.</div>}</div>
            <div className="metrics" style={{ marginTop: 14 }}>{METRICS.map(([k, help]) => <MetricCard key={k} label={`${METRIC_LABELS[k]} ${cutoffLabelFor(shown, B.metrics.k, B.metrics)} at ${shown}`} help={`${help} ${legacyEither ? "" : STAGE_HELP[shown] ?? ""}`} a={same ? undefined : st(A, shown)?.[k]} b={st(B, shown)?.[k]} />)}
              {B.metrics.abstention != null
                ? <MetricCard label="ABSTENTION on unanswerable"
                    help={`Out-of-corpus questions where the pipeline declined. False abstention on answerable ones: ${pct(B.metrics.false_abstention)}. `
                      + (B.metrics.measured?.answers
                        ? `Measured from the answer itself${B.metrics.measured?.grader ? ` (the grader alone would say ${pct(B.metrics.abstention_by?.crag)})` : ""}.`
                        : B.metrics.measured
                          ? "Measured from the retrieval grader only: generation was off, so no answer was written to decline."
                          // Older runs recorded no breakdown, and their rate counted questions that
                          // nothing had decided. Saying which signal produced it would be a guess.
                          : "Recorded before this lab distinguished the grader's verdict from the answer's, and before questions that nothing decided were excluded. Re-run to measure it.")}
                    a={same ? undefined : A.metrics.abstention ?? undefined} b={B.metrics.abstention} />
                // Nothing decided, so there is no number. Saying so beats hiding the card, which
                // reads as "this run had no abstention problem".
                : B.rows.some((r: Any) => r.type === "unanswerable") && <div className="panel metric" style={{ marginTop: 0 }}>
                    <div className="lab">ABSTENTION on unanswerable</div>
                    <div className="vals"><span className="a">not measured</span></div>
                    <p className="small muted" style={{ marginTop: 10 }}>This run had neither the retrieval grader nor
                      generation enabled, so nothing decided whether to decline. The questions were never asked, which is
                      not the same as answering them wrongly. Turn on corrective RAG, generation, or both to measure it.</p></div>}
              {B.metrics.groundedness != null && <MetricCard label="GROUNDEDNESS" help="Share of answer claims the checker found support for in the supplied passages." a={same ? undefined : A.metrics.groundedness ?? undefined} b={B.metrics.groundedness} />}
              {(A.metrics.faithfulness !== undefined || B.metrics.faithfulness !== undefined) && <MetricCard label="FAITHFULNESS (LLM judge)" help="Share of the answer's claims that the supplied context supports." a={same ? undefined : A.metrics.faithfulness} b={B.metrics.faithfulness} />}</div>
            <div className="panel"><div className="row" style={{ alignItems: "baseline", marginBottom: 12 }}><h3 className="grow">Where evidence is won and lost</h3><span className="small muted">recall@{B.metrics.k} after each stage, A over B</span></div>
              <p className="small muted" style={{ marginBottom: 10 }}>† per-channel rows show the best query variant for that channel, picked with the gold label. They say what a channel could reach, not what the pipeline delivered, so they are not comparable with the rows below them.
                {!legacyEither && <>&nbsp;* the prompt is scored whole rather than at the top {B.metrics.k}: every passage in it went to the model, and it is already capped by max passages.</>}</p>
              <div className="funnel">{stages.map((s) => <Fragment key={s}><span className="mono" style={{ color: CHANNEL[s] }} title={STAGE_HELP[s]}>{s}{DIAGNOSTIC.has(s) && <sup className="muted" title="Best query variant, chosen with the gold label">†</sup>}{WHOLE_PROMPT.has(s) && !legacyEither && <sup className="muted" title="Scored over the whole prompt, not the top k">*</sup>}</span><div className="pair">
                {(same ? [B] : [A, B]).map((r, n) => { const v = st(r, s)?.recall ?? st(r, s)?.hit; return <div key={n}><i style={{ width: `${(v ?? 0) * 82}%`, background: same || n ? B_COLOR : A_COLOR }} /><span>{v === undefined ? "not used" : `${Math.round(v * 100)}%`}</span></div>; })}</div></Fragment>)}</div></div>
            {Object.keys(B.metrics.by_type ?? {}).length > 1 && <div className="panel"><h3 style={{ marginBottom: 8 }}>By question type</h3><div className="funnel" style={{ gridTemplateColumns: "110px minmax(0,1fr)" }}>{Object.entries(B.metrics.by_type).map(([t, v]: Any) => <Fragment key={t}><span className="mono">{t} <span className="muted">({v.n})</span></span>
              <div className="pair">{(same ? [B] : [A, B]).map((r, n) => { const x = r.metrics.by_type?.[t]?.hit; return <div key={n}><i style={{ width: `${(x ?? 0) * 82}%`, background: same || n ? B_COLOR : A_COLOR }} /><span>{x == null ? "–" : `${Math.round(x * 100)}% hit`}</span></div>; })}</div></Fragment>)}</div></div>}
            <div className="panel"><h3>Per-question breakdown</h3><p className="small muted" style={{ marginTop: 2 }}>Each question names the trap it sets. Green documents are ones a careful human would cite.</p><div style={{ overflowX: "auto" }}><table style={{ marginTop: 8 }}><thead><tr><th>Question</th><th>Probes</th><th>Expected</th>{!same && <th>A delivered</th>}<th>B delivered</th><th className="num">rank</th><th className="num">RR</th></tr></thead>
              <tbody>{B.rows.map((row: Any, i: number) => { const ra = A.rows.find((x: Any) => x.question === row.question)?.ranks[shown]; const rb = row.ranks[shown]; const rr = (r?: number | null) => (r ? 1 / r : 0);
                const arow = A.rows.find((x: Any) => x.question === row.question); const Docs = ({ r }: { r: Any }) => {
                  if (r?.type !== "unanswerable") return <td>{(r?.retrieved ?? []).slice(0, 5).map((d: Any, n: number) => <span key={n} className={`tag ${d.ok ? "ok" : ""}`} style={{ margin: "0 3px 3px 0" }}>{String(d.doc).slice(0, 18)}</span>)}</td>;
                  // A row from before the split records neither component; it is not undecided,
                  // it is unbroken-down, so it keeps the verdict it was saved with.
                  const older = r.answer_abstained === undefined && r.crag_abstained === undefined;
                  const decided = older || r.answer_abstained != null || r.crag_abstained != null;
                  return <td>{decided ? <span className={`tag ${r.abstained ? "ok" : "bad"}`} title={older ? "Recorded before the grader's verdict and the answer's were separated" : undefined}>{r.abstained ? "abstained" : "answered anyway"}{older ? " *" : ""}</span>
                    : <span className="tag" title="Neither the grader nor the generator ran, so nothing decided">not measured</span>}</td>;
                };
                return <tr key={i}><td style={{ minWidth: 200 }}>{row.question}{row.judge?.reason && <div className="small muted">{row.judge.reason}</div>}
                  <EvidenceTrail row={row} stages={stages} />
                  {row.trace_id && <div><button className="link small" onClick={() => setTrace(row.trace_id)}>Open the full trace</button></div>}</td><td className="small muted" style={{ minWidth: 180 }}>{row.probes}</td>
                  <td>{(row.expected ?? []).map((e: string) => <span key={e} className="tag ok" style={{ margin: "0 3px 3px 0" }}>{e.slice(0, 22)}</span>)}</td>{!same && <Docs r={arow} />}<Docs r={row} /><td className="num">{!same && <><Rank r={ra} /> → </>}<Rank r={rb} /></td>
                  <td className="num" style={{ color: !same && rr(rb) > rr(ra) ? "var(--ok)" : !same && rr(rb) < rr(ra) ? "var(--bad)" : undefined }}>{same ? rr(rb).toFixed(2) : `${rr(ra).toFixed(2)} → ${rr(rb).toFixed(2)}`}</td></tr>; })}</tbody></table></div></div></>; })()}
          <div className="panel"><h3>All runs on “{current?.name ?? "no set selected"}”</h3>
            {!runs.length && <p className="muted" style={{ marginTop: 8 }}>No runs yet. Pick a question set and run an evaluation.</p>}
            {!!runs.length && <div style={{ overflowX: "auto" }}><table style={{ marginTop: 8 }}><thead><tr><th>Run</th><th>Index</th>{STAGES.map((s) => <th className="num" key={s}>{s}<br />recall / MRR</th>)}<th className="num">abstain</th><th className="num">grounded</th><th className="num">tokens</th><th className="num">cost</th><th className="num">ms</th><th /></tr></thead>
              <tbody>{runs.map((r) => <tr key={r.id}><td><button className="link" onClick={() => setOpen(r)}>{r.label}</button>
                {r.metrics?.partial && <span className="tag bad" title="Saved as it ran; the run did not finish">{r.metrics.n} of {current?.items?.length ?? "?"}, unfinished</span>}
                {isLegacy(r.metrics) && <span className="tag" title="Measured before the current metric definitions; not comparable with a current run">older metrics</span>}</td><td>{name(r.index_id)}</td>
                {STAGES.map((s) => { const m = r.metrics?.stages?.[s]; return <td className="num" key={s}>{m ? `${(m.recall ?? m.hit).toFixed(2)} / ${m.mrr.toFixed(2)}` : "–"}</td>; })}
                <td className="num">{r.metrics?.abstention == null ? "–" : `${Math.round(r.metrics.abstention * 100)}%`}</td><td className="num">{r.metrics?.groundedness == null ? "–" : `${Math.round(r.metrics.groundedness * 100)}%`}</td><td className="num">{r.metrics?.tokens || "–"}</td><td className="num">{r.metrics?.cost ? `$${r.metrics.cost.toFixed(4)}` : "–"}</td><td className="num">{r.metrics?.latency_ms || "–"}</td>
                <td className="num"><button className="link" onClick={() => guard("del", async () => { await api.del(`/api/evalruns/${r.id}`); await loadRuns(); })}>Delete</button></td></tr>)}</tbody></table></div>}
          </div>
          {open && <div className="panel"><div className="row"><h3 className="grow">{open.label}: rank of first evidence per question</h3><button className="link" onClick={() => setOpen(null)}>Close</button></div>
            <div style={{ overflowX: "auto" }}><table style={{ marginTop: 8 }}><thead><tr><th>Question</th>{STAGES.filter((s) => (open.rows ?? []).some((r: Any) => s in (r.ranks ?? {}))).map((s) => <th className="num" key={s}>{s}</th>)}</tr></thead>
              <tbody>{(open.rows ?? []).map((row: Any, i: number) => <tr key={i}><td>{row.question}{row.judge?.reason && <div className="small muted">{row.judge.reason}</div>}</td>
                {STAGES.filter((s) => s in (row.ranks ?? {})).map((s) => <td className="num" key={s} style={{ color: row.ranks[s] ? undefined : "var(--bad)" }}>{row.ranks[s] ?? "miss"}</td>)}</tr>)}</tbody></table></div>
            <details style={{ marginTop: 8 }}><summary className="small muted">Pipeline settings used</summary><pre>{JSON.stringify(open.config, null, 2)}</pre></details></div>}
          {trace && <div className="panel">
            <div className="row"><h3 className="grow">Trace: {traceData?.query ?? "loading…"}</h3><button className="link" onClick={() => setTrace("")}>Close</button></div>
            {!traceData && <p className="muted small" style={{ marginTop: 8 }}>Loading the trace…</p>}
            {traceData && <div className="stack" style={{ marginTop: 8 }}>
              {traceData.trace.map((ev: Any, n: number) => <details key={n}>
                <summary className="small"><b>{ev.title}</b> <span className="muted">{ev.ms} ms{ev.calls?.length ? `, ${ev.calls.length} model call${ev.calls.length > 1 ? "s" : ""}` : ""}</span></summary>
                {ev.stage === "context"
                  ? <div style={{ marginTop: 6 }}>{ev.data.blocks.map((b: Any) => <div className="chunkbox" key={b.n}>
                      <span className="tag">[{b.n}]</span> <b>{b.title}</b> <span className="muted">{b.section}</span> <span className="tag">{b.tokens} tok</span>
                      {b.expanded && <span className="tag ok">parent</span>}{b.refined && <span className="tag ok">refined</span>}{b.overflow && <span className="tag bad">over budget</span>}
                      <Passage text={b.text} /></div>)}
                      {!!ev.data.dropped?.length && <div className="note small">Budget dropped {ev.data.dropped.length}: {ev.data.dropped.map((d: Any) => `${d.title || `#${d.i}`} (${d.tokens} tok)`).join(", ")}</div>}
                    </div>
                  : <pre style={{ marginTop: 6, maxHeight: 420, overflow: "auto" }}>{JSON.stringify(ev.data, null, 2)}</pre>}
              </details>)}
              <details><summary className="small muted">Pipeline settings that actually ran</summary><pre>{JSON.stringify(traceData.config, null, 2)}</pre></details>
            </div>}
          </div>}
          {current && <div className="panel"><h3>Questions</h3><table style={{ marginTop: 8 }}><tbody>{current.items.map((it: Any, i: number) => <tr key={i}><td style={{ width: 110 }}><span className="tag">{it.type ?? (it.relevant_docs ? "golden" : "simple")}</span></td><td>{it.question}<div className="small muted">{it.probes}</div>
              <div className="small muted">{it.relevant_docs ? it.relevant_docs.join(", ") : Array.isArray(it.evidence) ? it.evidence.join("  +  ") : it.evidence}</div></td></tr>)}</tbody></table></div>}
        </div>
      </div>
    </>
  );
}
