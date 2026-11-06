import { useEffect, useState } from "react";
import { api, Any , usePersisted } from "../api";
import { CHANNEL, LineChart } from "../viz";
import { DIAGNOSTIC, HNSW_RECALL_HELP, METRIC_SHORT, STAGES, STAGE_HELP, WHOLE_PROMPT, hnswRecallCell, isLegacy, seriesLabel, stageLabel } from "../stages";

const PARAMS: [string, string, string, string][] = [
  ["dense.ef_search", "HNSW ef_search", "32, 64, 128, 256, 512",
   "Recall against latency: the approximation knob. Values below dense k run at k, since HNSW cannot return k neighbours from a narrower search — keep them above it or two points measure the same run."],
  ["rerank.top_n", "Rerank keep-top", "1, 2, 4, 6, 8, 12", "How many passages survive reranking."],
  ["fusion.rrf_k", "RRF k", "1, 10, 30, 60, 120", "How strongly fusion favours top ranks."],
  ["fusion.weights.bm25", "BM25 weight in fusion", "0, 0.5, 1, 1.5, 2, 3", "Lexical against dense influence."],
  ["bm25.b", "BM25 b (length normalisation)", "0, 0.25, 0.5, 0.75, 1", "How much long chunks are penalised."],
  ["bm25.k1", "BM25 k1 (term saturation)", "0.6, 1.0, 1.5, 2.0, 3.0", "How fast repeated terms stop adding score."],
  ["multi_query.n", "Multi-query variants", "0, 1, 2, 3, 5", "More variants cost a model call and widen recall."],
  ["context.max_chunks", "Max context passages", "1, 2, 3, 4, 6, 8", "What actually reaches the model."],
  ["dense.k", "Dense candidates k", "5, 10, 20, 30, 60", "Candidate pool size before fusion."],
  ["dense.mode", "Dense search: approximate or exact", "hnsw, exact", "What the HNSW approximation costs in recall, against an exhaustive search."],
  ["fusion.method", "Fusion rule", "rrf, weighted", "Reciprocal rank against weighted min-max scores."],
  ["context.order", "Context order", "rank, edges", "Best evidence first, or split between the start and the end of the prompt."],
  ["index", "Index (chunking strategy and size)", "", "Compare index-time choices: build one index per chunk size or strategy, then sweep across them."],
];

//: Settings whose values are words rather than numbers. Mirrors CHOICES in raglab/pipeline.py.
const CATEGORICAL: Record<string, string[]> = {
  "dense.mode": ["hnsw", "exact"],
  "fusion.method": ["rrf", "weighted"],
  "context.order": ["rank", "edges"],
};

export default function Sweeps({ pipeline }: { pipeline: Any }) {
  const [indexes, setIndexes] = useState<Any[]>([]); const [sets, setSets] = useState<Any[]>([]); const [setId, setSetId] = usePersisted("raglab.sweeps.set", ""); const [indexId, setIndexId] = useState(localStorage.getItem("raglab.index") ?? "");
  // Drop a remembered parameter that no longer exists, so `meta` below cannot come back undefined.
  const [param, setParam] = usePersisted("raglab.sweeps.param", PARAMS[0][0],
                                         (v, init) => (PARAMS.some((p) => p[0] === v) ? v as string : init));
  const [values, setValues] = usePersisted("raglab.sweeps.values", PARAMS[0][2]);
  const [picked, setPicked] = usePersisted<string[]>("raglab.sweeps.picked", []);
  const [stage, setStage] = usePersisted("raglab.sweeps.stage", "context");
  const [k, setK] = usePersisted("raglab.sweeps.k", 5);
  const [sharedRules, setSharedRules] = usePersisted("raglab.sweeps.sharedRules", false);
  const [sweeps, setSweeps] = useState<Any[]>([]); const [jobs, setJobs] = useState<Any>({}); const [open, setOpen] = useState(""); const [err, setErr] = useState("");

  const load = () => setId && api.get(`/api/sweeps?evalset_id=${setId}`).then((r) => { setSweeps(r.sweeps); setJobs(r.jobs); setOpen((o) => o || r.sweeps[0]?.id || ""); });
  useEffect(() => { api.get("/api/indexes").then((ix: Any[]) => { const r = ix.filter((i) => i.status === "ready"); setIndexes(r); if (!r.some((i) => i.id === indexId) && r[0]) setIndexId(r[0].id);
      // A remembered pick whose index is gone renders no checkbox, so drop it rather than
      // letting an invisible id be submitted with the sweep.
      setPicked((cur) => { const live = cur.filter((id) => r.some((i) => i.id === id)); return live.length === cur.length ? cur : live; }); });
    api.get("/api/evalsets").then((s: Any[]) => { setSets(s);
      setSetId((cur) => (s.some((x) => x.id === cur) ? cur : s[0]?.id ?? "")); }); }, []); // eslint-disable-line
  useEffect(() => { setOpen(""); load(); }, [setId]); // eslint-disable-line
  const active = Object.values(jobs).some((j: Any) => j.status === "running");
  useEffect(() => { if (!active) return; const t = setInterval(load, 1000); return () => clearInterval(t); }, [active, setId]); // eslint-disable-line

  const meta = PARAMS.find((p) => p[0] === param)!;
  // Number("typo") is NaN, and JSON.stringify(NaN) is null, so a typo used to travel to the server
  // as an absent value and silently remove the very limit it was meant to vary.
  const choices = CATEGORICAL[param];
  const entries = values.split(",").map((v) => v.trim()).filter(Boolean);
  const badValues = param === "index" ? []
    : choices ? entries.filter((v) => !choices.includes(v))
    : entries.filter((v) => !Number.isFinite(Number(v)));
  const start = async () => { setErr("");
    if (badValues.length) { setErr(`${choices ? `Not one of ${choices.join(", ")}` : "Not a number"}: ${badValues.join(", ")}`); return; }
    try { const vals = param === "index" ? picked : choices ? entries : entries.map(Number);
    const r = await api.post("/api/sweeps", { evalset_id: setId, index_id: indexId, param, values: vals, config: pipeline, k, label: meta[1], shared_rules: sharedRules }); setOpen(r.id); await load(); } catch (e) { setErr(String((e as Error).message)); } };
  const cur = sweeps.find((s) => s.id === open); const st = (p: Any, s: string, m: string) => p.metrics.stages[s]?.[m] ?? null;
  // Points carry the metric version that produced them, so a sweep from before the definitions
  // changed keeps its own labels rather than being relabelled with today's meaning.
  const first = cur?.points?.[0]?.metrics ?? null;
  const legacy = !!cur && isLegacy(first);
  // Mirrors raglab/evals.py STAGES. `delivered` was missing here, so a chunking or parent-expansion
  // sweep could not plot the one stage that says what the model actually read.
  const stages = cur ? STAGES.filter((s) => cur.points.some((p: Any) => p.metrics.stages[s])) : [];
  // Which stages exist depends on the sweep being viewed, so a remembered one can be absent.
  useEffect(() => { if (stages.length && !stages.includes(stage)) setStage(stages[stages.length - 1]); }, [stages, stage, setStage]);
  // The approximation's own number, plotted only for the knob that trades it, where a flat line of
  // gold-label metrics would otherwise say the knob does nothing.
  const showHnsw = cur?.spec?.param === "dense.ef_search" && cur.points.some((p: Any) => p.metrics.hnsw_recall != null);

  return (
    <>
      <div className="pagehead"><h2>Parameter sweeps</h2><p>Pick one knob, run the question set across its values, and read the curve. Everything else comes from the Workbench's current settings. Sweeps run controlled: generation is off so they measure retrieval, the semantic cache is bypassed, and adaptive routing cannot rewrite the knob under test.</p></div>
      {err && <div className="err" style={{ marginBottom: 12 }}>{err}</div>}
      <div className="split">
        <div className="panel stack"><h3>New sweep</h3>
          <label className="f"><span>Question set</span><select value={setId} onChange={(e) => setSetId(e.target.value)}>{sets.map((s) => <option key={s.id} value={s.id}>{s.name} ({s.items.length})</option>)}</select></label>
          {!sets.length && <p className="small muted">Create a question set on the Evaluation tab first.</p>}
          <label className="f"><span>Parameter</span><select value={param} onChange={(e) => { setParam(e.target.value); setValues(PARAMS.find((p) => p[0] === e.target.value)![2]); }}>{PARAMS.map((p) => <option key={p[0]} value={p[0]}>{p[1]}</option>)}</select></label>
          <p className="small muted">{meta[3]}</p>
          {param === "index" ? <div>{indexes.map((i) => <label className="check small" key={i.id} style={{ marginTop: 4 }}><input type="checkbox" checked={picked.includes(i.id)} onChange={(e) => setPicked(e.target.checked ? [...picked, i.id] : picked.filter((x) => x !== i.id))} />{i.name} <span className="muted">{i.config.chunking.strategy} {i.config.chunking.size}, {i.stats.chunks} chunks</span></label>)}</div>
            : <><label className="f"><span>Values, comma separated</span><input type="text" className="mono" value={values} aria-invalid={badValues.length ? true : undefined} onChange={(e) => setValues(e.target.value)} />
              {!!badValues.length && <span className="small bad">{choices ? `Not one of ${choices.join(", ")}` : "Not a number"}: {badValues.join(", ")}</span>}</label>
              <label className="f"><span>Index</span><select value={indexId} onChange={(e) => { setIndexId(e.target.value); localStorage.setItem("raglab.index", e.target.value); }}>{indexes.map((i) => <option key={i.id} value={i.id}>{i.name}</option>)}</select></label></>}
          <label className="f"><span>Score the top k</span><input type="number" value={k} onChange={(e) => setK(Number(e.target.value))} /></label>
          {param === "index" && <><label className="check small"><input type="checkbox" checked={sharedRules}
              onChange={(e) => setSharedRules(e.target.checked)} />Hold the first index's normalisation rules for all of them</label>
            <p className="small muted" style={{ marginTop: -4 }}>Off, each index is measured under its own rules, which is how
              it would actually be queried. On, only the index varies — useful when the indexes share a corpus and you want
              nothing else moving.</p></>}
          <button className="btn" disabled={!setId || active || !!badValues.length || (param === "index" ? picked.length < 2 : !values.trim())} onClick={start}>{active ? "Sweeping…" : "Run sweep"}</button>
          {Object.entries(jobs).filter(([, j]: Any) => j.status !== "done").map(([id, j]: Any) => <div key={id}>{j.status === "failed" || j.status === "cancelled" ? <div className={j.status === "cancelled" ? "note" : "err"}>{j.error}</div> :
            <><div className="row small"><span className="muted grow">{j.done}/{j.total} pipeline runs{j.partial ? `, ${j.partial} point${j.partial > 1 ? "s" : ""} saved` : ""}</span>
              <button className="link" onClick={() => api.post(`/api/sweeps/${id}/stop`).then(() => { load(); })}>Stop</button></div>
            <div className="progress"><i style={{ width: `${(100 * j.done) / Math.max(j.total, 1)}%` }} /></div></>}</div>)}
          {sweeps.length > 0 && <div><h4 style={{ marginBottom: 4 }}>Saved sweeps</h4>{sweeps.map((s) => <div className="row small" key={s.id} style={{ marginTop: 3 }}><label className="check small grow"><input type="radio" checked={open === s.id} onChange={() => setOpen(s.id)} />{s.label} <span className="muted">{s.points.length} points</span></label>
            <button className="link" onClick={async () => { await api.del(`/api/sweeps/${s.id}`); setOpen(""); load(); }}>Delete</button></div>)}</div>}
        </div>
        <div>{cur ? <>
          <div className="panel"><div className="row" style={{ marginBottom: 10 }}><h3 className="grow">{cur.label}</h3><label className="row small muted" style={{ gap: 6 }}>measured at<select style={{ width: 160 }} value={stage} onChange={(e) => setStage(e.target.value)}>{stages.map((s) => <option key={s} value={s}>{legacy ? s : stageLabel(s)}</option>)}</select></label></div>
            <p className="small muted" style={{ marginTop: -4, marginBottom: 8 }}>{STAGE_HELP[stage]}</p>
            <LineChart xs={cur.points.map((p: Any) => p.value)} xLabel={cur.label}
              right={{ label: `${legacy ? "ms" : "pipeline ms"} / question (dashed)`,
                       ys: cur.points.map((p: Any) => p.metrics.pipeline_ms ?? p.metrics.latency_ms) }}
              series={[{ label: seriesLabel("recall", stage, cur.spec.k, first), color: "var(--dense)", ys: cur.points.map((p: Any) => st(p, stage, "recall")) },
                       { label: seriesLabel("mrr", stage, cur.spec.k, first), color: "var(--bm25)", ys: cur.points.map((p: Any) => st(p, stage, "mrr")) },
                       { label: seriesLabel("ndcg", stage, cur.spec.k, first), color: "var(--ok)", ys: cur.points.map((p: Any) => st(p, stage, "ndcg")) },
                       ...(showHnsw ? [{ label: METRIC_SHORT.hnsw_recall, color: "var(--graph)", ys: cur.points.map((p: Any) => p.metrics.hnsw_recall ?? null) }] : [])]} />
            <div className="small muted" style={{ marginTop: 6 }}>† best query variant for that channel, picked with the gold label — what it could reach, not what was delivered.
              {!legacy && <> &nbsp; * scored over the whole prompt rather than the top k.</>}
              &nbsp; The ranking score is position quality times coverage, between 0 and 1 — not a textbook nDCG.</div>
            {cur.manifest?.normalization?.uniform === false && <div className="note small" style={{ marginTop: 6 }}>
              The indexes compared here were measured under different normalisation rules, so the curve varies the rules
              as well as the index. Re-run with the rules held fixed to isolate the index.</div>}
            {legacy && <div className="err small" style={{ marginTop: 6 }}>This sweep was measured before the current
              definitions: <code>context</code> recall was cut at the top k rather than covering the whole prompt, and the
              latency includes the cost of verifying the approximation. Its numbers are shown as recorded and are not
              comparable with a current sweep — re-run it to compare like with like.</div>}
            <div className="legend" style={{ marginTop: 6 }}>{[[METRIC_SHORT.recall, "var(--dense)"], [METRIC_SHORT.mrr, "var(--bm25)"], [METRIC_SHORT.ndcg, "var(--ok)"], ...(showHnsw ? [[METRIC_SHORT.hnsw_recall, "var(--graph)"]] : [])].map(([l, c]) => <span key={l}><svg width="14" height="6"><rect width="14" height="3" y="1.5" fill={c} /></svg>{l}</span>)}<span><svg width="14" height="6"><line x1="0" x2="14" y1="3" y2="3" stroke="var(--muted)" strokeDasharray="3 2" /></svg>latency</span></div></div>
          <div className="panel"><h3 style={{ marginBottom: 8 }}>Every point</h3><div style={{ overflowX: "auto" }}><table><thead><tr><th>{cur.label}</th>{stages.map((s) => <th className="num" key={s} style={{ color: CHANNEL[s] }} title={STAGE_HELP[s]}>{s}{DIAGNOSTIC.has(s) ? "†" : !legacy && WHOLE_PROMPT.has(s) ? "*" : ""}<br />recall / MRR</th>)}<th className="num" title={HNSW_RECALL_HELP}>{METRIC_SHORT.hnsw_recall}</th><th className="num">abstain on<br />unanswerable</th><th className="num">tokens</th><th className="num">cost</th><th className="num">ms</th></tr></thead>
            <tbody>{cur.points.map((p: Any) => <tr key={String(p.value)}><td className="mono">{String(p.value)}
              {p.degenerate && <div className="small muted" title="Below dense k, so the search ran wider than asked">ran at {String(p.effective)}</div>}</td>{stages.map((s) => <td className="num" key={s}>{st(p, s, "recall") === null ? "–" : `${st(p, s, "recall").toFixed(2)} / ${st(p, s, "mrr").toFixed(2)}`}</td>)}
              <td className="num">{hnswRecallCell(p.metrics)}</td><td className="num">{p.metrics.abstention == null ? "–" : `${Math.round(p.metrics.abstention * 100)}%`}</td><td className="num">{p.metrics.tokens}</td><td className="num">{p.metrics.cost ? `$${p.metrics.cost.toFixed(4)}` : "–"}</td><td className="num">{p.metrics.latency_ms}</td></tr>)}</tbody></table></div></div></>
          : <div className="panel"><h3>No sweep selected</h3><p className="muted" style={{ marginTop: 6 }}>Two curves worth running first: <b>HNSW ef_search</b> for recall against latency, and <b>Index</b> across several chunk sizes for the chunk-size curve.</p></div>}</div>
      </div>
    </>
  );
}
