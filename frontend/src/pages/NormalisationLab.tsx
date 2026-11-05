import { useEffect, useState } from "react";
import { api, Any , usePersisted } from "../api";

const TRY = ["Can I send back a ZX-481 after three weeks?", "Is the Ridgeline batery covered by guarantee?", "What is the ETA for an express order?", "My headlamp arrived broken and won't switch on", "Does the 481 survive being dropped in a lake?"];
const GROUPS: [string, string, string][] = [["aliases", "Aliases", "informal name → canonical identifier (appended and protected)"], ["abbreviations", "Abbreviations", "acronym → long form, keeping the acronym"], ["terminology", "Terminology", "user phrase → the corpus's own term (appended)"]];

export default function NormalisationLab() {
  const [indexes, setIndexes] = useState<Any[]>([]); const [indexId, setIndexId] = useState(localStorage.getItem("raglab.index") ?? "");
  const [rules, setRules] = useState<Any>(null); const [query, setQuery] = usePersisted("raglab.normalisation.query", TRY[0]); const [res, setRes] = useState<Any>(null);
  const [sugg, setSugg] = useState<Any>(null); const [busy, setBusy] = useState(""); const [err, setErr] = useState(""); const [dirty, setDirty] = useState(false);
  // Whether the rules on screen belong to this index or to every index. They used to be one global
  // row while this page asked you to pick an index, so tuning them for one corpus retuned the rest.
  const [scope, setScope] = useState<"shared" | "index">("shared");
  // Rule edits are only on the server once you press Save, and leaving this tab unmounts the
  // page, so unsaved edits are kept here as a draft and restored ahead of the server copy.
  const [draft, setDraft] = usePersisted<Any>("raglab.normalisation.draft", null);

  useEffect(() => { api.get("/api/indexes").then((ix: Any[]) => { const r = ix.filter((i) => i.status === "ready"); setIndexes(r); if (!r.some((i) => i.id === indexId) && r[0]) setIndexId(r[0].id); });
    }, []); // eslint-disable-line
  // Rules follow the selected index: its own set if it has one, otherwise the shared set. With no
  // index at all the shared set still loads — rules are editable before anything is built, and
  // making this conditional on an index turned the page blank on a fresh install.
  useEffect(() => {
    api.get(indexId ? `/api/normalization?index_id=${indexId}` : "/api/normalization").then((r) => {
      setScope(r.scope);
      if (draft) { setRules(draft); setDirty(true); } else { setRules(r.rules); setDirty(false); }
    }).catch((e) => setErr(String((e as Error).message))); }, [indexId]); // eslint-disable-line
  const test = async (q = query, rs = rules) => { if (!indexId || !rs) return; setErr(""); try { setRes(await api.post("/api/normalization/test", { index_id: indexId, query: q, rules: rs })); } catch (e) { setErr(String((e as Error).message)); } };
  useEffect(() => { if (rules && indexId) test(); }, [indexId, rules]); // eslint-disable-line

  // Never `return null`: a page that renders nothing is indistinguishable from a crash, which is
  // exactly how the last one went unnoticed.
  if (!rules) return <div className="panel">{err
    ? <><h3>The rules could not be loaded</h3><p className="err" style={{ marginTop: 8 }}>{err}</p></>
    : <p className="muted">Loading the normalisation rules…</p>}</div>;
  const edit = (g: string, k: string, v: string | null, newKey?: string) => { const group = { ...rules[g] }; if (v === null) delete group[k]; else { if (newKey !== undefined && newKey !== k) delete group[k]; group[newKey ?? k] = v; } const next = { ...rules, [g]: group }; setRules(next); setDirty(true); setDraft(next); };
  const save = async (to: "shared" | "index" = scope) => {
    setBusy("save"); setErr("");
    try { const r = await api.put("/api/normalization", { rules, index_id: to === "index" ? indexId : null });
      setRules(r.rules); setScope(to === "index" ? "index" : r.scope); setDirty(false); setDraft(null); }
    catch (e) { setErr(String((e as Error).message)); }
    setBusy("");
  };
  const unscope = async () => { setBusy("save"); setErr("");
    try { const r = await api.del(`/api/normalization?index_id=${indexId}`); setRules(r.rules); setScope(r.scope); setDirty(false); setDraft(null); }
    catch (e) { setErr(String((e as Error).message)); }
    setBusy(""); };
  const changed = (a: Any[], b: Any[]) => b.map((h) => ({ ...h, was: a.findIndex((x) => x.i === h.i) + 1 }));

  return (
    <>
      <div className="pagehead"><h2>Normalisation lab</h2>
        <p>Meet the corpus in its own words without losing the user's. Every rule appends rather than substitutes, identifiers are protected, and each change is recorded. The rules here are what the Workbench's “Query normalisation” stage applies.</p></div>
      {err && <div className="err" style={{ marginBottom: 12 }}>{err}</div>}
      <div className="split" style={{ gridTemplateColumns: "minmax(0, 1fr) minmax(0, 1fr)" }}>
        <div className="stack">
          <div className="panel stack">
            <div className="row"><select style={{ width: 200 }} value={indexId} onChange={(e) => { setIndexId(e.target.value); localStorage.setItem("raglab.index", e.target.value); }} aria-label="Index">{indexes.map((i) => <option key={i.id} value={i.id}>{i.name}</option>)}</select>
              <input className="grow" type="text" value={query} onChange={(e) => setQuery(e.target.value)} onKeyDown={(e) => e.key === "Enter" && test()} aria-label="Query to normalise" /><button className="btn" onClick={() => test()}>Normalise</button></div>
            <div className="row small" style={{ gap: 6 }}><span className="muted">Try</span>{TRY.map((t) => <button key={t} className="link" onClick={() => { setQuery(t); test(t); }}>{t.slice(0, 34)}…</button>)}</div>
            {!indexes.length && <p className="muted small">Build an index first: spelling correction uses the index's vocabulary.</p>}
          </div>
          {res && <div className="panel stack">
            <div><div className="small muted">Before</div><div className="mono">{res.original}</div></div>
            <div><div className="small muted">After</div><div className="mono" style={{ color: "var(--ok)" }}>{res.normalized}</div></div>
            {res.steps.length ? <table><tbody>{res.steps.map((s: Any, n: number) => <tr key={n}><td style={{ width: 96 }}><span className="tag ok">{s.rule}</span></td><td className="mono small" style={{ width: "36%" }}>{s.from} → {s.to}</td><td className="small muted">{s.note}</td></tr>)}</tbody></table>
              : <p className="muted small">No rule fired. Either the query already uses the corpus's wording, or the rules do not cover it yet.</p>}
            <div className="grid2" style={{ gridTemplateColumns: "1fr 1fr" }}>
              {([["BM25 before", res.bm25_before.map((h: Any) => ({ ...h, was: 0 }))], ["BM25 after", changed(res.bm25_before, res.bm25_after)]] as [string, Any[]][]).map(([label, hits]) => <div className="list bm25" key={label}><div className="small muted" style={{ marginBottom: 4 }}>{label}</div>
                {hits.map((h, r) => <div className="hit" key={h.i}><span className="r">{r + 1}</span><div className="t"><b>{h.section?.split(" > ").pop() || h.title}</b>{label === "BM25 after" && (h.was === 0 ? <span className="tag ok" style={{ marginLeft: 4 }}>new</span> : h.was > r + 1 ? <span className="move up"> ▲{h.was - r - 1}</span> : null)}<span className="snip">{h.title}</span></div><span className="s">{h.score}</span></div>)}
                {!hits.length && <div className="small muted">No lexical match.</div>}</div>)}
            </div>
            <p className="small muted">Lexical search is where normalisation pays: dense retrieval already bridges most paraphrases. Spelling is checked against {res.vocabulary} corpus terms.</p>
          </div>}
        </div>
        <div className="panel stack">
          <div className="row"><h3 className="grow">Rules</h3>{dirty && <span className="tag partial">unsaved</span>}
            <span className={`tag ${scope === "index" ? "ok" : ""}`} title={scope === "index"
              ? "These rules apply only to the selected index" : "These rules apply to every index that has none of its own"}>
              {scope === "index" ? "this index only" : "shared by all indexes"}</span>
            <button className="btn" disabled={!dirty || busy === "save"} onClick={() => save()}>Save</button></div>
          <div className="row small">
            <button className="link" disabled={busy === "save"} onClick={() => save("index")}>Save for this index only</button>
            <button className="link" disabled={busy === "save"} onClick={() => save("shared")}>Save as the shared set</button>
            {scope === "index" && <button className="link" disabled={busy === "save"} onClick={unscope}>Follow the shared set again</button>}
          </div>
          <p className="small muted" style={{ marginTop: -4 }}>A corpus has its own vocabulary. Rules saved for one index
            leave the others alone; the shared set is what an index uses until you give it its own.</p>
          <div className="row small"><button className="link" onClick={async () => { const r = await api.post("/api/normalization/preset?name=halcyon"); setRules(r.rules); setDirty(false); setDraft(null); }}>Load the Halcyon rule set</button>
            <button className="link" onClick={async () => { const r = await api.post("/api/normalization/preset?name=empty"); setRules(r.rules); setDirty(false); setDraft(null); }}>Clear all</button>
            <button className="link" disabled={!indexId || busy === "suggest"} onClick={async () => { setBusy("suggest"); setErr(""); try { setSugg(await api.post(`/api/normalization/suggest?index_id=${indexId}`)); } catch (e) { setErr(String((e as Error).message)); } setBusy(""); }}>{busy === "suggest" ? "Reading the corpus…" : "Suggest rules from this corpus with the router model"}</button></div>
          <div className="row"><label className="check small"><input type="checkbox" checked={rules.dates} onChange={(e) => { setRules({ ...rules, dates: e.target.checked }); setDirty(true); }} />Canonicalise durations to days</label>
            <label className="check small"><input type="checkbox" checked={rules.spelling} onChange={(e) => { setRules({ ...rules, spelling: e.target.checked }); setDirty(true); }} />Correct spelling against the corpus vocabulary</label></div>
          {GROUPS.map(([g, title, help]) => <div key={g}><h4>{title} <span className="muted small" style={{ fontWeight: 400 }}>{help}</span></h4>
            {Object.entries(rules[g] as Record<string, string>).map(([k, v]) => <div className="row" key={k} style={{ marginTop: 4, flexWrap: "nowrap" }}><input type="text" style={{ width: "34%" }} defaultValue={k} onBlur={(e) => e.target.value !== k && edit(g, k, v, e.target.value.toLowerCase())} />
              <input type="text" className="grow" value={v} onChange={(e) => edit(g, k, e.target.value)} /><button className="link small" onClick={() => edit(g, k, null)}>Remove</button></div>)}
            <button className="link small" style={{ marginTop: 4 }} onClick={() => edit(g, "", "")}>Add a rule</button>
            {sugg && Object.keys(sugg[g]).filter((k) => !(k in rules[g])).length > 0 && <div className="note small" style={{ marginTop: 6 }}><b>Suggested by the model</b> (check before accepting)
              {Object.entries(sugg[g] as Record<string, string>).filter(([k]) => !(k in rules[g])).map(([k, v]) => <div className="row" key={k} style={{ marginTop: 3 }}><span className="mono grow">{k} → {v}</span><button className="link" onClick={() => edit(g, k, v)}>Accept</button></div>)}</div>}
          </div>)}
        </div>
      </div>
    </>
  );
}
