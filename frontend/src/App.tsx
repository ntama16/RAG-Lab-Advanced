import { useEffect, useRef, useState } from "react";
import { api, Any } from "./api";
import Models from "./pages/Models";
import Corpus from "./pages/Corpus";
import Indexes from "./pages/Indexes";
import Workbench from "./pages/Workbench";
import Evaluate from "./pages/Evaluate";
import VectorSpace from "./pages/VectorSpace";
import NormalisationLab from "./pages/NormalisationLab";
import Compare from "./pages/Compare";
import Sweeps from "./pages/Sweeps";
import { Atlas, Guide } from "./pages/Learn";

const PHASES: [string, string[]][] = [["SETUP", ["Models"]], ["BUILD", ["Corpus", "Chunking", "Vector space"]], ["RUN", ["Normalisation lab", "Workbench", "Compare"]],
  ["EVALUATE", ["Evaluation", "Sweeps"]], ["LEARN", ["Technique atlas", "Guide"]]];
const TABS = ["Models", "Corpus", "Chunking", "Vector space", "Normalisation lab", "Workbench", "Compare", "Evaluation", "Sweeps", "Technique atlas", "Guide"] as const;
type Tab = (typeof TABS)[number];

function load<T>(key: string, fallback: T): T {
  try { const raw = localStorage.getItem(key); return raw ? JSON.parse(raw) : fallback; } catch { return fallback; }
}

export default function App() {
  const initial = (decodeURIComponent(location.hash.slice(1)) as Tab) || "Workbench";
  const [theme, setTheme] = useState<string>(load("raglab.theme", ""));
  useEffect(() => { if (theme) document.documentElement.dataset.theme = theme; localStorage.setItem("raglab.theme", JSON.stringify(theme)); }, [theme]);
  const [tab, setTab] = useState<Tab>(TABS.includes(initial) ? initial : "Workbench");
  const [defaults, setDefaults] = useState<Any>(null);
  const [pipeline, setPipelineState] = useState<Any>(null);
  const [models, setModels] = useState<Any>(null);

  const [bootErr, setBootErr] = useState("");
  const [booting, setBooting] = useState(true);
  const boot = () => {
    setBootErr(""); setBooting(true);
    api.get("/api/defaults").then((d) => {
      setDefaults(d);
      const saved = load<Any>("raglab.pipeline", null);
      setPipelineState(saved && Object.keys(d.pipeline).every((k) => k in saved) ? saved : d.pipeline);
    })
      // Without this the app sat on "Connecting…" for ever with nothing to act on.
      .catch((e: Error) => setBootErr(e.message || "The lab server did not answer."))
      .finally(() => setBooting(false));
    api.get("/api/models").then(setModels).catch(() => setModels(null));
  };
  useEffect(boot, []); // eslint-disable-line
  useEffect(() => { location.hash = tab; }, [tab]);
  // The nav wraps rather than scrolling sideways, so its height depends on the window. Anything
  // sticking below it needs the real number, not a constant that was right at one width.
  const nav = useRef<HTMLElement | null>(null);
  useEffect(() => {
    const el = nav.current;
    if (!el) return;
    // Border-box, because that is what a sticky offset has to clear. Republished on a resize and
    // once the fonts land as well as from the observer: a web font arriving changes the height
    // after the observer has already reported, and the sidebar would stick at the old number.
    const publish = () => document.documentElement.style.setProperty(
      "--nav-h", `${Math.round(el.getBoundingClientRect().height)}px`);
    publish();
    const ro = typeof ResizeObserver !== "undefined" ? new ResizeObserver(publish) : null;
    ro?.observe(el, { box: "border-box" });
    window.addEventListener("resize", publish);
    const frame = requestAnimationFrame(publish);
    document.fonts?.ready.then(publish).catch(() => undefined);
    return () => { ro?.disconnect(); window.removeEventListener("resize", publish); cancelAnimationFrame(frame); };
  }, [defaults, pipeline, models]);

  const setPipeline = (p: Any) => { setPipelineState(p); localStorage.setItem("raglab.pipeline", JSON.stringify(p)); };
  if (!defaults || !pipeline) return (
    <div className="page">
      {booting && !bootErr ? <span className="muted">Connecting to the lab server…</span> : (
        <div className="panel">
          <h3>Cannot reach the lab server</h3>
          <p className="muted" style={{ marginTop: 6 }}>{bootErr || "The server did not answer."}</p>
          <p className="small muted" style={{ marginTop: 10 }}>Start it with <code>raglab</code> (or
            <code> uvicorn raglab.main:app --port 8600</code>), then retry.</p>
          <div className="row" style={{ marginTop: 12 }}><button className="btn" onClick={boot}>Retry</button></div>
        </div>)}
    </div>
  );

  const m = models?.models;
  return (
    <>
      <nav className="top" ref={nav}>
        <div className="brand">
          <i aria-hidden><b style={{ background: "var(--bm25)" }} /><b style={{ background: "var(--dense)" }} /><b style={{ background: "var(--graph)" }} /></i>
          RAG Lab Advanced
        </div>
        <div className="phases">
          {PHASES.map(([phase, tabs]) => <div key={phase} className={`phase ${tabs.includes(tab) ? "here" : ""}`}><span>{phase}</span>
            <div className="tabs">{tabs.map((t) => <button key={t} aria-current={tab === t ? "page" : undefined} onClick={() => setTab(t as Tab)}>{t}</button>)}</div></div>)}
        </div>
        <div className="status">
          {models?.mock ? <span className="tag bad">mock models: for tests only</span>
            : m && <span className="roles" title={`embedding ${m.embedding.model} · generator ${m.generator.model}`}>
                embedding <code>{m.embedding.model}</code>&ensp;generator <code>{m.generator.model}</code></span>}
          <button className="themebtn" aria-label="Switch between light and dark" onClick={() => setTheme((document.documentElement.dataset.theme ?? (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light")) === "dark" ? "light" : "dark")}>◐</button>
        </div>
      </nav>
      <main className="page">
        {tab === "Workbench" && <Workbench pipeline={pipeline} setPipeline={setPipeline} defaults={defaults.pipeline} bounds={defaults.bounds ?? {}} presets={defaults.presets ?? []} goto={setTab as Any} />}
        {tab === "Chunking" && <Indexes defaults={defaults.index} />}
        {tab === "Vector space" && <VectorSpace />}
        {tab === "Corpus" && <Corpus />}
        {tab === "Evaluation" && <Evaluate pipeline={pipeline} defaults={defaults.pipeline} />}
        {tab === "Models" && <Models onSaved={setModels} />}
        {tab === "Normalisation lab" && <NormalisationLab />}
        {tab === "Compare" && <Compare pipeline={pipeline} />}
        {tab === "Sweeps" && <Sweeps pipeline={pipeline} />}
        {tab === "Technique atlas" && <Atlas pipeline={pipeline} setPipeline={setPipeline} defaults={defaults.pipeline} goto={setTab as Any} />}
        {tab === "Guide" && <Guide goto={setTab as Any} />}
      </main>
    </>
  );
}
