import { Any } from "../api";

type T = { name: string; prevents: string; how: string; where: string; apply?: Any; build?: string };
const GROUPS: { stage: string; blurb: string; items: T[] }[] = [
  { stage: "Prepare (index time)", blurb: "Retrieval quality is decided before any query arrives.", items: [
    { name: "Structure-aware chunking", prevents: "a sentence split from the heading that gives it meaning", how: "Markdown strategy cuts on headings and keeps the heading path with each chunk; recursive falls back paragraph → sentence → word.", where: "chunking.py", build: "Chunking: markdown or recursive" },
    { name: "Semantic chunking", prevents: "size-based cuts landing mid-topic", how: "Embeds a window round every sentence with your real embedding model and cuts where neighbouring windows drift apart (percentile threshold). The preview plots the distance curve.", where: "chunking.py · semantic_breakpoints", build: "Chunking: semantic" },
    { name: "Parent-child retrieval", prevents: "small chunks that match well but read badly", how: "Index small children for precise matching; hand the model the larger parent passage, de-duplicated.", where: "chunking.py · pipeline.py (context)", build: "Parent-child", apply: { context: { expand_parents: true } } },
    { name: "Contextual retrieval", prevents: "orphan chunks that never name their subject", how: "The router model writes one or two sentences situating each chunk in its document; that context is embedded and BM25-indexed with the chunk. Cached, so rebuilds are free.", where: "indexer.py · prompts.CONTEXTUALIZE", build: "Contextual retrieval" },
    { name: "Near-duplicate removal", prevents: "the same paragraph pasted into two documents crowding out other evidence", how: "Six-word shingles; a chunk is dropped when its Jaccard overlap with a chunk of another document exceeds 0.5.", where: "indexer.py · dedupe", build: "Near-duplicate removal" },
    { name: "HNSW index", prevents: "exact search cost growing linearly with the corpus", how: "hnswlib graph with M and ef_construction at build time and ef_search at query time. Exact search runs alongside so every trace reports the recall you paid.", where: "vectors.py", apply: { dense: { enabled: true, mode: "hnsw", compare_exact: true } } },
    { name: "Knowledge graph and community reports", prevents: "multi-hop questions and whole-corpus questions that no single passage answers", how: "LLM entity and relation extraction per chunk; label-propagation communities, each summarised by the model and embedded for retrieval.", where: "graph.py", build: "Knowledge graph (+ community reports)" },
  ] },
  { stage: "Understand the query", blurb: "Meet the corpus in its own words, without losing the user's.", items: [
    { name: "Semantic cache", prevents: "paying the whole pipeline again for a question already answered", how: "Cosine between query embeddings, per index and per exact configuration. Evaluations always bypass it.", where: "pipeline.py (cache)", apply: { cache: { enabled: true } } },
    { name: "Conversational rewriting", prevents: "“is it free?” being searched literally", how: "In follow-up mode the router rewrites each message into a standalone question using the last few turns.", where: "prompts.CONDENSE", apply: { condense: { enabled: true } } },
    { name: "Query normalisation", prevents: "“three weeks” missing a policy written as “21 days”; “Ridgeline” missing “ZX-481”", how: "Alias protection, acronym expansion, corpus terminology, duration canonicalisation, spelling against the index vocabulary. Rules are data, and the model can propose them from the corpus.", where: "normalize.py", apply: { normalize: { enabled: true } } },
    { name: "Routing and intent gate", prevents: "retrieving for “thanks”, or treating a part-number lookup like an essay question", how: "Router model classifies chitchat, lookup, conceptual, multi-hop or global and re-weights or switches stages accordingly.", where: "prompts.ROUTE", apply: { route: { enabled: true, apply: true } } },
    { name: "Self-query metadata filters", prevents: "“in the runbooks…” being matched as words instead of applied as a filter", how: "Router model extracts filters against the index's real metadata schema; unknown fields are rejected and shown. Filters apply inside HNSW search.", where: "pipeline.py (self_query)", apply: { self_query: { enabled: true } } },
    { name: "Decomposition and step-back", prevents: "a compound question whose halves live in two documents retrieving neither well", how: "Sub-questions and one more general question each retrieve independently and meet again in fusion.", where: "prompts.DECOMPOSE · STEP_BACK", apply: { decompose: { enabled: true }, step_back: { enabled: true } } },
    { name: "Multi-query and HyDE", prevents: "one phrasing of the question missing documents that phrase it differently", how: "Variants widen both BM25 and dense recall. HyDE writes a hypothetical answer and embeds it as a document, moving the query into document space.", where: "prompts.MULTI_QUERY · HYDE", apply: { multi_query: { enabled: true }, hyde: { enabled: true } } },
  ] },
  { stage: "Retrieve and fuse", blurb: "Signals with opposite blind spots, merged by rank because their scores share no scale.", items: [
    { name: "Hybrid search", prevents: "lexical search missing paraphrases; dense search missing exact identifiers", how: "Okapi BM25 beside cosine over real embeddings, plus personalised PageRank over the entity graph.", where: "bm25.py · vectors.py · graph.py", apply: { bm25: { enabled: true }, dense: { enabled: true } } },
    { name: "Freshness filter", prevents: "a superseded policy outranking the current one because it is shorter and denser", how: "Documents whose front matter carries superseded_by are removed before fusion, and the trace lists what was removed.", where: "pipeline.py (_retrieve)", apply: { freshness: { enabled: true } } },
    { name: "Reciprocal rank fusion", prevents: "BM25's unbounded scores drowning cosine's 0–1 range", how: "score = Σ weight / (k + rank). The rank-flow chart follows each passage from every list to its fused place.", where: "pipeline.py · fuse", apply: { fusion: { method: "rrf" } } },
    { name: "Weighted score fusion", prevents: "having no dial when one signal is known to be stronger", how: "Min-max normalise per retriever, then a weighted sum.", where: "pipeline.py · fuse", apply: { fusion: { method: "weighted" } } },
  ] },
  { stage: "Rerank, correct, assemble", blurb: "Read the shortlist properly, then decide what the model actually sees.", items: [
    { name: "Cross-encoder reranking", prevents: "bi-encoder similarity ranking a merely on-topic passage above the one that answers", how: "Query and passage scored jointly by a cross-encoder or rerank API; LLM pointwise scoring as the fallback.", where: "providers.py · rerank", apply: { rerank: { enabled: true } } },
    { name: "Corrective RAG", prevents: "confidently answering from passages that do not contain the answer", how: "Each passage graded; weak evidence triggers a rewrite and a wider second retrieval; none makes the generator abstain.", where: "pipeline.py (crag)", apply: { crag: { enabled: true } } },
    { name: "Knowledge refinement", prevents: "relevant passages padding the context with irrelevant sentences", how: "Passages are cut into strips and the router keeps only those bearing on the question.", where: "prompts.REFINE", apply: { crag: { enabled: true, refine: true } } },
    { name: "MMR and ordering", prevents: "five near-identical passages, and the best evidence lost in the middle", how: "Maximal marginal relevance over real embeddings; optional best-at-both-ends order; a hard token budget.", where: "pipeline.py · mmr", apply: { context: { mmr: { enabled: true, lambda: 0.7 }, order: "edges" } } },
  ] },
  { stage: "Generate and verify", blurb: "An answer is a claim about its sources.", items: [
    { name: "Cited generation", prevents: "answers nobody can check", how: "Numbered passages, mandatory [n] citations, and the trace shows which supplied passages went unused.", where: "prompts.GENERATE_SYSTEM", apply: { generate: { enabled: true } } },
    { name: "Groundedness check", prevents: "a fluent sentence that no passage supports", how: "The answer is split into claims and each is checked against the supplied passages. Scored in evaluations too.", where: "prompts.VERIFY", apply: { verify: { enabled: true } } },
  ] },
];

function merge(p: Any, patch: Any): Any { const out = { ...p }; for (const [k, v] of Object.entries(patch)) out[k] = v && typeof v === "object" && !Array.isArray(v) ? merge(p[k] ?? {}, v) : v; return out; }

/**
 * Whether the pipeline already carries every setting a technique applies.
 *
 * Read from the configuration rather than remembered, because remembering was a second copy of
 * the same fact and the two drifted immediately. A single `done` name meant only one button could
 * say "Applied" at a time, so applying a second technique made the first claim it was off while
 * its settings were still in place — and switching a technique off in the Workbench left the atlas
 * still calling it applied.
 *
 * The match is exact, including values that are not just on/off. That is what it should mean:
 * `fusion.method` can only be `rrf` or `weighted`, so exactly one of those two rows is applied,
 * and a technique whose numbers have been tuned away from the ones it recommends is no longer the
 * thing this row describes. Pressing it again restores them.
 */
function isApplied(p: Any, patch: Any): boolean {
  return Object.entries(patch).every(([k, v]) =>
    v && typeof v === "object" && !Array.isArray(v) ? isApplied(p?.[k] ?? {}, v) : p?.[k] === v);
}

/**
 * Whether a technique has anything that can be switched off.
 *
 * A boolean flag can. A value that names a choice cannot: `fusion.method` has to be `rrf` or
 * `weighted` and `context.order` has to be `rank` or `edges`, so there is no off for them — the
 * other option is its own row, and pressing that row is how you leave this one.
 */
function hasSwitch(patch: Any): boolean {
  return Object.values(patch).some((v) =>
    v && typeof v === "object" && !Array.isArray(v) ? hasSwitch(v) : typeof v === "boolean");
}

/**
 * The pipeline with this technique switched off.
 *
 * Two rules, because the leaves are two different kinds of thing. A boolean is cleared outright
 * rather than reverted — roughly half of these techniques are on in the shipped defaults, so
 * "revert to default" would have put them straight back on and the button would once again have
 * looked like it did nothing. Anything else goes back to the default, so that a technique setting
 * `context.order: "edges"` leaves the ordering as it found it rather than as a residue nobody
 * asked to keep.
 */
function turnOff(p: Any, patch: Any, defaults: Any): Any {
  const out = { ...p };
  for (const [k, v] of Object.entries(patch)) {
    if (v && typeof v === "object" && !Array.isArray(v)) out[k] = turnOff(p?.[k] ?? {}, v, defaults?.[k] ?? {});
    else if (typeof v === "boolean") out[k] = false;
    else if (defaults?.[k] !== undefined) out[k] = defaults[k];
  }
  return out;
}

/** The Apply / Applied control for one technique, or the link to where it is chosen at build time. */
function ApplyCell({ t, pipeline, setPipeline, defaults, goto }:
                   { t: T; pipeline: Any; setPipeline: (p: Any) => void; defaults: Any; goto: (tab: string) => void }) {
  if (!t.apply) return <button className="link" onClick={() => goto("Chunking")}>{t.build}</button>;
  const on = isApplied(pipeline, t.apply);
  // Nothing to switch off, and it is on: a marker rather than a control that would do nothing.
  if (on && !hasSwitch(t.apply)) return <span className="tag">Applied</span>;
  return (
    <button className="btn ghost" aria-pressed={on}
            title={on ? "Switch this off in the Workbench pipeline" : "Switch this on in the Workbench pipeline"}
            onClick={() => setPipeline(on ? turnOff(pipeline, t.apply, defaults) : merge(pipeline, t.apply))}>
      {on ? "Applied" : "Apply"}</button>
  );
}

export function Atlas({ pipeline, setPipeline, defaults, goto }:
                      { pipeline: Any; setPipeline: (p: Any) => void; defaults: Any; goto: (t: string) => void }) {
  return (
    <>
      <div className="pagehead"><h2>Technique atlas</h2><p>Every technique in the lab: the failure it prevents, how it works here, and where the code lives. “Apply” switches it on in the Workbench's pipeline and “Applied” switches it back off; index-time techniques are chosen when you build an index.</p></div>
      {GROUPS.map((g) => <section className="panel" key={g.stage}><div className="row" style={{ alignItems: "baseline", marginBottom: 6 }}><h3 className="grow">{g.stage}</h3><span className="small muted">{g.blurb}</span></div>
        <table><tbody>{g.items.map((t) => <tr key={t.name}><td style={{ width: "19%" }}><b>{t.name}</b><div className="mono small muted">{t.where}</div></td>
          <td style={{ width: "28%" }}><span className="small muted">Prevents</span><div>{t.prevents}</div></td><td><span className="small muted">How</span><div>{t.how}</div></td>
          <td className="num" style={{ width: 130 }}>
            <ApplyCell t={t} pipeline={pipeline} setPipeline={setPipeline} defaults={defaults} goto={goto} />
          </td></tr>)}</tbody></table></section>)}
      <section className="panel"><h3>Deliberately not here yet</h3><p className="muted" style={{ marginTop: 6 }}>Late-interaction retrieval (ColBERT) needs per-token embeddings and learned sparse retrieval (SPLADE) needs a sparse encoder; neither is served by Ollama or the common embedding APIs, and this lab does not ship stand-ins. An agentic retrieve-reason loop is parked until the sweeps and compare tools have been used enough to evaluate it honestly.</p></section>
    </>
  );
}

const EXPERIMENTS: [string, string][] = [
  ["Why stale documents need explicit handling", "Load Halcyon, ask “Can I send back a ZX-481 after three weeks?” with the freshness filter off, then on. Compare shows the superseded v2 policy leaving the context and the answer changing from 30 days to 21."],
  ["What normalisation buys, and where", "In the Normalisation lab try “Is the Ridgeline batery covered by guarantee?”. Watch BM25's top five change. Then run the golden set with normalisation off and on: the gain shows at the bm25 stage and mostly washes out after dense fusion. That is the honest result."],
  ["The recall you pay for speed", "Sweeps → HNSW ef_search. On a small corpus recall is flat at 100%; rebuild HNSW with M = 4 on the Chunking tab and sweep again to see the curve appear."],
  ["The chunk-size curve", "Build the same corpus at 80, 160, 320 and 640 tokens, then Sweeps → Index. Small chunks rank well and read badly; turn parent-child on and compare again."],
  ["Does the reranker earn its latency?", "Evaluation: run with rerank off, then on. Read MRR at ‘fused’ against ‘reranked’, and the ms per question beside it."],
  ["Will it admit ignorance?", "Generate a question set with a share of unanswerable questions. With corrective RAG on, the abstention rate should be high and false abstention low. Turn it off and ask one in the Workbench."],
];

export function Guide({ goto }: { goto: (t: string) => void }) {
  const Step = ({ n, tab, title, text }: { n: number; tab: string; title: string; text: string }) => <div className="stage on" data-n={n}><div className="row"><b>{title}</b><button className="link small" onClick={() => goto(tab)}>Open {tab}</button></div><p className="muted" style={{ marginTop: 2 }}>{text}</p></div>;
  return (
    <>
      <div className="pagehead"><h2>Guide</h2><p>RAG Lab Advanced runs every technique against real models on your machine and shows each stage's inputs, outputs, prompts, tokens and timings.</p></div>
      <div className="grid2">
        <section className="panel"><h3 style={{ marginBottom: 12 }}>First fifteen minutes</h3>
          <Step n={1} tab="Models" title="Point the four roles at models" text="Embedding, reranker, router, generator. Save and check all four. Set prices for cloud models so cost shows up in traces and evaluations." />
          <Step n={2} tab="Corpus" title="Load Halcyon" text="25 documents written to expose failure modes: conflicting policy versions, product aliases, acronyms, lexical traps, near-duplicate SOPs. It also installs the matching normalisation rules." />
          <Step n={3} tab="Chunking" title="Build an index" text="Start with markdown chunking at 160 tokens. Open it to see the chunk map and how many near-duplicates were removed." />
          <Step n={4} tab="Vector space" title="Look at the space" text="Colour by doc_type. Drop in “broken on arrival” and see it land beside the DOA playbook." />
          <Step n={5} tab="Workbench" title="Ask the flagship question" text="“Can I send back a ZX-481 after three weeks?” Read the trace top to bottom, then save the configuration as a preset." />
          <Step n={6} tab="Compare" title="Change one thing" text="Turn the freshness filter off, and compare against your preset. The diff names the passages only one side delivered." />
          <Step n={7} tab="Evaluation" title="Measure, don't cherry-pick" text="Load the Halcyon golden set, run the frozen lexical baseline as A and your settings as B." />
          <Step n={8} tab="Sweeps" title="Draw a curve" text="One knob, one question set, one chart." />
        </section>
        <div>
          <section className="panel" style={{ marginTop: 0 }}><h3 style={{ marginBottom: 8 }}>Guided experiments</h3>{EXPERIMENTS.map(([t, x]) => <div key={t} style={{ borderTop: "1px solid var(--rule)", padding: "8px 0" }}><b>{t}</b><p className="muted" style={{ marginTop: 2 }}>{x}</p></div>)}</section>
          <section className="panel"><h3 style={{ marginBottom: 8 }}>Reading the numbers</h3>
            <p><b>Recall@k</b> is the share of expected documents (or evidence quotes) inside the top k. <b>MRR</b> rewards putting the first of them at the top. <b>nDCG</b> discounts gently by position. Each is reported after every stage, so you can see where evidence is won and lost.</p>
            <p><b>Abstention</b> is scored only on unanswerable questions; <b>false abstention</b> on the rest. <b>Groundedness</b> is the share of answer claims a checker found support for.</p>
            <p className="muted">Ten questions is a smoke test, not a benchmark: one question is ten points. Prefer hand-written questions that each name the trap they set; LLM-written “simple” questions are easy because they are written from the chunk that answers them, so mix in paraphrased, multi-document and unanswerable ones.</p></section>
          <section className="panel"><h3 style={{ marginBottom: 8 }}>Using your own documents</h3><p className="muted">Use headings. Put metadata in YAML front matter: <code>doc_type</code>, <code>year</code>, <code>team</code> become self-query filters; <code>doc_key</code> is what golden questions cite; <code>superseded_by</code> feeds the freshness filter. Then ask the router model to propose normalisation rules for your corpus in the Normalisation lab, and check them before saving.</p></section>
        </div>
      </div>
    </>
  );
}
