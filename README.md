# RAG Lab Advanced

A local lab for advanced RAG work. Every technique runs for real against real models (local Ollama,
any OpenAI-compatible server, Anthropic, or a rerank API), and every stage shows its inputs, outputs,
timings, prompts and raw model responses.

<p align="center">
  <a href="https://ntama16.github.io/raglab-tour/"><img src="https://ntama16.github.io/raglab-tour/poster.jpg" alt="RAG Lab Advanced: a 70-second tour" width="100%"></a>
  <br><sub>A 70-second tour, from model setup to comparing two pipeline runs. Click to watch it on the tour page.</sub>
</p>

## Run it

```bash
git clone https://github.com/ntama16/RAG-Lab-Advanced.git && cd RAG-Lab-Advanced
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
ollama pull nomic-embed-text && ollama pull llama3.1:8b     # or point the roles at cloud models
raglab                                                       # http://127.0.0.1:8600
```

The UI is pre-built into `raglab/static`, so Node is only needed if you change the frontend.
The navigation follows the work: **Setup** (Models) → **Build** (Corpus, Chunking, Vector space) → **Run** (Normalisation lab, Workbench, Compare)
→ **Evaluate** (Evaluation, Sweeps) → **Learn** (Technique atlas, Guide). The in-app Guide has a fifteen-minute first run and six guided experiments.

Two sample corpora ship with it: **Halcyon Trailworks** (25 documents with a 10-question golden set and
normalisation rules) and **Kestrel** (8 cross-referencing engineering docs, good for graph and multi-hop).

No model server handy? `RAGLAB_MOCK=1 raglab` runs the whole app on deterministic stand-ins. That mode exists
for tests and UI work only; its results mean nothing.

It is a single-user tool with no authentication, so by default it listens on `127.0.0.1` only. Do not start it with
`--host 0.0.0.0` on a network you do not control: anyone who can reach it can read your corpus and spend
your model credits. API keys typed into the Models page are stored in plain text in `raglab.config.json`;
prefer the "API key env var" field. See [SECURITY.md](SECURITY.md).

## A tour

**Setup and build.** Point four model roles at local or cloud endpoints, load a corpus, build indexes several ways, and look at the space they make.

<table>
<tr>
<td width="50%"><img src="https://ntama16.github.io/raglab-tour/screens/01-models-four-roles.jpg" alt="Models page with embedding, reranker, router and generator roles"><br><sub><b>Models.</b> Embedding, reranker, router and generator, each on Ollama, an OpenAI-compatible server, Anthropic or a cross-encoder.</sub></td>
<td width="50%"><img src="https://ntama16.github.io/raglab-tour/screens/02-corpus-documents-metadata.jpg" alt="Corpus page listing documents with metadata"><br><sub><b>Corpus.</b> Front matter becomes filterable metadata. Halcyon and Kestrel load with one click.</sub></td>
</tr>
<tr>
<td><img src="https://ntama16.github.io/raglab-tour/screens/03-chunking-build-indexes.jpg" alt="Chunking page with index build options and built indexes"><br><sub><b>Chunking.</b> Each build is its own index: strategy, size, parent-child, contextual retrieval, dedupe, graph.</sub></td>
<td><img src="https://ntama16.github.io/raglab-tour/screens/04-vector-space-pca-map.jpg" alt="Vector space PCA map of every chunk"><br><sub><b>Vector space.</b> Every chunk on a PCA or UMAP map. Drop in a phrase to see where a query lands.</sub></td>
</tr>
</table>

**Run.** Rewrite the query in the corpus's own words, then trace one question through every stage of the pipeline.

<table>
<tr>
<td width="50%"><img src="https://ntama16.github.io/raglab-tour/screens/05-normalisation-lab-rules-bm25.jpg" alt="Normalisation lab showing rules and BM25 results before and after"><br><sub><b>Normalisation lab.</b> Aliases, acronyms, terminology and durations, with BM25 results before and after.</sub></td>
<td width="50%"><img src="https://ntama16.github.io/raglab-tour/screens/06-workbench-pipeline-timing.jpg" alt="Workbench with the pipeline settings, answer and timing bar"><br><sub><b>Workbench.</b> Twelve switchable stages, the answer, and where every millisecond and token went.</sub></td>
</tr>
<tr>
<td><img src="https://ntama16.github.io/raglab-tour/screens/07-workbench-query-variants-retrieval.jpg" alt="Query variants and dense and BM25 result lists"><br><sub><b>Retrieval.</b> Multi-query variants, then dense HNSW and BM25 lists side by side, with HNSW recall against exact search.</sub></td>
<td><img src="https://ntama16.github.io/raglab-tour/screens/08-workbench-query-map-fusion.jpg" alt="Query on the vector map and a rank-flow chart for fusion"><br><sub><b>Fusion.</b> The query starred among its nearest chunks, and each passage's rank traced into the fused order.</sub></td>
</tr>
<tr>
<td><img src="https://ntama16.github.io/raglab-tour/screens/10-workbench-rerank-cross-encoder.jpg" alt="Rerank stage showing rank movement"><br><sub><b>Rerank.</b> A cross-encoder reorders the candidates, and every move up or down is shown.</sub></td>
<td><img src="https://ntama16.github.io/raglab-tour/screens/11-workbench-corrective-rag-grading.jpg" alt="Corrective RAG grading each passage"><br><sub><b>Corrective RAG.</b> Each passage graded relevant, partial or irrelevant before the context is assembled.</sub></td>
</tr>
<tr>
<td colspan="2"><img src="https://ntama16.github.io/raglab-tour/screens/12-workbench-context-generation-groundedness.jpg" alt="Context assembly, generation and groundedness check"><br><sub><b>Context, answer, groundedness.</b> Context packed to a token budget, a cited answer, and each claim checked against the passages it was given.</sub></td>
</tr>
</table>

**Compare.** One question under two configurations.

<table>
<tr>
<td width="50%"><img src="https://ntama16.github.io/raglab-tour/screens/13-compare-two-runs.jpg" alt="Compare page showing two answers and which passages reached the model"><br><sub><b>Two runs.</b> The settings that differed, both answers, and which passages reached the model on each side.</sub></td>
<td width="50%"><img src="https://ntama16.github.io/raglab-tour/screens/14-compare-stage-by-stage.jpg" alt="Stage-by-stage time and token comparison"><br><sub><b>Stage by stage.</b> Time and tokens for every stage of both runs, and what differed at each.</sub></td>
</tr>
</table>

## What is in it

| Stage | What actually runs |
|---|---|
| Chunking | fixed, sentence, recursive, markdown-heading aware, and **semantic** (embeds a window round every sentence, cuts at a percentile of neighbour distance, with a preview that plots the distances). Optional **parent-child** and **contextual retrieval** (LLM situates each chunk before it is embedded and BM25-indexed). |
| Dense index | **HNSW** via hnswlib with `M`, `ef_construction`, `ef_search` exposed. Exact search runs alongside, so every trace reports HNSW recall against ground truth. Rebuild the graph with new parameters from stored vectors, no re-embedding. Metadata filters are applied inside the HNSW search. |
| Lexical index | Okapi BM25 on an inverted index; `k1` and `b` are query-time knobs; per-term idf shown. |
| Near-duplicate removal | Six-word shingles, Jaccard > 0.5 across documents; removed chunks are listed. |
| Graph index | LLM entity and relation extraction per chunk, entity linking at query time (exact, substring, then embedding similarity), **personalised PageRank** from the linked entities to score chunks. Optional label-propagation **communities**, each summarised by the model and embedded. |
| Semantic cache | Query-embedding cosine per index and per exact configuration. Evaluations and sweeps always bypass it. |
| Conversational rewriting | Follow-up mode: the router rewrites each message into a standalone question from the last few turns. |
| Query normalisation | Alias protection, acronym expansion, corpus terminology, duration canonicalisation, spelling against the index vocabulary; all append-only, all traced. Rules are editable data, and the router model can propose rules from your corpus (you accept them one by one). |
| Route / intent gate | Router model classifies chitchat, lookup, conceptual, multi-hop and in/out of scope; can skip retrieval or re-weight the pipeline. |
| Self-query | Router model extracts metadata filters against the index's real schema (front matter plus optional LLM-extracted document metadata); unknown fields are rejected and shown. |
| Query transformation | **Decomposition** into sub-questions, **step-back** question, multi-query variants and **HyDE** (hypothetical passage, embedded as a document). |
| Freshness filter | Documents whose front matter has `superseded_by` are removed before fusion; the trace lists what was removed and why. |
| Fusion | Reciprocal rank fusion or weighted min-max scores, with per-retriever contributions per passage. |
| Rerank | Cross-encoder in process, any Cohere-style `/rerank` endpoint (llama.cpp, vLLM, infinity, Jina, Cohere, Voyage), TEI, or pointwise LLM scoring as a fallback. Rank movement shown. |
| Corrective RAG | Each passage graded relevant / partial / irrelevant; weak evidence triggers a query rewrite and a wider second retrieval; no evidence makes the generator abstain. Local-only, so the corrective action is re-retrieval rather than web search. Optional **knowledge refinement**: passages are cut into strips and only the relevant ones are kept. |
| Context assembly | Parent expansion, MMR, token budget, optional best-at-both-ends ordering, optional **graph community reports** for whole-corpus questions. |
| Generation | Streamed, cited answer; cited vs. supplied-but-unused passages; full prompt visible. |
| Groundedness check | The answer is split into claims and each is checked against the supplied passages. |
| Evaluation | Gold = documents a human would cite (hand-written golden sets with `probes`) or verbatim evidence quotes, so one set scores every index. Generator mixes simple, **paraphrased**, **multi-document** and **unanswerable** questions. Recall@k / MRR / nDCG **per stage**, abstention and false abstention, groundedness, tokens, cost, latency, by question type. A/B cards, stage funnel, one-click frozen lexical baseline. |

## Experiment tooling

- **Compare**: one question under two presets (or two indexes): settings diff, answers, a rank-flow of which passages reached the model on each side, and per-stage time and tokens.
- **Sweeps**: one knob across values (`ef_search`, RRF k, rerank top-n, BM25 k1/b, fusion weights, context size…) or across **indexes** for index-time choices such as chunk size; metric and latency curves, saved with their settings.
- **Presets**: named pipeline configurations, stored server-side, exportable and importable as JSON.
- **Token and cost accounting**: real usage from Ollama, OpenAI-compatible and Anthropic responses (estimated otherwise), per call, per stage, per run; set `price_in` / `price_out` per role.
- **Robustness**: per-role `timeout` and `retries` with backoff; index-time model calls (chunk contexts, graph extraction, community reports, document metadata) are cached by prompt, so rebuilds are free.

## What you can see

- **Chunking**: a chunk map (one bar per document, one block per chunk, sized by tokens), size histogram, and "inside a chunk":
  source text beside the chunks it became, hover-linked, with overlap highlighted. Semantic chunking plots its distance curve and threshold.
- **Vector space**: every chunk on a zoomable PCA map (UMAP with `pip install -e '.[umap]'`), coloured and shaped by document or any
  metadata field. Drop in a phrase to see where a query lands; pin a chunk to list its exact cosine neighbours.
- **Workbench**: score bars on every retrieval list, the query starred on the vector map with links to its nearest chunks, the graph walk
  drawn as a node-link view, a rank-flow chart for fusion (each passage's rank per retriever → fused rank) and again for reranking,
  grade distribution for corrective RAG, the context token budget as a stacked bar, and a latency waterfall across stages.
- **Evaluation**: A vs B metric cards with bars and point deltas at any stage, a per-stage hit-rate funnel showing where evidence is
  won and lost, a per-question rank table, and a one-click frozen lexical baseline.

## Stack, and why

- **Python + FastAPI backend.** hnswlib, cross-encoders and numpy are Python-native; the async model suits fanning
  out many model calls (grading, graph extraction) and streaming traces over SSE.
- **No LangChain / LlamaIndex.** In a lab the algorithm is the thing being studied. Each technique is a short,
  readable module (`chunking.py`, `bm25.py`, `vectors.py`, `graph.py`, `pipeline.py`) and all prompts live in `prompts.py`.
- **Thin provider layer over HTTP** (`providers.py`) rather than vendor SDKs: four roles (embedding, reranker,
  router, generator), each independently local or cloud.
- **SQLite + files** for persistence. Embeddings are stored, so indexes reload instantly and HNSW can be rebuilt freely.
  Single user, local only: a vector database server would add nothing here.
- **Vite + React + TypeScript UI**, built into the Python package and served by the Python process. One command, one port.

## Layout

```
raglab/            Python package (API, pipeline, indexes, providers, built UI in static/)
raglab/sample_corpus/  the Halcyon and Kestrel sample corpora, shipped inside the package
frontend/          UI source.  npm install && npm run dev  (proxies /api to :8600);  npm run build
tests/             end-to-end test on the mock provider:  pip install -e '.[dev]' && pytest
```

Settings live in `raglab.config.json` (see the example file); data in `./data`. Override with `RAGLAB_CONFIG` and `RAGLAB_DATA`.

## Parked, and why

- **ColBERT / SPLADE**: need per-token embeddings or a sparse encoder, which Ollama and the common embedding APIs do not serve. No stand-ins here; they would arrive as optional in-process providers.
- **Agentic retrieve-reason loop**: hard to judge without the compare and sweep tools being exercised first. Corrective RAG already covers the bounded re-retrieve case.
- **Ingestion** (layout-aware PDF, DOCX/HTML, incremental indexing): not started yet.
- Out of scope by design: multi-tenancy, auth, non-text corpora.

## License

Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE). The sample corpora in `raglab/sample_corpus` are
fictional and are covered by the same licence.
