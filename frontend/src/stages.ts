/**
 * Pipeline stages as the evaluator measures them. Mirrors `STAGES`, `DIAGNOSTIC_STAGES` and
 * `WHOLE_PROMPT_STAGES` in raglab/evals.py; keep the two in step.
 *
 * Three groups, measured three different ways. Showing them in one funnel without saying so is how
 * a chart ends up implying a comparison that was never made:
 *
 *  - `bm25` / `dense` / `graph` — what each channel could reach, taking the best query variant for
 *    that channel as judged against the gold label. A diagnostic, not a pipeline result.
 *  - `fused` / `reranked` — one consistently defined ranked list, scored over its top k.
 *  - `context` / `delivered` — the prompt itself, scored whole. Not a candidate list: every passage
 *    in it went to the model, and it is already capped by `context.max_chunks`.
 */
export const STAGES: string[] = ["bm25", "dense", "graph", "fused", "reranked", "context", "delivered"];

export const DIAGNOSTIC = new Set<string>(["bm25", "dense", "graph"]);
export const WHOLE_PROMPT = new Set<string>(["context", "delivered"]);

export const STAGE_HELP: Record<string, string> = {
  bm25: "Diagnostic: the best query variant for this channel, chosen with the gold label. What lexical search could reach, not what the pipeline delivered.",
  dense: "Diagnostic: the best query variant for this channel, chosen with the gold label. What vector search could reach, not what the pipeline delivered.",
  graph: "Diagnostic: the best query variant for this channel, chosen with the gold label. What the graph walk could reach, not what the pipeline delivered.",
  fused: "One ranked list after fusion, scored over its top k.",
  reranked: "One ranked list after reranking, scored over its top k.",
  context: "The passages selected for the prompt, read as their original chunk text. Scored over the whole prompt, not the top k.",
  delivered: "The text the model was actually given, after parent expansion, knowledge refinement and community reports. Scored over the whole prompt, not the top k. Compare with `context` to see what expansion won and what refinement cut.",
};

/** Stage name with the marker that says how it was measured. */
export function stageLabel(stage: string): string {
  if (DIAGNOSTIC.has(stage)) return `${stage} (diagnostic)`;
  if (WHOLE_PROMPT.has(stage)) return `${stage} (whole prompt)`;
  return stage;
}

/**
 * Suffix for a metric card's title. `@k` is only honest where a top-k cut-off was applied; the
 * prompt stages have none, and labelling them `@k` invited a comparison across different cut-offs.
 */
export function cutoffLabel(stage: string, k: number): string {
  return WHOLE_PROMPT.has(stage) ? "over the whole prompt" : `@${k}`;
}

/** Metrics definitions this build understands. Mirrors METRICS_VERSION in raglab/evals.py. */
export const METRICS_VERSION = 2;

/**
 * Whether a saved run predates the current metric definitions.
 *
 * Such a run's `context` recall was cut at k and its abstention rate averaged questions that
 * nothing had decided. Labelling those numbers with today's semantics, or drawing them in the same
 * funnel as current ones, states something about them that is not true.
 */
export function isLegacy(metrics: { version?: number } | null | undefined): boolean {
  return (metrics?.version ?? 1) < METRICS_VERSION;
}

/** Cut-off suffix for a metric card, honest about runs measured under the older definitions. */
export function cutoffLabelFor(stage: string, k: number, metrics: { version?: number } | null | undefined): string {
  if (isLegacy(metrics)) return `@${k}`;
  return cutoffLabel(stage, k);
}

/**
 * The three scores, named once. Evaluation and Sweeps drew the same numbers under different names —
 * "RANKING SCORE" against "nDCG@k" — which reads as two different measurements.
 *
 * `ndcg` is deliberately not called nDCG in the UI: it is position quality times coverage, bounded
 * by 1, and borrowing a familiar name would imply a definition this does not use.
 */
export const METRIC_LABELS: Record<string, string> = {
  recall: "RECALL",
  mrr: "MRR",
  ndcg: "RANKING SCORE",
};

/** Short form for a chart legend, where the card titles do not fit. */
export const METRIC_SHORT: Record<string, string> = {
  recall: "recall",
  mrr: "MRR",
  ndcg: "ranking score",
};

/** A metric's chart-series label, carrying the cut-off that was actually applied. */
export function seriesLabel(metric: string, stage: string, k: number, metrics?: { version?: number } | null): string {
  return `${METRIC_SHORT[metric] ?? metric} ${cutoffLabelFor(stage, k, metrics)}`;
}
