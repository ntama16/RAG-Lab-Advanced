/* Visual building blocks shared by every page: group colours and shapes, score bars, the vector map,
   the rank-flow chart, the stage waterfall and the sub-graph view. Plain SVG, no chart library. */
import { ReactNode, useMemo, useRef, useState } from "react";
import { Any } from "./api";

export const PALETTE = ["#3f82e6", "#dd5a2e", "#21a371", "#d4900f", "#d84b85", "#4c9a36", "#8664d8", "#1c9db0", "#a9743a", "#7c8aa0"];
const SHAPES = ["circle", "square", "diamond", "triangle", "plus", "hex"] as const;
export const CHANNEL: Record<string, string> = { bm25: "var(--bm25)", dense: "var(--dense)", graph: "var(--graph)", fused: "var(--ink)", reranked: "var(--ok)", context: "var(--ok)" };

export type Groups = { keys: string[]; of: (docId: string) => string; color: (key: string) => string; shape: (key: string) => number };

/** Group chunks by document, or by any document metadata field (doc_type, team, year...). */
export function makeGroups(docs: Any[], field: string): Groups {
  const label = (d: Any) => field === "document" ? d.title : String((Array.isArray(d.meta[field]) ? d.meta[field][0] : d.meta[field]) ?? "none");
  const byDoc = new Map<string, string>(docs.map((d) => [d.id, label(d)]));
  const keys = Array.from(new Set(docs.map(label)));
  const idx = (k: string) => Math.max(0, keys.indexOf(k));
  return { keys, of: (id) => byDoc.get(id) ?? "none", color: (k) => PALETTE[idx(k) % PALETTE.length], shape: (k) => idx(k) % SHAPES.length };
}

export function Marker({ x, y, r, shape, fill, opacity = 1, stroke }: { x: number; y: number; r: number; shape: number; fill: string; opacity?: number; stroke?: string }) {
  const p = { fill, opacity, stroke, strokeWidth: stroke ? 1.5 : 0 };
  switch (SHAPES[shape]) {
    case "square": return <rect x={x - r} y={y - r} width={2 * r} height={2 * r} {...p} />;
    case "diamond": return <path d={`M${x} ${y - r * 1.3}L${x + r * 1.3} ${y}L${x} ${y + r * 1.3}L${x - r * 1.3} ${y}Z`} {...p} />;
    case "triangle": return <path d={`M${x} ${y - r * 1.3}L${x + r * 1.25} ${y + r}L${x - r * 1.25} ${y + r}Z`} {...p} />;
    case "plus": { const a = r * 0.45, b = r * 1.25; return <path d={`M${x - a} ${y - b}h${2 * a}v${b - a}h${b - a}v${2 * a}h${-(b - a)}v${b - a}h${-2 * a}v${-(b - a)}h${-(b - a)}v${-2 * a}h${b - a}Z`} {...p} />; }
    case "hex": return <path d={[0, 1, 2, 3, 4, 5].map((k) => `${k ? "L" : "M"}${x + r * 1.2 * Math.cos((Math.PI / 3) * k)} ${y + r * 1.2 * Math.sin((Math.PI / 3) * k)}`).join("") + "Z"} {...p} />;
    default: return <circle cx={x} cy={y} r={r} {...p} />;
  }
}

export function Legend({ groups }: { groups: Groups }) {
  return <div className="legend">{groups.keys.slice(0, 14).map((k) => <span key={k}><svg width="14" height="14" aria-hidden><Marker x={7} y={7} r={4.2} shape={groups.shape(k)} fill={groups.color(k)} /></svg>{k}</span>)}
    {groups.keys.length > 14 && <span className="muted">+{groups.keys.length - 14} more</span>}</div>;
}

export function ScoreBar({ value, max, color = "var(--ink)" }: { value: number; max: number; color?: string }) {
  return <div className="sbar"><i style={{ width: `${Math.max(2, Math.min(100, (100 * value) / (max || 1)))}%`, background: color }} /></div>;
}

export function Tiles({ items }: { items: [string | number, string][] }) {
  return <div className="tiles">{items.map(([v, l]) => <div key={l}><b>{v}</b><span>{l}</span></div>)}</div>;
}

/* ------------------------------------------------------------------ vector map */
export type MapQuery = { label: string; kind: string; xy: number[]; links?: number[] };

export function VectorMap({ points, groups, hits, rings, queries = [], pinned, labelled, onPin, tip, height = 420 }: {
  points: Any[]; groups: Groups; hits?: Set<number>; rings?: Set<number>; queries?: MapQuery[]; pinned?: number | null;
  labelled?: Set<number>; onPin?: (i: number | null) => void; tip: (i: number) => { title: string; sub?: string; text: string }; height?: number;
}) {
  const W = 900, H = Math.round((height / 420) * 520);
  const [view, setView] = useState({ k: 1, x: 0, y: 0 });
  const [hover, setHover] = useState<number | null>(null);
  const drag = useRef<{ x: number; y: number; moved: boolean } | null>(null);
  const svg = useRef<SVGSVGElement>(null);

  const { X, Y, pos } = useMemo(() => {
    const all = [...points.map((p) => p.xy), ...queries.map((q) => q.xy)];
    const xs = all.map((p) => p[0]), ys = all.map((p) => p[1]);
    const [x0, x1, y0, y1] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
    const fx = (v: number) => 40 + ((v - x0) / (x1 - x0 || 1)) * (W - 80), fy = (v: number) => H - 40 - ((v - y0) / (y1 - y0 || 1)) * (H - 80);
    return { X: fx, Y: fy, pos: new Map<number, [number, number]>(points.map((p) => [p.i, [fx(p.xy[0]), fy(p.xy[1])]])) };
  }, [points, queries, H]);

  const wheel = (e: React.WheelEvent) => {
    const box = svg.current!.getBoundingClientRect();
    const mx = ((e.clientX - box.left) / box.width) * W, my = ((e.clientY - box.top) / box.height) * H;
    const k = Math.max(1, Math.min(12, view.k * (e.deltaY < 0 ? 1.18 : 1 / 1.18)));
    setView({ k, x: mx - ((mx - view.x) / view.k) * k, y: my - ((my - view.y) / view.k) * k });
  };
  const move = (e: React.PointerEvent) => {
    if (!drag.current) return;
    const box = svg.current!.getBoundingClientRect();
    const dx = ((e.clientX - drag.current.x) / box.width) * W, dy = ((e.clientY - drag.current.y) / box.height) * H;
    if (Math.abs(dx) + Math.abs(dy) > 2) drag.current.moved = true;
    drag.current.x = e.clientX; drag.current.y = e.clientY;
    setView((v) => ({ ...v, x: v.x + dx, y: v.y + dy }));
  };
  const s = 1 / view.k; // keep marker size constant on screen while zooming
  const focus = hits && hits.size > 0;
  const hp = hover !== null ? pos.get(hover) : null;
  const info = hover !== null ? tip(hover) : null;

  return (
    <div className="vmap" style={{ height }}>
      <svg ref={svg} viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="xMidYMid meet" role="img" aria-label="Two-dimensional map of chunk embeddings"
        onWheel={wheel} onPointerDown={(e) => { drag.current = { x: e.clientX, y: e.clientY, moved: false }; }} onPointerMove={move}
        onPointerUp={() => { if (drag.current && !drag.current.moved && hover === null) onPin?.(null); drag.current = null; }}
        onPointerLeave={() => { drag.current = null; }} onDoubleClick={() => setView({ k: 1, x: 0, y: 0 })}>
        <defs><pattern id="grid" width="90" height="65" patternUnits="userSpaceOnUse"><path d="M90 0H0V65" fill="none" stroke="var(--rule)" strokeWidth="0.6" opacity="0.6" /></pattern>
          <radialGradient id="glow"><stop offset="0" stopColor="#f2b63c" stopOpacity="0.45" /><stop offset="1" stopColor="#f2b63c" stopOpacity="0" /></radialGradient></defs>
        <rect width={W} height={H} fill="url(#grid)" />
        <g transform={`translate(${view.x} ${view.y}) scale(${view.k})`}>
          {queries.flatMap((q, n) => (q.links ?? []).map((i) => { const p = pos.get(i); return p && <line key={`${n}-${i}`} x1={X(q.xy[0])} y1={Y(q.xy[1])} x2={p[0]} y2={p[1]} stroke="#f2b63c" strokeWidth={1.2 * s} strokeDasharray={`${5 * s} ${4 * s}`} opacity="0.7" />; }))}
          {points.map((p) => { const [x, y] = pos.get(p.i)!; const g = groups.of(p.doc_id); const on = !focus || hits!.has(p.i);
            return <g key={p.i} onPointerEnter={() => setHover(p.i)} onPointerLeave={() => setHover(null)} onClick={(e) => { e.stopPropagation(); onPin?.(p.i); }} style={{ cursor: onPin ? "pointer" : "default" }}>
              <circle cx={x} cy={y} r={9 * s} fill="transparent" />
              <Marker x={x} y={y} r={(on && focus ? 5.4 : 4.2) * s} shape={groups.shape(g)} fill={groups.color(g)} opacity={on ? 0.95 : 0.22} />
              {(rings?.has(p.i) || pinned === p.i) && <circle cx={x} cy={y} r={10 * s} fill="none" stroke={pinned === p.i ? "var(--ink)" : "#f2b63c"} strokeWidth={1.6 * s} />}
              {labelled?.has(p.i) && <text x={x + 9 * s} y={y + 4 * s} fontSize={11 * s} fill="var(--muted)" fontFamily="var(--mono)">{tip(p.i).title.slice(0, 26)}</text>}</g>; })}
          {queries.map((q, n) => { const x = X(q.xy[0]), y = Y(q.xy[1]); const main = q.kind === "original" || q.kind === "probe";
            return <g key={n}>{main && <circle cx={x} cy={y} r={34 * s} fill="url(#glow)" />}
              {main ? <path transform={`translate(${x} ${y}) scale(${s})`} d="M0 -11L3.2 -3.6 11 -3.4 4.9 1.6 7 9.4 0 5 -7 9.4 -4.9 1.6 -11 -3.4 -3.2 -3.6Z" fill="#f2b63c" stroke="var(--surface)" strokeWidth="1" />
                : <path transform={`translate(${x} ${y}) scale(${s})`} d="M0 -7L6 0 0 7 -6 0Z" fill={q.kind === "hyde" ? "var(--graph)" : "var(--bm25)"} stroke="var(--surface)" strokeWidth="1" />}
              {q.kind !== "variant" && <text x={x + 13 * s} y={y + (q.kind === "hyde" ? -8 : 4) * s} fontSize={12 * s} fill={main ? "#f2b63c" : "var(--ink)"} fontFamily="var(--mono)">{q.label.slice(0, 40)}</text>}<title>{q.label}</title></g>; })}
        </g>
      </svg>
      {info && hp && <div className="vtip" style={{ left: `${Math.min(62, ((hp[0] * view.k + view.x) / W) * 100 + 1.5)}%`, top: `${Math.min(70, ((hp[1] * view.k + view.y) / H) * 100 + 3)}%` }}>
        <b>{info.title}</b>{info.sub && <span className="muted">{info.sub}</span>}<p>{info.text.slice(0, 190)}{info.text.length > 190 ? "…" : ""}</p></div>}
      <div className="vhint">scroll to zoom, drag to pan, double-click to reset</div>
    </div>
  );
}

/* ------------------------------------------------------------------ rank flow (bump chart) */
export function RankFlow({ columns, items, note }: { columns: { label: string; color?: string; rank: (id: number) => number | null }[];
  items: { id: number; label: string; color: string }[]; note?: string }) {
  const rowH = 27, top = 46, colW = 210, left = 56, labelW = 300;
  const n = items.length; const W = left + colW * (columns.length - 1) + labelW; const absentY = top + rowH * (n + 0.6);
  const cap = n; // ranks beyond the visible range are squeezed just above the "absent" line
  const y = (r: number | null) => r === null ? absentY : r <= cap ? top + rowH * (r - 1) : top + rowH * (cap - 0.35) + Math.min(12, (r - cap) * 1.2);
  const anyAbsent = items.some((it) => columns.some((c) => c.rank(it.id) === null));
  const [hot, setHot] = useState<number | null>(null);
  return (
    <div style={{ overflowX: "auto" }}>
      <svg viewBox={`0 0 ${W} ${absentY + 26}`} style={{ width: "100%", minWidth: 520, maxWidth: W, display: "block" }} role="img" aria-label="Rank of each passage in every list and after merging">
        {columns.map((c, k) => <text key={k} x={left + colW * k} y={20} fontSize="14" fill={c.color ?? "var(--ink)"} textAnchor={k === columns.length - 1 ? "start" : "middle"} fontWeight="600">{c.label}</text>)}
        {anyAbsent && <><line x1={left - 30} x2={left + colW * (columns.length - 1)} y1={absentY} y2={absentY} stroke="var(--rule)" strokeDasharray="3 4" />
          <text x={left - 34} y={absentY + 16} fontSize="11" fill="var(--muted)" fontFamily="var(--mono)">absent from list</text></>}
        {items.map((it) => { const ranks = columns.map((c) => c.rank(it.id)); const dim = hot !== null && hot !== it.id;
          return <g key={it.id} opacity={dim ? 0.18 : 1} onPointerEnter={() => setHot(it.id)} onPointerLeave={() => setHot(null)}>
            {ranks.slice(0, -1).map((r, k) => { const x1 = left + colW * k, x2 = left + colW * (k + 1), y1 = y(r), y2 = y(ranks[k + 1]);
              return <path key={k} d={`M${x1} ${y1}C${x1 + colW / 2} ${y1} ${x2 - colW / 2} ${y2} ${x2} ${y2}`} fill="none" stroke={it.color} strokeWidth={hot === it.id ? 3 : 2}
                strokeDasharray={r === null || ranks[k + 1] === null ? "5 5" : undefined} />; })}
            {ranks.map((r, k) => r !== null && <g key={k}><circle cx={left + colW * k} cy={y(r)} r="5.5" fill={it.color} stroke="var(--surface)" strokeWidth="2"><title>{`${columns[k].label}: ${r}`}</title></circle>
              {k < ranks.length - 1 && r <= cap && <text x={left + colW * k - 12} y={y(r) + 4} fontSize="11" textAnchor="end" fill="var(--muted)" fontFamily="var(--mono)">{r}</text>}</g>)}
            <text x={left + colW * (columns.length - 1) + 14} y={y(ranks[ranks.length - 1]) + 4} fontSize="12.5" fill="var(--ink)" fontFamily="var(--mono)">{ranks[ranks.length - 1]}  {it.label.slice(0, 34)}</text></g>; })}
      </svg>
      {note && <p className="small muted" style={{ marginTop: 4 }}>{note}</p>}
    </div>
  );
}

/* ------------------------------------------------------------------ stage waterfall */
export const STAGE_COLOR: Record<string, string> = { cache: "#7c8aa0", condense: "#4c9a36", normalize: "#1c9db0", verify: "#4c9a36", route: "#8664d8", self_query: "#2aa7a0", transform: "#d4900f", retrieve: "#3f82e6", fuse: "#7c8aa0", rerank: "#21a371", crag: "#d84b85", context: "#a9743a", generate: "#dd5a2e" };

export function Waterfall({ events, metric = "ms" }: { events: Any[]; metric?: "ms" | "tokens" | "cost" }) {
  const val = (e: Any) => metric === "ms" ? e.ms : metric === "tokens" ? (e.usage?.tokens_in ?? 0) + (e.usage?.tokens_out ?? 0) : e.usage?.cost ?? 0;
  const sum = Math.max(events.reduce((a, e) => a + val(e), 0), 1e-9);
  const fmt = (v: number) => metric === "cost" ? `$${v.toFixed(4)}` : String(Math.round(v));
  return (
    <div>
      <div className="wfall">{events.map((e, n) => val(e) > 0 && <i key={n} title={`${e.title}: ${fmt(val(e))}`} style={{ width: `${Math.max(0.6, (100 * val(e)) / sum)}%`, background: STAGE_COLOR[e.stage] ?? "var(--muted)" }} />)}</div>
      <div className="legend small" style={{ marginTop: 6 }}>{events.map((e, n) => <span key={n}><svg width="10" height="10"><rect width="10" height="10" rx="2" fill={STAGE_COLOR[e.stage] ?? "var(--muted)"} /></svg>{e.stage.replace("_", " ")} <b className="mono">{fmt(val(e))}</b></span>)}</div>
    </div>
  );
}

/* ------------------------------------------------------------------ sub-graph */
export function GraphView({ sub }: { sub: { nodes: Any[]; edges: Any[] } }) {
  const W = 640, H = 300; const seeds = sub.nodes.filter((n) => n.seed), rest = sub.nodes.filter((n) => !n.seed);
  const pos = new Map<string, [number, number]>();
  seeds.forEach((n, k) => pos.set(n.id, [W / 2 + (seeds.length > 1 ? 70 * Math.cos((2 * Math.PI * k) / seeds.length) : 0), H / 2 + (seeds.length > 1 ? 34 * Math.sin((2 * Math.PI * k) / seeds.length) : 0)]));
  rest.forEach((n, k) => { const a = (2 * Math.PI * k) / Math.max(rest.length, 1) - Math.PI / 2; pos.set(n.id, [W / 2 + 250 * Math.cos(a), H / 2 + 118 * Math.sin(a)]); });
  const max = Math.max(...sub.nodes.map((n) => n.rank), 1e-9);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="gview" role="img" aria-label="Entities reached from the query by personalised PageRank">
      {sub.edges.map((e, k) => { const a = pos.get(e.s), b = pos.get(e.t); return a && b && <line key={k} x1={a[0]} y1={a[1]} x2={b[0]} y2={b[1]} stroke="var(--graph)" strokeWidth="1" opacity="0.4"><title>{`${e.s} ${e.r} ${e.t}`}</title></line>; })}
      {sub.nodes.map((n) => { const [x, y] = pos.get(n.id)!; const r = 4 + 12 * Math.sqrt(n.rank / max);
        return <g key={n.id}><circle cx={x} cy={y} r={r} fill={n.seed ? "var(--graph)" : "var(--surface)"} stroke="var(--graph)" strokeWidth="1.5" />
          <text x={x} y={y + r + 12} fontSize="10.5" textAnchor="middle" fill="var(--ink)">{n.label.slice(0, 22)}</text><title>{`${n.label}: PageRank mass ${n.rank}`}</title></g>; })}
    </svg>
  );
}

export function Section({ n, title, aside, children }: { n?: string; title: string; aside?: ReactNode; children: ReactNode }) {
  return <section className="panel"><div className="row" style={{ alignItems: "baseline", marginBottom: 12 }}>{n && <span className="mono muted">{n}</span>}<h3 className="grow">{title}</h3>{aside && <span className="small muted" style={{ maxWidth: "52ch", textAlign: "right" }}>{aside}</span>}</div>{children}</section>;
}


/** Metric vs parameter value; a second, dashed series (latency) on its own right-hand scale. */
export function LineChart({ xs, series, right, xLabel }: { xs: (string | number)[]; series: { label: string; color: string; ys: (number | null)[] }[];
  right?: { label: string; ys: number[] }; xLabel: string }) {
  const W = 720, H = 300, L = 48, R = 56, T = 16, B = 46; const n = xs.length;
  const X = (k: number) => L + (n === 1 ? (W - L - R) / 2 : (k / (n - 1)) * (W - L - R)); const Y = (v: number) => T + (1 - v) * (H - T - B);
  const rmax = Math.max(...(right?.ys ?? [1]), 1) * 1.1; const YR = (v: number) => T + (1 - v / rmax) * (H - T - B);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", maxWidth: W, display: "block" }} role="img" aria-label={`Metrics against ${xLabel}`}>
      {[0, 0.25, 0.5, 0.75, 1].map((g) => <g key={g}><line x1={L} x2={W - R} y1={Y(g)} y2={Y(g)} stroke="var(--rule)" strokeWidth="0.7" /><text x={L - 8} y={Y(g) + 4} fontSize="11" textAnchor="end" fill="var(--muted)" fontFamily="var(--mono)">{Math.round(g * 100)}%</text></g>)}
      {xs.map((x, k) => <text key={k} x={X(k)} y={H - B + 18} fontSize="11.5" textAnchor="middle" fill="var(--ink)" fontFamily="var(--mono)">{String(x).slice(0, 14)}</text>)}
      <text x={(L + W - R) / 2} y={H - 6} fontSize="12" textAnchor="middle" fill="var(--muted)">{xLabel}</text>
      {right && <><polyline fill="none" stroke="var(--muted)" strokeWidth="1.5" strokeDasharray="5 4" points={right.ys.map((v, k) => `${X(k)},${YR(v)}`).join(" ")} />
        {right.ys.map((v, k) => <text key={k} x={X(k)} y={YR(v) - 7} fontSize="10.5" textAnchor="middle" fill="var(--muted)" fontFamily="var(--mono)">{Math.round(v)}</text>)}
        <text x={W - 6} y={T + 8} fontSize="11" textAnchor="end" fill="var(--muted)">{right.label}</text></>}
      {series.map((sr) => <g key={sr.label}><polyline fill="none" stroke={sr.color} strokeWidth="2.2" points={sr.ys.map((v, k) => v === null ? "" : `${X(k)},${Y(v)}`).join(" ")} />
        {sr.ys.map((v, k) => v !== null && <circle key={k} cx={X(k)} cy={Y(v)} r="4" fill={sr.color} stroke="var(--surface)" strokeWidth="1.5"><title>{`${sr.label} at ${xs[k]}: ${Math.round(v * 100)}%`}</title></circle>)}</g>)}
    </svg>
  );
}
