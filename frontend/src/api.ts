/* eslint-disable @typescript-eslint/no-explicit-any */
import { Dispatch, SetStateAction, useEffect, useState } from "react";

export type Any = any;

async function req(method: string, url: string, body?: unknown) {
  const r = await fetch(url, {
    method,
    headers: body instanceof FormData || body === undefined ? undefined : { "Content-Type": "application/json" },
    body: body instanceof FormData ? body : body === undefined ? undefined : JSON.stringify(body),
  });
  // A 405 means the path exists but not for this verb, and in this app that has one cause worth
  // naming: the page is newer than the process serving it. Static files are read from disk on
  // every request, so a rebuild reaches the browser at once, while FastAPI fixes its routes when
  // the process starts. "Method Not Allowed" on its own sends you looking for a bug that is not
  // there.
  if (r.status === 405) {
    throw new Error(`The server has no ${method} ${url}. This page is running a newer build than the `
      + `raglab process serving it — restart raglab, then reload this page.`);
  }
  if (!r.ok) {
    let detail = r.statusText;
    try { detail = (await r.json()).detail ?? detail; } catch { /* keep status text */ }
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return r.json();
}

export const api = {
  get: (u: string) => req("GET", u),
  post: (u: string, b?: unknown) => req("POST", u, b ?? {}),
  put: (u: string, b: unknown) => req("PUT", u, b),
  patch: (u: string, b: unknown) => req("PATCH", u, b),
  del: (u: string) => req("DELETE", u),
};

/** POST /api/query and hand each server-sent event to `onEvent`. */
export async function streamQuery(body: unknown, onEvent: (ev: Any) => void, signal?: AbortSignal) {
  const r = await fetch("/api/query", { method: "POST", headers: { "Content-Type": "application/json" },
                                        body: JSON.stringify(body), signal });
  if (!r.ok || !r.body) {
    let detail = r.statusText;
    try { detail = (await r.json()).detail ?? detail; } catch { /* ignore */ }
    throw new Error(detail);
  }
  const reader = r.body.getReader();
  const dec = new TextDecoder();
  let buf = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    let cut: number;
    while ((cut = buf.indexOf("\n\n")) !== -1) {
      const frame = buf.slice(0, cut);
      buf = buf.slice(cut + 2);
      if (frame.startsWith("data: ")) onEvent(JSON.parse(frame.slice(6)));
    }
  }
}

export function deepMerge(base: Any, patch: Any): Any {
  const out = { ...base };
  for (const [k, v] of Object.entries(patch ?? {})) out[k] = v && typeof v === "object" && !Array.isArray(v) ? deepMerge(base?.[k] ?? {}, v) : v;
  return out;
}

export type Run = { events: Any[]; chunks: Record<number, Any>; answer: string; total: number; usage: Any; error: string; traceId: string };

/** Run a pipeline to completion and return its stage events, answer, totals and trace id. */
export async function runCollect(body: unknown): Promise<Run> {
  const out: Run = { events: [], chunks: {}, answer: "", total: 0, usage: {}, error: "", traceId: "" };
  await streamQuery(body, (ev) => {
    if (ev.type === "stage") { out.events.push(ev); Object.assign(out.chunks, ev.chunks); if (ev.data.answer) out.answer = ev.data.answer; }
    else if (ev.type === "token") out.answer += ev.text;
    else if (ev.type === "done") { out.total = ev.total_ms; out.usage = ev.usage ?? {}; out.traceId = ev.trace_id ?? ""; }
    else if (ev.type === "error") out.error = ev.message;
  });
  return out;
}

/**
 * Rebuild a finished run from its stored trace, in the same shape `runCollect` returns.
 *
 * Every query is stored server-side, so a page that kept only the trace id can show the run again
 * after being unmounted — without holding whole passages and prompts in localStorage.
 */
export async function runFromTrace(traceId: string): Promise<Run> {
  const t = await api.get(`/api/traces/${traceId}`);
  const events: Any[] = t.trace ?? [];
  const generated = [...events].reverse().find((e: Any) => e.data?.answer !== undefined);
  return {
    events,
    chunks: events.reduce((acc: Record<number, Any>, e: Any) => Object.assign(acc, e.chunks ?? {}), {}),
    answer: generated?.data?.answer ?? "",
    total: events.reduce((n: number, e: Any) => n + (e.ms ?? 0), 0),
    usage: events.reduce((u: Any, e: Any) => ({ tokens_in: (u.tokens_in ?? 0) + (e.usage?.tokens_in ?? 0),
      tokens_out: (u.tokens_out ?? 0) + (e.usage?.tokens_out ?? 0), cost: (u.cost ?? 0) + (e.usage?.cost ?? 0) }), {}),
    error: "",
    traceId,
  };
}

/**
 * `useState` that survives unmounting. Every page is unmounted when its tab is left, so any
 * setting held in plain state is silently reset on the way back; this keeps it in
 * localStorage instead, under a `raglab.*` key.
 *
 * @param key localStorage key, conventionally `raglab.<page>.<setting>`.
 * @param initial Value to use when nothing is stored yet, or when what is stored is unusable.
 * @param revive Optional check on the stored value before it is trusted — use it to drop a
 *   value whose shape has changed since it was written, or to merge in new defaults. Receives
 *   `undefined` when nothing is stored.
 * @returns The usual `[value, setValue]` pair; `setValue` takes a value or an updater.
 */
export function usePersisted<T>(key: string, initial: T,
                                revive?: (stored: unknown, initial: T) => T): [T, Dispatch<SetStateAction<T>>] {
  const [value, setValue] = useState<T>(() => {
    let stored: unknown;
    try {
      const raw = localStorage.getItem(key);
      if (raw !== null) stored = JSON.parse(raw);
    } catch (e) {
      console.warn(`Ignoring unreadable saved value for ${key}:`, e);
    }
    if (revive) return revive(stored, initial);
    return stored === undefined ? initial : (stored as T);
  });
  useEffect(() => {
    try { localStorage.setItem(key, JSON.stringify(value)); }
    catch (e) { console.warn(`Could not save ${key}:`, e); }   // private mode, or quota
  }, [key, value]);
  return [value, setValue];
}

