import { Fragment, useEffect, useMemo, useRef, useState } from "react";
import { api, Any, usePersisted } from "../api";

/**
 * Metadata keys that identify a single document rather than group several, so they make no facet.
 *
 * `corpus` is excluded because it has a control of its own: collections come from the server with
 * counts, and a dataset with nothing in it has to stay selectable — which a facet built from the
 * documents present can never do, since there are none to build it from.
 */
const NOT_A_FACET = new Set(["source", "title", "doc_key", "date", "superseded_by", "corpus"]);
/** A field with more distinct values than this is closer to an identifier than a category. */
const MAX_FACET_VALUES = 24;

/**
 * Filter controls derived from the metadata actually present, never a fixed list.
 *
 * The two sample corpora already disagree — Halcyon uses `Policy, Manual, Engineering…` and Kestrel
 * uses `spec, runbook, incident…` — and an uploaded corpus will use something else again. A
 * hardcoded Type or Status dropdown is wrong for every corpus but the one it was written for. This
 * mirrors how `Runtime.schema()` discovers filterable fields for self-query, one level up.
 */
function facetsOf(docs: Any[]): { key: string; values: string[] }[] {
  const seen = new Map<string, Map<string, number>>();
  for (const d of docs) {
    for (const [k, v] of Object.entries(d.metadata ?? {})) {
      if (NOT_A_FACET.has(k)) continue;
      for (const item of Array.isArray(v) ? v : [v]) {
        if (item === null || item === undefined || typeof item === "object") continue;
        const bucket = seen.get(k) ?? new Map<string, number>();
        bucket.set(String(item), (bucket.get(String(item)) ?? 0) + 1);
        seen.set(k, bucket);
      }
    }
  }
  return [...seen.entries()]
    .filter(([, values]) => values.size > 1 && values.size <= MAX_FACET_VALUES)
    .map(([key, values]) => ({ key, values: [...values.keys()].sort() }))
    .sort((a, b) => a.key.localeCompare(b.key));
}

/**
 * The collection a document is in, matched without regard to case.
 *
 * Mirrors `collection_of` in raglab/main.py, which groups the counts the same way: dataset names
 * are unique case-insensitively, so `Engineering docs` and `engineering docs` have to be one
 * collection here too or the filter finds nothing in the group the server says has documents.
 */
function inCollection(doc: Any, key: string): boolean {
  const own = String(doc.metadata?.corpus ?? "").trim() || "uploaded";
  return own.toLowerCase() === key.toLowerCase();
}

/** Whether a document carries `value` for `key`, list-valued fields included. */
function has(doc: Any, key: string, value: string): boolean {
  const v = doc.metadata?.[key];
  return (Array.isArray(v) ? v : [v]).some((x) => String(x) === value);
}

/** Metadata the lab owns: shown, never typed over. Mirrors READONLY_METADATA in raglab/main.py. */
const READONLY_METADATA = ["source", "corpus"];

/**
 * How a typed value will be stored, in the reader's words.
 *
 * Mirrors `metadata_value` in raglab/main.py. Shown next to each field because the conversion is
 * otherwise invisible until after saving: `2025` becoming a number and `a, b` becoming two values
 * are both what people want, and both worth seeing before they commit to them.
 */
function readAs(raw: string): string {
  const text = raw.trim();
  if (!text) return "empty — the field will be removed";
  const bracketed = text.startsWith("[") && text.endsWith("]");
  if (bracketed || text.includes(",")) {
    const inner = bracketed ? text.slice(1, -1) : text;
    const members = inner.split(",").map((s) => s.trim()).filter(Boolean);
    if (members.length) return `list of ${members.length}: ${members.join(" · ")}`;
  }
  if (["true", "false"].includes(text.toLowerCase())) return "true/false";
  if (text !== "" && !isNaN(Number(text))) return "number";
  return "text";
}

/** A row in the editor. `was` is what the document already held, absent for a field being added. */
/** What a stored value already is, for a row nobody has typed in. */
function describe(value: Any): string {
  if (Array.isArray(value)) return `list of ${value.length}`;
  if (typeof value === "boolean") return "true/false";
  if (typeof value === "number") return "number";
  return "text";
}

type Field = { k: string; v: string; was?: Any };

/** How a stored value is shown in a text box. */
function show(value: Any): string {
  return String(Array.isArray(value) ? value.join(", ") : value);
}

/**
 * Add, rename, retype and remove a document's metadata.
 *
 * An uploaded PDF has no front matter, so it arrives with nothing to filter on and nothing to
 * show self-query. Without this there is no way to give it any.
 */
function MetadataEditor({ doc, open, setOpen, onSaved }:
                        { doc: Any; open: boolean; setOpen: (v: boolean) => void; onSaved: (r: Any) => void }) {
  const editable = Object.entries(doc.metadata ?? {}).filter(([k]) => !READONLY_METADATA.includes(k));
  const asFields = (): Field[] => editable.map(([k, v]) => ({ k, v: show(v), was: v }));
  const [title, setTitle] = useState(doc.title ?? "");
  const [fields, setFields] = useState<Field[]>(asFields);
  const [errors, setErrors] = useState<Any[]>([]);
  const [saving, setSaving] = useState(false);

  const set = (i: number, patch: Partial<Field>) =>
    setFields(fields.map((f, n) => (n === i ? { ...f, ...patch } : f)));
  // A field nobody typed in keeps exactly what the document held. Re-reading it from its own
  // rendering is what turned `version: "2.0"` into the number 2.0 and split `"teachers,
  // principals"` — one string — into two.
  const untouched = (f: Field) => f.was !== undefined && f.v === show(f.was);
  const duplicate = (i: number) => {
    const name = fields[i].k.trim();
    return !!name && fields.some((f, n) => n !== i && f.k.trim() === name);
  };
  const close = () => { setOpen(false); setErrors([]); setTitle(doc.title ?? ""); setFields(asFields()); };

  const save = async () => {
    if (fields.some((_, i) => duplicate(i))) {
      setErrors([{ field: "metadata", message: "Two fields have the same name. Rename or remove one." }]);
      return;
    }
    setErrors([]); setSaving(true);
    try {
      const metadata: Record<string, Any> = {};
      for (const f of fields) if (f.k.trim()) metadata[f.k.trim()] = untouched(f) ? { keep: f.was } : f.v;
      onSaved(await api.patch(`/api/documents/${doc.id}`, { title, metadata }));
      setOpen(false);
    } catch (e) {
      // The server answers 422 with a message per field; anything else is shown as it arrived.
      let parsed: Any = null;
      try { parsed = JSON.parse(e instanceof Error ? e.message : String(e)); } catch { /* not JSON */ }
      setErrors(parsed?.errors ?? [{ field: "", message: e instanceof Error ? e.message : String(e) }]);
    } finally { setSaving(false); }
  };

  const meta = Object.entries(doc.metadata ?? {}).filter(([k]) => k !== "source");
  if (!open) {
    return (
      <div className="stack" style={{ gap: 6, marginBottom: 10 }}>
        {!!meta.length && <dl className="kv">
          {meta.map(([k, v]) => <Fragment key={k}><dt>{k}</dt><dd>{show(v)}</dd></Fragment>)}</dl>}
        <div><button className="link" onClick={() => setOpen(true)}>
          {meta.length ? "Edit metadata" : "Add metadata"}</button></div>
      </div>
    );
  }
  return (
    <div className="stack" style={{ gap: 8, marginBottom: 10 }}>
      <label className="f"><span>Title</span>
        <input type="text" value={title} onChange={(e) => setTitle(e.target.value)} disabled={saving} /></label>
      {fields.map((f, i) => <div key={i} className="stack" style={{ gap: 2 }}>
        {/* Top-aligned: a long value makes a tall box, and a centred name and Remove drift to the
            middle of it, away from the field they belong to. */}
        <div className="row" style={{ gap: 6, alignItems: "flex-start" }}>
          <input type="text" value={f.k} placeholder="doc_type" aria-label={`Field ${i + 1} name`}
                 style={{ width: 150 }} disabled={saving} onChange={(e) => set(i, { k: e.target.value })} />
          <textarea className="grow field" rows={Math.min(8, Math.ceil((f.v.length || 1) / 60))}
                    value={f.v} placeholder="guide" aria-label={`Field ${i + 1} value`}
                    disabled={saving} onChange={(e) => set(i, { v: e.target.value })} />
          <button className="link" disabled={saving} aria-label={`Remove ${f.k || `field ${i + 1}`}`}
                  onClick={() => setFields(fields.filter((_, n) => n !== i))}>Remove</button>
        </div>
        <span className="small muted" style={{ paddingLeft: 156 }}>
          {duplicate(i) ? <b className="err-text">used twice</b>
            : untouched(f) ? `unchanged — kept as ${describe(f.was)}`
            : `stored as ${readAs(f.v)}`}</span>
      </div>)}
      <div className="row">
        <button className="btn ghost" disabled={saving}
                onClick={() => setFields([...fields, { k: "", v: "" }])}>Add field</button>
      </div>
      <p className="small muted" style={{ margin: 0 }}>Values are stored as text unless they look like a
        number or <code>true</code>/<code>false</code>, or are separated by commas — then they become a list,
        and a filter matches any one of them. Emptying a value removes the field.{" "}
        {READONLY_METADATA.map((k, i) => <span key={k}>{i ? " and " : ""}<code>{k}</code></span>)} are set by
        the lab: a document's collection follows the dataset it was added to.</p>
      {!!errors.length && <div className="err small">
        {errors.map((x: Any, i: number) => <div key={i}>{x.field ? <b>{x.field}: </b> : null}{x.message}</div>)}</div>}
      <div className="row">
        <button className="btn" disabled={saving} onClick={save}>{saving ? "Saving…" : "Save"}</button>
        <button className="btn ghost" disabled={saving} onClick={close}>Cancel</button>
      </div>
    </div>
  );
}

/**
 * The document itself, over the list rather than below it.
 *
 * Appending the preview under the table meant clicking a title near the top of thirty-odd rows
 * looked like it did nothing. A panel fixed to the viewport also leaves the list's scroll position
 * exactly where it was.
 *
 * The text is shown raw, not rendered: what you are checking is the extraction — front matter, PDF
 * layout damage, encoding — and rendering hides precisely the damage worth seeing.
 */
function Drawer({ doc, usage, onClose, onStep, position, editing, onSaved, onDoneEditing }:
                { doc: Any; usage: Any[]; onClose: () => void; onStep: (d: number) => void; position: string;
                  editing: boolean; onSaved: (r: Any) => void; onDoneEditing: () => void }) {
  // Whether the metadata editor is open lives here rather than inside it, because the shortcuts
  // below have to know: Escape closing the drawer, or an arrow key stepping to the next document,
  // would throw away half-typed fields without a word.
  const [editorOpen, setEditorOpen] = useState(editing);
  // The drawer stays mounted when a different row is opened, so "Add metadata" on a second row
  // has to reopen the editor rather than relying on the initial value alone.
  useEffect(() => { if (editing) setEditorOpen(true); }, [editing, doc.id]);
  useEffect(() => { if (!editorOpen) onDoneEditing(); }, [editorOpen, onDoneEditing]);
  useEffect(() => {
    const key = (e: KeyboardEvent) => {
      if (editorOpen) return;
      if (e.key === "Escape") onClose();
      if (e.key === "ArrowDown") { e.preventDefault(); onStep(1); }
      if (e.key === "ArrowUp") { e.preventDefault(); onStep(-1); }
    };
    window.addEventListener("keydown", key);
    return () => window.removeEventListener("keydown", key);
  }, [onClose, onStep, editorOpen]);

  return (
    <>
      <div className="drawer-scrim" onClick={onClose} aria-hidden />
      <aside className="drawer" role="dialog" aria-label={doc.title}>
        <div className="row" style={{ marginBottom: 10 }}>
          <h3 className="grow" style={{ margin: 0 }}>{doc.title}</h3>
          <button className="link" disabled={editorOpen} onClick={() => onStep(-1)} aria-label="Previous document">↑</button>
          <button className="link" disabled={editorOpen} onClick={() => onStep(1)} aria-label="Next document">↓</button>
          <button className="link" onClick={onClose} aria-label="Close">Close</button>
        </div>
        <div className="row small muted" style={{ marginBottom: 10 }}>
          <span>{position}</span>
          <span>{Math.round((doc.text?.length ?? 0) / 4)} approx. tokens</span>
          {doc.metadata?.source && <span className="mono">{doc.metadata.source}</span>}
        </div>

        {/* Keyed on the document so stepping to the next one cannot leave the previous one's
            half-typed fields on screen against the wrong title. */}
        <MetadataEditor key={doc.id} doc={doc} open={editorOpen} setOpen={setEditorOpen} onSaved={onSaved} />

        <div className="note small" style={{ marginBottom: 10 }}>
          {usage.length
            ? <>Stored in {usage.length === 1 ? "one index" : `${usage.length} indexes`}:{" "}
                {usage.map((u: Any) => `${u.name} (${u.chunks} chunk${u.chunks === 1 ? "" : "s"})`).join(", ")}. An index keeps its own copy,
                so it holds this text until it is rebuilt.</>
            : <>No index has been built from this document yet, so it cannot be retrieved.</>}
        </div>

        <h4 style={{ marginBottom: 6 }}>Extracted text</h4>
        <p className="small muted" style={{ marginTop: 0 }}>Shown as extracted, not rendered — this is what gets
          chunked and embedded, so damage from a PDF or an encoding shows here.</p>
        <pre className="drawer-text">{doc.text}</pre>
      </aside>
    </>
  );
}

/** What an import actually did, file by file.
 *
 * A file that yields no text used to be dropped in silence, which is indistinguishable from the
 * button being broken — so failures are named, with the reason, and a scanned PDF says so.
 */
function Report({ report, onClose }: { report: Any; onClose: () => void }) {
  const bad: Any[] = (report.results ?? []).filter((r: Any) => !r.ok);
  const good = (report.results ?? []).length - bad.length;
  // Naming the destination is the point: an upload that landed in the wrong collection is
  // otherwise only discoverable by going and looking for it.
  const into = report.dataset ? ` to ${report.dataset}` : "";
  const headline = report.kind === "sample"
    ? report.added
      ? `Loaded ${report.added} document${report.added === 1 ? "" : "s"} from ${report.name}.`
        + (report.skipped ? ` ${report.skipped} were already in the corpus.` : "")
      : `Nothing to load — all ${report.skipped} ${report.name} documents are already in the corpus.`
    : report.kind === "paste"
      ? `Added the pasted document${into}.`
      : `Added ${good} document${good === 1 ? "" : "s"}${into}.`
        + (bad.length ? ` ${bad.length} file${bad.length === 1 ? "" : "s"} could not be imported.` : "");
  return (
    <div className={bad.length ? "err" : "note"} style={{ marginBottom: 12 }}>
      {headline} <button className="link" onClick={onClose}>Dismiss</button>
      {!!bad.length && <ul className="small" style={{ margin: "6px 0 0", paddingLeft: 18 }}>
        {bad.map((r: Any, i: number) => <li key={`${r.filename}-${i}`}><b>{r.filename}</b> — {r.reason}</li>)}</ul>}
    </div>
  );
}

/** The special value standing for "unassigned" in the destination select. */
const UNASSIGNED = "";
/** The special value standing for the create-a-dataset choice. */
const NEW_DATASET = "__new";

/**
 * Where uploads and pasted documents land.
 *
 * Deliberately not styled or placed like the filters on the right: this one changes what happens
 * to the next import, and the others only change what is on screen. Confusing the two would mean
 * filtering to a collection and expecting the next upload to join it.
 */
function Destination({ datasets, value, onChange, onCreate, disabled }: {
  datasets: Any[]; value: string; onChange: (v: string) => void;
  onCreate: (name: string) => Promise<void>; disabled: boolean;
}) {
  const [creating, setCreating] = useState(false);
  const [name, setName] = useState("");
  const [problem, setProblem] = useState("");
  const [saving, setSaving] = useState(false);

  const cancel = () => { setCreating(false); setName(""); setProblem(""); };
  const create = async () => {
    setProblem(""); setSaving(true);
    try { await onCreate(name); cancel(); }
    catch (e) { setProblem(e instanceof Error ? e.message : String(e)); }
    finally { setSaving(false); }
  };

  return (
    <div className="stack" style={{ gap: 6 }}>
      <label className="f"><span>Add to dataset</span>
        <select value={value} disabled={disabled || creating}
                onChange={(e) => (e.target.value === NEW_DATASET ? setCreating(true) : onChange(e.target.value))}>
          <option value={UNASSIGNED}>Uploaded / unassigned</option>
          {datasets.map((d: Any) => <option key={d.dataset_id} value={d.dataset_id}>{d.label} ({d.docs})</option>)}
          <option value={NEW_DATASET}>New dataset…</option>
        </select></label>
      {creating && <div className="stack" style={{ gap: 6 }}>
        <input type="text" value={name} autoFocus placeholder="Product manuals"
               aria-label="New dataset name" disabled={saving}
               onChange={(e) => { setName(e.target.value); setProblem(""); }}
               onKeyDown={(e) => { if (e.key === "Enter") create(); if (e.key === "Escape") cancel(); }} />
        {problem && <p className="small err" style={{ margin: 0 }}>{problem}</p>}
        <div className="row"><button className="btn" disabled={saving || !name.trim()} onClick={create}>
          {saving ? "Creating…" : "Create"}</button>
          <button className="btn ghost" disabled={saving} onClick={cancel}>Cancel</button></div>
      </div>}
    </div>
  );
}

export default function Corpus() {
  const [docs, setDocs] = useState<Any[]>([]);
  const [allDocs, setAllDocs] = useState<Any[]>([]);
  // An unfinished document is a draft, so it survives leaving the tab.
  const [title, setTitle] = usePersisted("raglab.corpus.draftTitle", "");
  const [text, setText] = usePersisted("raglab.corpus.draftText", "");
  const [openId, setOpenId] = useState(""); const [open, setOpen] = useState<Any>(null);
  // Set when the drawer is opened by "Add metadata" rather than by the title, so it opens with
  // the editor already up. Cleared once the editor is closed, so stepping to the next document
  // does not keep forcing it open.
  const [editing, setEditing] = useState(false);
  const [err, setErr] = useState("");
  const file = useRef<HTMLInputElement>(null); const [samples, setSamples] = useState<Any[]>([]);
  // Which indexes hold a stored copy of each document. An index is a snapshot, so removing a
  // document from the corpus leaves that copy behind; saying so up front is what makes Remove
  // explainable rather than surprising.
  const [usage, setUsage] = useState<Record<string, Any[]>>({});
  const [note, setNote] = useState("");
  // The document awaiting confirmation. Inline rather than window.confirm: the rest of the app
  // explains itself in the page, and a native dialog cannot say which indexes are affected.
  const [pending, setPending] = useState<Any>(null);
  // Search is served by the API because the list carries no text; the facets are client-side
  // because the list already carries every document's metadata.
  const [query, setQuery] = usePersisted("raglab.corpus.search", "");
  const [picked, setPicked] = usePersisted<Record<string, string>>("raglab.corpus.facets", {});
  const [searching, setSearching] = useState(false);
  // Which import is in flight, and what the last one did. Without both, an upload that skipped a
  // scanned PDF looks the same as one still running, and the same as one that is broken.
  const [busy, setBusy] = useState("");
  const [report, setReport] = useState<Any>(null);
  // Collections and the chosen upload destination. The destination is a dataset id, "" for the
  // unassigned bucket; it survives leaving the tab, and falls back to "" if that dataset is gone.
  const [collections, setCollections] = useState<Any[]>([]);
  const [dest, setDest] = usePersisted("raglab.corpus.destination", UNASSIGNED);
  const [collection, setCollection] = usePersisted("raglab.corpus.collection", "");

  const refresh = async (q = query) => {
    const [d, u, all, cs] = await Promise.all([
      api.get(`/api/documents${q.trim() ? `?q=${encodeURIComponent(q.trim())}` : ""}`),
      api.get("/api/documents-usage"),
      api.get("/api/documents"),
      api.get("/api/collections"),
    ]);
    setDocs(d); setUsage(u); setAllDocs(all); setCollections(cs);
    // A remembered destination or filter that no longer exists would silently send the next
    // upload somewhere the user cannot see, so both fall back to their neutral value.
    setDest((cur) => (!cur || cs.some((c: Any) => c.dataset_id === cur) ? cur : UNASSIGNED));
    setCollection((cur) => (!cur || cs.some((c: Any) => c.key === cur) ? cur : ""));
  };
  // How much of each sample is loaded comes from the server, so it is refetched whenever the
  // corpus changes — including after a removal, which puts "Load 1 more" back on the button.
  const reloadSamples = () => api.get("/api/samples").then(setSamples).catch((e) => setErr(String(e)));
  useEffect(() => { refresh(); reloadSamples(); }, []); // eslint-disable-line
  // Debounced: the search runs on the server, so it should not run on every keystroke.
  useEffect(() => {
    const t = setTimeout(() => {
      setSearching(true);
      refresh(query).catch((e) => setErr(String(e))).finally(() => setSearching(false));
    }, 250);
    return () => clearTimeout(t);
  }, [query]); // eslint-disable-line

  const run = async (fn: () => Promise<unknown>, label = "") => {
    setErr(""); setBusy(label);
    try { await fn(); await refresh(); await reloadSamples(); } catch (e) { setErr(String(e)); } finally { setBusy(""); }
  };
  const remove = (d: Any) => run(async () => {
    const r = await api.del(`/api/documents/${d.id}`);
    setPending(null);
    if (openId === d.id) { setOpenId(""); setOpen(null); }
    setNote(r.note);
  });
  const upload = (files: FileList | null) => {
    if (!files?.length) return;
    const fd = new FormData(); Array.from(files).forEach((f) => fd.append("files", f));
    // Read once, here: changing the selector while an upload is in flight must not redirect it.
    const into = dest;
    setReport(null);
    run(async () => {
      const r = await api.post(`/api/documents/upload${into ? `?dataset=${encodeURIComponent(into)}` : ""}`, fd);
      setReport({ kind: "upload", results: r.results ?? [], dataset: r.dataset });
    }, "upload");
  };
  const createDataset = async (name: string) => {
    // Only the create is inside the caller's try: a reload that fails afterwards is not a problem
    // with the name, and reporting it under the name field would say it was.
    const made = await api.post("/api/datasets", { name: name.trim() });
    setDest(made.id);   // the one you just named is the one you meant to upload into
    refresh().catch((e) => setErr(String(e)));
  };
  const loadSample = (name: string) => {
    setReport(null);
    run(async () => setReport({ kind: "sample", name, ...(await api.post(`/api/documents/sample?name=${name}`)) }),
        `sample:${name}`);
  };

  // Facets come from the whole corpus, not the filtered view, so choosing one value does not make
  // every other value vanish from its own dropdown.
  const facets = useMemo(() => facetsOf(allDocs.length ? allDocs : docs), [allDocs, docs]);
  const shown = useMemo(() => docs.filter((d) =>
    (!collection || inCollection(d, collection))
    && Object.entries(picked).every(([k, v]) => !v || has(d, k, v))), [docs, picked, collection]);
  const filtered = !!query.trim() || !!collection || Object.values(picked).some(Boolean);
  const clear = () => { setQuery(""); setPicked({}); setCollection(""); };

  // The drawer needs the full text, which the list deliberately does not carry.
  useEffect(() => {
    if (!openId) { setOpen(null); return; }
    api.get(`/api/documents/${openId}`).then(setOpen).catch(() => { setOpenId(""); setOpen(null); });
  }, [openId]);
  const at = shown.findIndex((d) => d.id === openId);
  const step = (delta: number) => { const next = shown[at + delta]; if (next) setOpenId(next.id); };

  return (
    <>
      <div className="pagehead"><h2>Corpus</h2>
        <p>Plain text, Markdown and PDF. YAML front matter (<code>doc_type: runbook</code>, <code>year: 2025</code>) becomes filterable metadata for self-query.</p></div>
      {err && <div className="err" style={{ marginBottom: 12 }}>{err}</div>}
      {note && <div className="note" style={{ marginBottom: 12 }}>{note} <button className="link" onClick={() => setNote("")}>Dismiss</button></div>}
      {report && <Report report={report} onClose={() => setReport(null)} />}
      {pending && <div className="err" style={{ marginBottom: 12 }}>
        Remove <b>{pending.title}</b> from the corpus?{" "}
        {(usage[pending.id] ?? []).length
          ? <>It stops being retrieved straight away, but {(usage[pending.id] ?? []).length === 1
                ? "one index still holds" : `${(usage[pending.id] ?? []).length} indexes still hold`} a stored
              copy of its chunks ({(usage[pending.id] ?? []).map((u: Any) => `${u.name}: ${u.chunks}`).join(", ")}).
              Rebuild {(usage[pending.id] ?? []).length === 1 ? "it" : "those"} to drop it entirely.</>
          : <>No index has been built from it, so nothing else is affected.</>}{" "}
        Use Confirm on its row, or Cancel.</div>}
      <div className="split">
        <div className="panel stack">
          <h3>Add documents</h3>
          <Destination datasets={collections.filter((c: Any) => c.kind === "dataset")} value={dest}
                       onChange={setDest} onCreate={createDataset} disabled={!!busy} />
          <p className="small muted" style={{ marginTop: -4 }}>Uploads and pasted documents go here.
            Sample corpora always load as themselves.</p>
          <input ref={file} type="file" multiple accept=".md,.txt,.pdf,.markdown" hidden onChange={(e) => { upload(e.target.files); e.target.value = ""; }} />
          <div className="row"><button className="btn" disabled={!!busy} onClick={() => file.current?.click()}>
            {busy === "upload" ? "Importing…" : "Upload files"}</button></div>
          <div><h4 style={{ marginBottom: 4 }}>Sample corpora</h4>{samples.map((sm) => {
            const have = sm.loaded ?? 0, complete = have >= sm.docs;
            return <div key={sm.name} style={{ borderTop: "1px solid var(--rule)", padding: "8px 0" }}>
              <div className="row"><b className="grow" style={{ textTransform: "capitalize" }}>{sm.name}</b>
                <button className="btn ghost" disabled={!!busy || complete} onClick={() => loadSample(sm.name)}>
                  {busy === `sample:${sm.name}` ? "Loading…"
                    : complete ? `Loaded · ${have}/${sm.docs}`
                    : have ? `Load ${sm.docs - have} more` : `Load ${sm.docs} docs`}</button></div>
              <p className="small muted" style={{ marginTop: 4 }}>{sm.about}</p></div>;
          })}</div>
          <label className="f"><span>Or paste text. Title</span><input type="text" value={title} onChange={(e) => setTitle(e.target.value)} /></label>
          <textarea value={text} onChange={(e) => setText(e.target.value)} placeholder="Paste document text or Markdown" />
          <button className="btn ghost" disabled={!text.trim() || !!busy}
                  onClick={() => { const into = dest; setReport(null); run(async () => {
              const r = await api.post("/api/documents", { title, text, dataset: into });
              setTitle(""); setText("");
              setReport({ kind: "paste", dataset: r.dataset });
            }, "paste"); }}>
            {busy === "paste" ? "Adding…" : "Add pasted document"}</button>
        </div>
        <div className="panel">
          {/* Wraps: a corpus with several metadata fields puts six or more controls on this row,
              and without wrapping they squeeze the search box down to its border. */}
          <div className="row" style={{ marginBottom: 10, flexWrap: "wrap", rowGap: 8 }}>
            <input className="grow" type="text" value={query} onChange={(e) => setQuery(e.target.value)}
                   style={{ minWidth: 220 }}
                   placeholder="Search titles and document text…" aria-label="Search documents" />
            <select value={collection} aria-label="collection" style={{ width: 168 }}
                    onChange={(e) => setCollection(e.target.value)}>
              <option value="">every collection</option>
              {collections.map((c: Any) => <option key={c.key} value={c.key}>{c.label} ({c.docs})</option>)}
            </select>
            {facets.map((f) => <select key={f.key} value={picked[f.key] ?? ""} aria-label={f.key} style={{ width: 148 }}
                                       onChange={(e) => setPicked({ ...picked, [f.key]: e.target.value })}>
              <option value="">{f.key}: any</option>
              {f.values.map((v) => <option key={v} value={v}>{v}</option>)}</select>)}
          </div>

          <div className="row" style={{ marginBottom: 8 }}>
            <h3 className="grow" style={{ margin: 0 }}>
              {filtered ? `Showing ${shown.length} of ${allDocs.length}` : `${shown.length}`} document{shown.length === 1 ? "" : "s"}
              {searching && <span className="small muted"> · searching…</span>}
            </h3>
            {filtered && <button className="link" onClick={clear}>Clear filters</button>}
          </div>

          {!allDocs.length && <p className="muted">The corpus is empty. Upload files or load a sample corpus to get started.</p>}
          {!!allDocs.length && !shown.length && (collection && collections.find((c: Any) => c.key === collection)?.docs === 0
            ? <p className="muted"><b>{collections.find((c: Any) => c.key === collection)?.label ?? collection}</b> is empty — nothing has been uploaded into it yet.
                Choose it under “Add to dataset” and upload some files.{" "}
                <button className="link" onClick={clear}>Clear filters</button></p>
            : <p className="muted">No document matches. <button className="link" onClick={clear}>Clear filters</button></p>)}

          {!!shown.length && <table className="fixed" style={{ marginTop: 4 }}><thead><tr>
            <th style={{ width: "26%" }}>Document</th><th>Metadata</th>
            <th className="num" style={{ width: 118 }}>Approx. tokens</th>
            <th className="num" style={{ width: 84 }}>In indexes</th>
            <th style={{ width: 118 }} /></tr></thead><tbody>
            {shown.map((d) => <tr key={d.id}>
              <td>
                <button className="link" onClick={() => setOpenId(d.id)}>{d.title}</button>
                {d.metadata?.superseded_by && <> <span className="tag bad" title={`Replaced by ${d.metadata.superseded_by}`}>superseded</span></>}
                {d.metadata?.doc_key && <div className="small muted mono">{d.metadata.doc_key}</div>}
                {d.snippet && <div className="small muted" style={{ marginTop: 2 }}>{d.snippet}</div>}
              </td>
              <td>{Object.entries(d.metadata).filter(([k]) => !["source", "doc_key", "superseded_by"].includes(k))
                .map(([k, v]) => <span className="tag" key={k} style={{ marginRight: 4 }}
                                       title={`${k}: ${show(v)}`}>{k}: {show(v)}</span>)}
                {" "}<button className="link small" title="Add or change fields for self-query filtering"
                             onClick={() => { setEditing(true); setOpenId(d.id); }}>
                  {Object.keys(d.metadata).some((k) => !["source", "corpus"].includes(k)) ? "Edit" : "Add metadata"}
                </button></td>
              <td className="num" title="Characters divided by four, not a tokenizer count">{Math.round(d.chars / 4)}</td>
              <td className="num">{(usage[d.id] ?? []).length
                ? <span className="tag" title={(usage[d.id] ?? []).map((u: Any) => `${u.name} (${u.chunks} chunk${u.chunks === 1 ? "" : "s"})`).join(", ")}>
                    {(usage[d.id] ?? []).length}</span>
                : <span className="muted">—</span>}</td>
              <td className="num">{pending?.id === d.id
                ? <><button className="link" onClick={() => remove(d)}>Confirm</button>{" "}
                    <button className="link" onClick={() => setPending(null)}>Cancel</button></>
                : <button className="link" onClick={() => { setNote(""); setPending(d); }}>Remove</button>}</td></tr>)}
          </tbody></table>}
        </div>
      </div>
      {open && <Drawer doc={open} usage={usage[open.id] ?? []} position={at < 0 ? "" : `${at + 1} of ${shown.length}`}
                       onClose={() => { setOpenId(""); setEditing(false); }} onStep={step}
                       editing={editing} onDoneEditing={() => setEditing(false)}
                       onSaved={(saved: Any) => {
                         setOpen(saved);
                         const stale: Any[] = saved.stale ?? [];
                         const one = stale.length === 1;
                         setNote(stale.length
                           ? `Saved. ${one ? "One index" : `${stale.length} indexes`} `
                             + `(${stale.map((u: Any) => u.name).join(", ")}) ${one ? "was" : "were"} built before `
                             + `this change and ${one ? "still filters" : "still filter"} on the old fields. `
                             + `Rebuild to pick the new ones up.`
                           : "Saved.");
                         refresh().catch((e) => setErr(String(e)));
                       }} />}
    </>
  );
}
