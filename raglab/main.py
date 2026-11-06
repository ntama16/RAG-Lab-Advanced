"""FastAPI app: REST + SSE for the lab UI, which it also serves from /."""
from __future__ import annotations
import asyncio, copy, datetime, io, json, logging, re, sqlite3, time, uuid
from pathlib import Path

import yaml
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from . import config, evals, indexer, normalize, providers
from .chunking import chunk_text
from .pipeline import BOUNDS as PIPELINE_BOUNDS, DEFAULT_PIPELINE, Pipeline, controlled as pipeline_controlled, \
    inactive as pipeline_inactive, merged as merged_pipeline, validate as pipeline_validate
from .store import Store

app = FastAPI(title="RAG Lab Advanced")
store = Store()
MODELS = config.load_models()
STATIC = Path(__file__).parent / "static"
log = logging.getLogger("raglab")

# Background work, keyed by the id the client polls. asyncio keeps only weak references to tasks, so
# a build or an evaluation that is not held here can be collected while it is still running; holding
# them is also what makes Stop possible.
TASKS: dict[str, asyncio.Task] = {}


def _track(coro, key: str, on_cancelled=None) -> asyncio.Task:
    """Run a job in the background under the id the client polls.

    :param coro: The job.
    :param key: The id to hold it under.
    :param on_cancelled: Called once the task has been cancelled. A job stopped while queued never
        started, so anything it would have cleaned up after itself has to be done from here.
    :returns: The running task.
    """
    task = asyncio.create_task(coro)
    TASKS[key] = task

    def finished(done: asyncio.Task) -> None:
        TASKS.pop(key, None)
        if on_cancelled and done.cancelled():
            on_cancelled()
    task.add_done_callback(finished)
    return task


# One batch job at a time. Builds, evaluations and sweeps all drive the same local model server, and
# running two together makes both slower and their measured latencies meaningless. Interactive
# queries deliberately stay outside this permit so the Workbench keeps answering during a long run.
BATCH = asyncio.Semaphore(1)


async def _batch(coro):
    try:
        await BATCH.acquire()
    except asyncio.CancelledError:
        coro.close()  # stopped in the queue: it never started, so nothing would ever await it
        raise
    try:
        return await coro
    finally:
        BATCH.release()


def _start_build(iid: str) -> asyncio.Task:
    """Queue an index build, and fail the index if it is stopped before it gets to start."""
    return _track(_batch(indexer.build(store, iid, _models_snapshot())), iid,
                  on_cancelled=lambda: indexer.mark_stopped(store, iid))


def _mark_interrupted_builds() -> int:
    """Fail any index left mid-build by a stopped process, rather than leaving it stuck on
    'building' with no job behind it and no way to tell it from one still going.

    :returns: How many indexes were marked.
    """
    stuck = store.q("SELECT id FROM indexes WHERE status='building'")
    if stuck:
        store.x("UPDATE indexes SET status='failed', stats=? WHERE status='building'",
                (json.dumps({"error": "Interrupted: the server stopped while this index was building. Rebuild it."}),))
    return len(stuck)


_mark_interrupted_builds()


#: How deep a nested front-matter map is flattened before the rest is dropped. Past this, the
#: dotted key is longer than anything a reader would recognise in a filter dropdown.
META_DEPTH = 4


def _scalar(value):
    """One front-matter value, in a form the store, the facets and self-query can all use.

    :param value: Whatever the YAML reader produced.
    :returns: A JSON-storable scalar. Dates become ISO strings — `date: 2025-05-30` parses as a
        :class:`datetime.date`, which :func:`json.dumps` cannot write.
    """
    if isinstance(value, (datetime.date, datetime.datetime)):
        return value.isoformat()
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _flatten(node: dict, out: dict, prefix: str = "", depth: int = 0) -> dict:
    """Fold a parsed front-matter mapping into flat ``dotted.key`` entries.

    Nested maps become `author.name`; lists keep only their scalar members; nulls are dropped,
    because a key with no value is not something anyone filters on.

    :param node: The mapping to read.
    :param out: Accumulator, mutated in place.
    :param prefix: Dotted path of the enclosing keys.
    :param depth: Current nesting depth, against :data:`META_DEPTH`.
    :returns: ``out``.
    """
    for key, value in node.items():
        path = f"{prefix}{key}"
        if isinstance(value, dict):
            if depth < META_DEPTH:
                _flatten(value, out, f"{path}.", depth + 1)
        elif isinstance(value, list):
            members = [_scalar(v) for v in value if v is not None and not isinstance(v, (dict, list))]
            if members:
                out[path] = members
        elif value is not None:
            out[path] = _scalar(value)
    return out


def _loose_pairs(block: str) -> dict:
    """Read a front-matter block a line at a time, for blocks a YAML reader refuses.

    Kept as a fallback so a malformed block — an unquoted title with a colon in it, most often —
    still imports with its fields rather than losing all of them.

    :param block: The text between the `---` fences.
    :returns: The `key: value` pairs it can make out.
    """
    meta = {}
    for line in block.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        value = value.strip()
        if value.startswith("[") and value.endswith("]"):
            meta[key.strip()] = [x.strip().strip("'\"") for x in value[1:-1].split(",") if x.strip()]
        else:
            try:
                meta[key.strip()] = int(value)
            except ValueError:
                meta[key.strip()] = value.strip("'\"")
    return meta


def _front_matter(text: str) -> tuple[dict, str]:
    """Split a leading YAML front-matter block off a document.

    Read with a real YAML parser, because the hand-rolled one mangled ordinary documents in
    silence: a nested `author:` map became an empty `author` plus invented top-level keys, and a
    folded `summary: >` stored the `>` as the value. Front matter is what self-query filters on and
    what the Corpus facets are built from, so a wrong key is a filter that quietly matches nothing.

    :param text: The document, which may or may not open with a `---` block.
    :returns: ``(metadata, body)``; metadata is empty when there is no block.
    """
    m = re.match(r"^---\s*\n(.*?)\n---\s*\n", text, re.S)
    if not m:
        return {}, text
    body = text[m.end():]
    try:
        loaded = yaml.safe_load(m.group(1))
    except Exception as exc:
        # Deliberately broader than `yaml.YAMLError`: deeply nested flow collections raise
        # `RecursionError`, which is not one, and an uncaught one here returns 500 for the whole
        # request — losing the healthy files uploaded alongside it, which is the failure this
        # endpoint's per-file reporting exists to prevent.
        log.warning("front matter could not be parsed as YAML (%s); reading it line by line instead", exc)
        return _loose_pairs(m.group(1)), body
    return (_flatten(loaded, {}), body) if isinstance(loaded, dict) else ({}, body)


def _ingest(name: str, text: str, extra: dict | None = None, title: str = "") -> str:
    """Store one document, taking its title from the caller, its front matter or its first heading.

    :param name: Filename or other origin, recorded as `source`.
    :param text: The raw document, front matter included.
    :param extra: Metadata to merge over anything the front matter declared.
    :param title: An explicit title, which wins over the front matter's.
    :returns: The new document id.
    """
    meta, body = _front_matter(text)
    declared = str(meta.pop("title", ""))
    heading = next((l.lstrip("# ").strip() for l in body.splitlines() if l.startswith("#")), "")
    return store.add_document(title or declared or heading or Path(name).stem,
                              body.strip(), {"source": name, **meta, **(extra or {})})


# ------------------------------------------------------------ models
@app.get("/api/models")
def get_models():
    return {"models": config.redact(MODELS), "mock": all(m.get("provider") == "mock" for m in MODELS.values())}


class ModelsIn(BaseModel):
    models: dict


@app.put("/api/models")
def put_models(body: ModelsIn):
    for role, cfg in body.models.items():
        if role not in config.DEFAULT_MODELS:
            continue
        if str(cfg.get("api_key", "")).startswith("••••"):
            cfg["api_key"] = MODELS.get(role, {}).get("api_key")
        MODELS[role] = {k: v for k, v in cfg.items() if v not in (None, "")}
    return {**get_models(), "saved": config.save_models(MODELS)}


@app.get("/api/models/health")
async def models_health():
    return await providers.health(MODELS)


class RoleCheckIn(BaseModel):
    role: str
    config: dict


@app.post("/api/models/test")
async def models_test(body: RoleCheckIn):
    """Check one role against a config that has not been saved, so a bad endpoint is caught before
    it is written to raglab.config.json and before three other roles are disturbed."""
    if body.role not in config.DEFAULT_MODELS:
        raise HTTPException(404, f"No model role named {body.role!r}")
    cfg = dict(body.config)
    if str(cfg.get("api_key", "")).startswith("••••"):  # the UI never sees the real key
        cfg["api_key"] = MODELS.get(body.role, {}).get("api_key")
    return (await providers.health({body.role: cfg}, roles=(body.role,)))[body.role]


@app.get("/api/models/ollama")
async def ollama_models(base_url: str = "http://localhost:11434"):
    try:
        r = await providers.client().get(base_url.rstrip("/") + "/api/tags", timeout=4)
        return {"models": [m["name"] for m in r.json().get("models", [])]}
    except Exception as e:  # noqa: BLE001
        return {"models": [], "error": str(e)}


# ------------------------------------------------------------ corpus
#: Characters of context shown either side of a content match.
SNIPPET_PAD = 90


def _snippet(text: str, needle: str) -> str | None:
    """The first match with a little context, so a content hit can be judged without opening it."""
    at = text.lower().find(needle.lower())
    if at < 0:
        return None
    start, end = max(0, at - SNIPPET_PAD), min(len(text), at + len(needle) + SNIPPET_PAD)
    return ("…" if start else "") + " ".join(text[start:end].split()) + ("…" if end < len(text) else "")


@app.get("/api/documents")
def list_documents(q: str = ""):
    """The corpus, optionally narrowed by a search over titles and document text.

    Searching the text has to happen here: the list deliberately returns `length(text)` rather than
    the text itself, so the client has nothing to match against. A matching document comes back with
    a snippet and where the match was, which is the difference between "this one matched somehow"
    and knowing why.
    """
    needle = q.strip()
    docs = store.documents(with_text=bool(needle))
    if not needle:
        return docs
    hits = []
    for d in docs:
        text, title = d.pop("text", ""), d.get("title") or ""
        in_title = needle.lower() in title.lower()
        snippet = _snippet(text, needle)
        if in_title or snippet:
            hits.append({**d, "match": "title" if in_title and not snippet else "text" if snippet and not in_title
                         else "title and text", "snippet": snippet})
    return hits


@app.get("/api/documents/{did}")
def get_document(did: str):
    doc = store.document(did)
    if not doc:
        raise HTTPException(404)
    return doc


#: Metadata keys the editor will not touch. `source` is provenance — the filename a document came
#: from — and `corpus` is which collection it belongs to, which is changed by choosing a dataset,
#: not by typing over a field. Keeping one way to move a document is what stops the two disagreeing.
READONLY_METADATA = ("source", "corpus")

#: Ceilings on an edited field. Metadata is copied onto every chunk of a document at build time, so
#: a runaway value is paid for once per chunk, and the self-query prompt lists these fields.
#: The value ceiling applies to text someone typed. A value being carried through untouched was
#: already accepted at ingest, and refusing it would make a document with a long front-matter
#: field impossible to edit at all — with the error naming a field the user never opened.
MAX_METADATA_KEYS = 40
MAX_METADATA_KEY = 60
MAX_METADATA_VALUE = 400
MAX_KEPT_VALUE = 20_000
MAX_TITLE = 300


def metadata_value(raw: str):
    """Read one edited metadata value the way a reader would expect a form field to be read.

    Deliberately narrower than YAML. A whole YAML parse would turn `no` into False and `on` into
    True, which is defensible in front matter and startling in a text box. The rule is: something
    that looks like a number is a number, `true`/`false` is a boolean, a comma-separated or
    bracketed list is a list, and everything else is text.

    :param raw: What was typed.
    :returns: A JSON-storable scalar, or a list of them.
    """
    text = raw.strip()
    if not text:
        return ""
    inner = text[1:-1].strip() if text.startswith("[") and text.endswith("]") else None
    if inner is not None or "," in text:
        members = [metadata_value(part) for part in (inner if inner is not None else text).split(",")]
        members = [m for m in members if m != ""]
        return members or text
    if text.lower() in ("true", "false"):
        return text.lower() == "true"
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def _kept_value(value) -> tuple[object, str | None]:
    """Check a value being carried through unchanged, rather than read from typed text.

    :param value: A scalar or a list of scalars, exactly as it is already stored.
    :returns: ``(value, problem)``; problem is None when it can be stored.
    """
    members = value if isinstance(value, list) else [value]
    if any(not isinstance(m, (str, int, float, bool)) for m in members):
        return value, "can only hold text, numbers and true/false"
    if len(json.dumps(value)) > MAX_KEPT_VALUE:
        return value, f"is too large to store ({MAX_KEPT_VALUE} characters at most)"
    return value, None


def _clean_metadata(raw: dict, keep: dict) -> tuple[dict, list[dict]]:
    """Check an edited metadata mapping and read its values.

    A value arrives one of two ways, and the difference matters. A string is text someone typed,
    and is read by :func:`metadata_value`. A ``{"keep": value}`` wrapper is a field the editor
    showed but nobody touched, and is stored exactly as it came.

    Without that distinction every save re-guessed the type of every field from its rendering, so
    editing one field quietly rewrote the others: `version: "2.0"` became the float 2.0,
    `doc_key: "007"` became 7, and `"teachers, principals"` — one string — became two.

    :param raw: ``{key: typed text | {"keep": stored value}}`` as sent by the editor.
    :param keep: The document's current metadata, for the keys the editor may not set.
    :returns: ``(metadata, errors)``; metadata is complete, read-only keys included.
    :raises: Nothing — every problem comes back as a ``{"field", "message"}``.
    """
    errors: list[dict] = []
    if len(raw) > MAX_METADATA_KEYS:
        errors.append({"field": "metadata", "message": f"at most {MAX_METADATA_KEYS} fields, got {len(raw)}"})
    out: dict = {}
    for key, value in raw.items():
        name = str(key).strip()
        if not name:
            errors.append({"field": key, "message": "a field needs a name"})
        elif len(name) > MAX_METADATA_KEY:
            errors.append({"field": name, "message": f"name must be under {MAX_METADATA_KEY} characters"})
        elif name in READONLY_METADATA:
            errors.append({"field": name, "message": f"{name!r} is set by the lab and cannot be edited here"})
        elif name in out:
            errors.append({"field": name, "message": "used twice. Field names have to be unique."})
        elif isinstance(value, dict) and set(value) == {"keep"}:
            kept, problem = _kept_value(value["keep"])
            if problem:
                errors.append({"field": name, "message": problem})
            elif kept != "":
                out[name] = kept
        elif not isinstance(value, str):
            errors.append({"field": name, "message": f"must be text, got {type(value).__name__}"})
        elif len(value) > MAX_METADATA_VALUE:
            errors.append({"field": name, "message": f"value must be under {MAX_METADATA_VALUE} characters"})
        else:
            read = metadata_value(value)
            if read != "":
                out[name] = read        # an emptied field is a removed field
    for key in READONLY_METADATA:
        if key in keep:
            out[key] = keep[key]
    return out, errors


class DocEdit(BaseModel):
    title: str
    #: Every editable field, as typed. Sent whole rather than as a patch: a field the editor left
    #: out has been deleted, and a merge cannot express that.
    metadata: dict = {}


@app.patch("/api/documents/{did}")
def edit_document(did: str, body: DocEdit):
    """Change a document's title and metadata. Its text is never touched.

    Uploaded PDFs arrive with no metadata at all, because there is no front matter to read, which
    left them unfilterable and invisible to self-query with no way to fix it.

    Editing here does not reach an index that was already built: a build copies each document's
    metadata onto its chunks, and that copy is what self-query filters on. The response says which
    indexes still hold the old values so the page can too.

    :param did: Document id.
    :param body: The new title and the complete set of editable fields.
    :returns: The updated document, plus ``stale`` — indexes still holding the previous metadata.
    :raises HTTPException: 404 if there is no such document, 422 with per-field messages.
    """
    doc = store.document(did)
    if not doc:
        raise HTTPException(404, "No such document. It may have been removed.")
    title = body.title.strip()
    errors = []
    if not title:
        errors.append({"field": "title", "message": "a document needs a title"})
    elif len(title) > MAX_TITLE:
        errors.append({"field": "title", "message": f"must be under {MAX_TITLE} characters"})
    metadata, problems = _clean_metadata(body.metadata, doc["metadata"])
    errors += problems
    if errors:
        raise HTTPException(422, {"message": "Some fields could not be saved.", "errors": errors})
    store.update_document(did, title, metadata)
    # No cache to invalidate, and that is the point worth being clear about: an index stores its
    # own copy of each chunk's metadata, taken at build time, and nothing short of a rebuild
    # changes it. `stale` names the indexes still answering with the old values.
    return {**store.document(did), "stale": document_usage(did)}


class DocIn(BaseModel):
    title: str = ""
    text: str
    metadata: dict = {}
    #: Dataset id to file this under. Empty means the unassigned bucket, which is what every
    #: caller written before datasets existed sends.
    dataset: str = ""


@app.post("/api/documents")
def add_document(doc: DocIn):
    """Store a pasted document. The typed title wins over any the pasted text declares, and the
    chosen dataset wins over any `corpus:` the text declares."""
    into = _destination(doc.dataset)
    extra = {**doc.metadata, **({"corpus": into} if into else {})}
    return {"id": _ingest(doc.title or "pasted", doc.text, extra, title=doc.title), "dataset": into}


#: Why a file produced nothing, keyed by whether it was a PDF. Worth spelling out for a PDF: a
#: scanned one looks identical to a working one, and "nothing happened" reads as a broken feature.
NO_TEXT = {
    True: "No text could be extracted. A scanned PDF holds page images rather than a text layer, "
          "so it needs to be run through OCR before it can be indexed.",
    False: "The file holds no text.",
}


def _extract(name: str, raw: bytes) -> str:
    """The text of an uploaded file: pypdf for `.pdf`, UTF-8 for everything else.

    :param name: Filename, which decides how the bytes are read.
    :param raw: The uploaded bytes.
    :returns: Extracted text, which may legitimately be empty.
    :raises Exception: Whatever pypdf raises on a file it cannot open; the caller reports it
        against that one file rather than failing the whole upload.
    """
    if not name.lower().endswith(".pdf"):
        return raw.decode("utf-8", errors="replace")
    from pypdf import PdfReader
    return "\n\n".join((page.extract_text() or "") for page in PdfReader(io.BytesIO(raw)).pages)


@app.post("/api/documents/upload")
async def upload(files: list[UploadFile] = File(...), dataset: str = ""):
    """Import uploaded files into a dataset, reporting what happened to each one.

    Returning only the ids that worked made a failed import indistinguishable from a broken
    feature: a scanned PDF yields no text, was dropped without a word, and the page said nothing at
    all. One file failing does not stop the rest.

    :param files: The uploaded files.
    :param dataset: Dataset id to file them under; empty means the unassigned bucket. Checked
        before any file is read, so a stale id fails outright instead of scattering half an import
        into the wrong collection.
    :returns: ``ids`` of the documents created, ``dataset`` the collection they went to, and
        ``results`` — one ``{filename, ok, id | reason}`` per file, in the order they were sent.
    """
    into = _destination(dataset)
    # Merged after the front matter by `_ingest`, so an explicit destination beats a `corpus:`
    # the document declares about itself. Everything else it declares is kept.
    extra = {"corpus": into} if into else None
    ids: list[str] = []
    results: list[dict] = []
    for f in files:
        name = f.filename or "unnamed"
        try:
            text = _extract(name, await f.read())
        except Exception as exc:
            log.warning("upload: could not read %s (%s bytes declared): %s", name, f.size, exc)
            results.append({"filename": name, "ok": False, "reason": f"Could not be read: {exc}"})
            continue
        if not text.strip():
            results.append({"filename": name, "ok": False, "reason": NO_TEXT[name.lower().endswith(".pdf")]})
            continue
        try:
            ids.append(_ingest(name, text, extra))
        except Exception as exc:
            log.warning("upload: could not store %s: %s", name, exc)
            results.append({"filename": name, "ok": False, "reason": f"Could not be stored: {exc}"})
            continue
        results.append({"filename": name, "ok": True, "id": ids[-1]})
    return {"ids": ids, "dataset": into, "results": results}


SAMPLES = Path(__file__).parent / "sample_corpus"
SAMPLE_INFO = {
    "halcyon": "Halcyon Trailworks (25 docs): an outdoor-gear retailer. Conflicting policy versions, product aliases, acronyms, "
               "lexical traps and near-duplicate SOPs, with a 10-question golden set and normalisation rules.",
    "kestrel": "Kestrel parcel platform (8 docs): cross-referencing specs, a runbook and an incident review. Good for graph and multi-hop.",
}


@app.get("/api/samples")
def samples():
    """Each sample corpus, and how many of its documents the corpus already holds.

    ``loaded`` is counted here, against the sample's own filenames, rather than on the page by
    counting documents tagged with the corpus name: a document of the user's own can carry
    `corpus: kestrel` in its front matter, and counting those would make the page report a sample
    as complete — and disable its button — while some of its documents were still missing.
    """
    have = {d["metadata"].get("source") for d in store.documents()}
    out = []
    for name in sorted(SAMPLE_INFO):
        folder = SAMPLES / name
        if not folder.is_dir():
            continue
        files = [p.name for p in folder.glob("*.md")]
        out.append({"name": name, "about": SAMPLE_INFO.get(name, ""), "docs": len(files),
                    "loaded": sum(1 for f in files if f in have),
                    "golden": (folder / "golden.json").exists()})
    return out


@app.post("/api/documents/sample")
def load_sample(name: str = "kestrel"):
    folder = SAMPLES / name
    if not folder.is_dir():
        raise HTTPException(404, f"No sample corpus named {name}")
    have = {d["metadata"].get("source") for d in store.documents()}
    files = sorted(folder.glob("*.md"))
    fresh = [p for p in files if p.name not in have]
    ids = [_ingest(p.name, p.read_text(), {"corpus": name}) for p in fresh]
    if name == "halcyon" and not store.q("SELECT 1 FROM kv WHERE key='normalization'"):
        store.x("INSERT OR REPLACE INTO kv VALUES ('normalization', ?)", (json.dumps(normalize.HALCYON_RULES),))
    # `skipped` is how the page can say "already loaded" instead of leaving a button that looks
    # like it did nothing the second time it is pressed.
    return {"ids": ids, "added": len(ids), "skipped": len(files) - len(fresh)}


# ------------------------------------------------------------ datasets
#: The key for documents that never had a collection. Spelled the same in the stored config the
#: Chunking page remembers, so an existing `raglab.chunking.corpus` of "uploaded" still resolves.
UNASSIGNED = "uploaded"

#: Longest dataset name accepted. Long enough for a real label, short enough to render in a
#: dropdown without pushing the page sideways.
MAX_DATASET_NAME = 80


def _reserved_names() -> set[str]:
    """Collection names a dataset may not take, lowercased.

    The sample corpora already own theirs; `uploaded` and `unassigned` name the bucket for
    documents that were never assigned one; `all` and `all documents` are how both pages spell
    "no filter". A dataset called any of these would be indistinguishable from what it shadows.
    """
    return {UNASSIGNED, "unassigned", "all", "all documents", *(s.lower() for s in SAMPLE_INFO)}


def _dataset_name_error(name: str) -> str | None:
    """Why ``name`` cannot be used for a new dataset, or None when it can.

    :param name: The name as typed, already trimmed.
    :returns: A message to put next to the field, or None.
    """
    if not name:
        return "Give the dataset a name."
    if len(name) > MAX_DATASET_NAME:
        return f"Keep the name under {MAX_DATASET_NAME} characters ({len(name)} given)."
    if name.lower() in _reserved_names():
        return f"\u201c{name}\u201d is the name of a built-in collection. Choose another."
    if any(d["name"].lower() == name.lower() for d in store.datasets()):
        return f"There is already a dataset called \u201c{name}\u201d."
    return None


def _destination(dataset: str) -> str | None:
    """The collection name an import should write, resolved from a dataset id.

    Called before anything is read or stored: a stale id has to fail the whole request rather than
    scatter half an import into the wrong collection.

    :param dataset: A dataset id, or empty for the unassigned bucket.
    :returns: The dataset's name, or None to leave `corpus` unset.
    :raises HTTPException: 404 when the id names no dataset.
    """
    if not dataset:
        return None
    row = store.dataset(dataset)
    if not row:
        raise HTTPException(404, f"No dataset with id {dataset!r}. It may have been removed \u2014 "
                                 "pick another destination and try again.")
    return row["name"]


class DatasetIn(BaseModel):
    name: str = ""


@app.get("/api/datasets")
def list_datasets():
    """User-created datasets, oldest first, with how many documents each holds."""
    counts = _collection_counts()
    return [{**d, "docs": counts.get(d["name"].casefold(), ("", 0))[1]} for d in store.datasets()]


@app.post("/api/datasets")
def create_dataset(body: DatasetIn):
    """Register a dataset so it can be chosen as an upload destination.

    A dataset exists before it holds anything, which is the point: you name it, then upload into
    it, and it is still there after a restart if you never did.

    :param body: ``{"name": ...}``; surrounding whitespace is trimmed.
    :returns: The stored dataset, with a `docs` count of 0.
    :raises HTTPException: 422 with a message for the field when the name cannot be used.
    """
    name = body.name.strip()
    if (problem := _dataset_name_error(name)):
        raise HTTPException(422, problem)
    try:
        did = store.add_dataset(name)
    except sqlite3.IntegrityError:
        # Two requests raced for the same name and the unique index settled it.
        raise HTTPException(422, f"There is already a dataset called \u201c{name}\u201d.")
    return {**store.dataset(did), "docs": 0}


def collection_of(doc: dict) -> str:
    """The collection a document belongs to, as written.

    Mirrored in the two pages that filter by it (`Corpus.tsx`, `Indexes.tsx`); the comparison is
    case-insensitive on all three sides, so keep them in step.
    """
    return str(doc["metadata"].get("corpus") or "").strip() or UNASSIGNED


def _collection_counts() -> dict[str, tuple[str, int]]:
    """How many documents each collection holds, grouped without regard to case.

    Grouping is case-insensitive because the dataset name check is: `Engineering docs` is refused
    as a duplicate of `engineering docs`, so the two cannot be allowed to appear as separate
    collections either. They used to, which split one collection into two near-identical rows in
    both dropdowns — you would upload into one and filter by the other.

    :returns: ``{casefolded key: (name as displayed, count)}``.
    """
    counts: dict[str, tuple[str, int]] = {}
    for doc in store.documents():
        name = collection_of(doc)
        shown, seen = counts.get(name.casefold(), (name, 0))
        counts[name.casefold()] = (shown, seen + 1)
    return counts


@app.get("/api/collections")
def collections():
    """Every collection a document can be in: user datasets, sample corpora, and what came before.

    One endpoint rather than a list each page assembles for itself. The Corpus filter and the
    Chunking selector have to agree about what exists and how many documents are in it, and two
    screens computing the same thing separately have already drifted apart twice in this codebase.

    A document belongs to the collection named by its `metadata.corpus`, which is the single
    answer for all three kinds. A dataset whose name matches a `corpus:` value that documents
    already carry therefore adopts them, rather than shadowing them with a second empty group.

    :returns: One ``{key, label, kind, dataset_id, docs}`` per collection. `kind` is `dataset`,
        `sample` or `legacy`; `docs` is 0 for a dataset nothing has been uploaded into yet.
    """
    counts = _collection_counts()
    # A dataset's own spelling is the one shown, whatever case the documents happen to carry.
    out = [{"key": d["name"], "label": d["name"], "kind": "dataset", "dataset_id": d["id"],
            "docs": counts.pop(d["name"].casefold(), ("", 0))[1]} for d in store.datasets()]
    out += [{"key": counts[n][0], "label": counts[n][0], "kind": "sample", "dataset_id": None,
             "docs": counts.pop(n)[1]} for n in sorted(SAMPLE_INFO) if n in counts]
    out += [{"key": shown, "label": "Uploaded / unassigned" if shown == UNASSIGNED else shown,
             "kind": "legacy", "dataset_id": None, "docs": n}
            for _, (shown, n) in sorted(counts.items())]
    return out


def document_usage(did: str) -> list[dict]:
    """Indexes holding a stored copy of this document's chunks.

    An index is a snapshot: it keeps its own chunk text and embeddings, so it survives the document
    being deleted. Knowing which ones is what makes "Remove" explainable rather than surprising.
    """
    rows = store.q("SELECT c.index_id AS id, count(*) AS chunks FROM chunks c WHERE c.doc_id=? AND c.kind='chunk' "
                   "GROUP BY c.index_id", (did,))
    names = {i["id"]: i["name"] for i in store.indexes()}
    return [{"index_id": r["id"], "name": names.get(r["id"], r["id"]), "chunks": r["chunks"]}
            for r in rows if r["id"] in names]


@app.get("/api/documents-usage")
def all_document_usage():
    """Index membership for every document, in one query rather than one request per row."""
    names = {i["id"]: i["name"] for i in store.indexes()}
    out: dict[str, list[dict]] = {}
    for r in store.q("SELECT doc_id, index_id, count(*) AS chunks FROM chunks WHERE kind='chunk' "
                     "GROUP BY doc_id, index_id"):
        if r["index_id"] in names:
            out.setdefault(r["doc_id"], []).append(
                {"index_id": r["index_id"], "name": names[r["index_id"]], "chunks": r["chunks"]})
    return out


@app.get("/api/documents/{did}/usage")
def get_document_usage(did: str):
    if not store.document(did):
        raise HTTPException(404)
    return {"indexes": document_usage(did)}


@app.delete("/api/documents/{did}")
def delete_document(did: str):
    """Remove a document from the corpus, and stop it being retrieved.

    The stored chunks stay where they are — they cannot be pulled out of an index in place, because
    every chunk's ordinal is a position in the embedding matrix, the BM25 index and the HNSW graph.
    They are excluded at query time instead (see Runtime.live_mask), so the document stops answering
    questions immediately, and the affected indexes are named here so the caller can say what
    happened and offer a rebuild.
    """
    if not store.document(did):
        raise HTTPException(404)
    affected = document_usage(did)
    store.x("DELETE FROM documents WHERE id=?", (did,))
    # Loaded runtimes computed their exclusion mask when they were built, so they have to be
    # rebuilt or they would keep serving the document they no longer should.
    indexer._RUNTIME.clear()
    return {"ok": True, "indexes": affected,
            "chunks_excluded": sum(i["chunks"] for i in affected),
            "note": (f"Removed from the corpus and excluded from retrieval. "
                     f"{'One index still holds' if len(affected) == 1 else f'{len(affected)} indexes still hold'} "
                     f"a stored copy — rebuild {'it' if len(affected) == 1 else 'them'} to drop it."
                     if affected else "Removed from the corpus. No index had been built from it.")}


# ------------------------------------------------------------ indexes
class IndexIn(BaseModel):
    name: str = ""
    config: dict = {}


@app.get("/api/defaults")
def defaults():
    return {"index": indexer.DEFAULT_INDEX, "pipeline": DEFAULT_PIPELINE, "presets": PIPELINE_PRESETS,
            "bounds": {k: {"min": lo, "max": hi, "int": is_int} for k, (lo, hi, is_int) in PIPELINE_BOUNDS.items()}}


class CheckIn(BaseModel):
    config: dict = {}


@app.post("/api/pipeline/check")
def pipeline_check(body: CheckIn):
    """Field-level errors and inactive-control warnings for a config, without running anything."""
    cfg = merged_pipeline(body.config)
    return {"errors": pipeline_validate(cfg), "warnings": pipeline_inactive(cfg), "calls": estimate_calls(cfg)}


def estimate_calls(cfg: dict) -> dict:
    """Roughly how many model requests one question will cost, so a preset can say so up front.

    Counts requests, not tokens: on a local server the request count is what the wait is made of.
    """
    n = {"router": 0, "generator": 0, "reranker": 0, "embedding": 0}
    if cfg["route"]["enabled"]:
        n["router"] += 1
    if cfg["multi_query"]["enabled"]:
        n["router"] += 1
    if cfg["decompose"]["enabled"]:
        n["router"] += 1
    if cfg["step_back"]["enabled"]:
        n["router"] += 1
    if cfg["self_query"]["enabled"]:
        n["router"] += 1
    if cfg["hyde"]["enabled"]:
        n["generator"] += 1
    if cfg["dense"]["enabled"]:
        n["embedding"] += 1
    candidates = cfg["fusion"]["top_n"]
    if cfg["rerank"]["enabled"]:
        n["reranker"] += candidates  # one request per candidate for the `llm` reranker; batched otherwise
        candidates = min(candidates, cfg["rerank"]["top_n"])
    if cfg["crag"]["enabled"]:
        n["router"] += candidates
    if cfg["generate"]["enabled"]:
        n["generator"] += 1
    if cfg["verify"]["enabled"]:
        n["router"] += 1
    return {**n, "total": sum(n.values()),
            "note": "The reranker figure assumes the `llm` provider, which scores one passage per request. "
                    "A cross-encoder or a rerank API scores them all in one."}


PIPELINE_PRESETS = [
    {"name": "Fast hybrid baseline",
     "about": "BM25 and dense retrieval, fused, with a cited answer. No extra model calls before retrieval and no "
              "reranking, so a first run finishes in a handful of requests and any latency you see is the generator's.",
     "roles": ["embedding", "generator"],
     "config": {"route": {"enabled": False, "apply": False}, "multi_query": {"enabled": False},
                "hyde": {"enabled": False}, "decompose": {"enabled": False}, "step_back": {"enabled": False},
                "rerank": {"enabled": False}, "crag": {"enabled": False}, "verify": {"enabled": False},
                "dense": {"compare_exact": False}}},
    {"name": "Reranked",
     "about": "Adds query variants and a reranker over the fused candidates. Lower fusion.top_n to keep the cost "
              "down when the reranker is an LLM.",
     "roles": ["embedding", "router", "reranker", "generator"],
     "config": {"route": {"enabled": False, "apply": False}, "multi_query": {"enabled": True, "n": 3},
                "rerank": {"enabled": True, "top_n": 8}, "fusion": {"top_n": 12}, "crag": {"enabled": False},
                "verify": {"enabled": False}}},
    {"name": "Everything on",
     "about": "Routing, query transformation, reranking, corrective grading and a groundedness check. The slowest "
              "and most expensive path: read the waterfall to see where the time goes.",
     "roles": ["embedding", "router", "reranker", "generator"],
     "config": {"route": {"enabled": True, "apply": True}, "multi_query": {"enabled": True, "n": 3},
                "rerank": {"enabled": True, "top_n": 8}, "crag": {"enabled": True}, "verify": {"enabled": True}}},
]


@app.get("/api/indexes")
def list_indexes():
    return [{**i, "job": indexer.JOBS.get(i["id"])} for i in store.indexes()]


@app.post("/api/indexes")
async def create_index(body: IndexIn):
    cfg = _validated_index(indexer.merged_config(body.config))
    name = body.name.strip() or indexer.suggest_name(
        cfg, MODELS["embedding"].get("model", ""), [i["name"] for i in store.indexes()])
    iid = store.create_index(name, cfg)
    _start_build(iid)
    return {"id": iid, "name": name}


@app.post("/api/indexes/{iid}/stop")
def index_stop(iid: str):
    """Stop a build in progress. A half-built index is marked failed rather than left pending."""
    return {"stopped": _stop(iid)}


@app.delete("/api/indexes/{iid}")
def delete_index(iid: str):
    indexer._RUNTIME.pop(iid, None)
    store.delete_index(iid)
    return {"ok": True}


def _rt(iid: str, embedding: bool = False):
    """Load an index runtime, optionally asserting it can still be queried with today's models.

    :param iid: Index id.
    :param embedding: Check that the embedding role still matches the one the index was built with.
        Every path that embeds a query needs this, not just the Workbench: a different model of the
        same width produces neighbours that look plausible and mean nothing.
    :raises HTTPException: 409 when the index is not ready, or when the embedding role has moved.
    """
    try:
        rt = indexer.runtime(store, iid)
    except ValueError as e:
        raise HTTPException(409, str(e)) from e
    if embedding:
        _check_embedding(rt)
    return rt


def _check_embedding(rt) -> None:
    """The API's view of :func:`raglab.indexer.compatibility`. One implementation, two callers: the
    pipeline raises, this turns the same answer into a 409."""
    if (problems := indexer.compatibility(rt, MODELS)):
        raise HTTPException(409, " ".join(problems))


def _models_snapshot() -> dict:
    """A frozen copy of the model roles for a job to run under.

    MODELS is mutable and edited live from the Models page. A queued job that reads it at execution
    time can run under a model that never passed the compatibility check the user watched succeed —
    and the batch permit makes that window longer, not shorter.
    """
    return copy.deepcopy(MODELS)


def _validated(cfg: dict) -> dict:
    """Reject a pipeline config with out-of-range values before it reaches a model or a native call."""
    errors = pipeline_validate(merged_pipeline(cfg))
    if errors:
        raise HTTPException(422, {"message": "Some pipeline settings are out of range.", "errors": errors})
    return cfg


def _validated_index(cfg: dict) -> dict:
    """Reject a build config the same way. A bad value here used to be accepted, then kill the build
    thread with a native type error and leave the index sitting at 'failed' with no usable reason."""
    errors = indexer.validate_index(cfg)
    if errors:
        raise HTTPException(422, {"message": "Some index settings are out of range.", "errors": errors})
    return cfg


@app.get("/api/indexes/{iid}")
def index_detail(iid: str):
    rt = _rt(iid)
    graph = None
    if rt.graph:
        top = sorted(rt.graph.nodes.items(), key=lambda kv: -len(kv[1]["chunks"]))[:60]
        keep = {k for k, _ in top}
        graph = {"nodes": [{"id": k, "label": v["label"], "type": v["type"], "chunks": len(v["chunks"])} for k, v in top],
                 "edges": [e for e in rt.graph.edges if e["s"] in keep and e["t"] in keep][:200]}
    docs = {}
    for c in rt.chunks:
        docs.setdefault(c["doc_id"], {"id": c["doc_id"], "title": c["meta"].get("title", ""),
                                      "meta": {k: v for k, v in c["meta"].items() if k not in ("section", "title", "source")}})
    return {"id": iid, "name": rt.name, "created": rt.created, "config": rt.config, "stats": rt.stats, "schema": rt.schema(), "graph": graph,
            "removed_documents": {"documents": len(rt.removed_docs), "chunks": int(rt.removed.sum())},
            "docs": list(docs.values()), "pca_variance": round(rt.pca_var, 4),
            "communities": [{k: c[k] for k in ("id", "title", "summary", "entities")} for c in rt.communities],
            "truncated": max(0, len(rt.chunks) - DETAIL_LIMIT), "detail_limit": DETAIL_LIMIT,
            "projection": [{"i": c["i"], "xy": xy, "doc_id": c["doc_id"]}
                           for c, xy in zip(rt.chunks, rt.project(rt.matrix))][:DETAIL_LIMIT],
            "chunks": [{"i": c["i"], "doc_id": c["doc_id"], "title": c["meta"].get("title", ""),
                        "section": c["meta"].get("section", ""), "text": c["text"], "context": c["context"],
                        "tokens": len(c["text"]) // 4, "parent": c["parent"], "start": c["start"], "end": c["end"]}
                       for c in rt.chunks[:DETAIL_LIMIT]]}


#: How much of an index the detail endpoint carries in one response. Beyond this the chunks and the
#: projection are a page, and the rest is fetched on demand.
DETAIL_LIMIT = 5000


@app.get("/api/indexes/{iid}/chunks")
def index_chunks(iid: str, offset: int = 0, limit: int = 200, ids: str = ""):
    """A page of chunks, or the specific ones named in `ids`.

    The detail endpoint stops at :data:`DETAIL_LIMIT`, but an exact-search probe ranges over the
    whole matrix, so a neighbour past that point had no record to show. This is how to fetch it.
    """
    rt = _rt(iid)
    if ids:
        wanted = [int(x) for x in ids.split(",") if x.strip().lstrip("-").isdigit()]
        picked = [rt.chunks[i] for i in wanted if 0 <= i < len(rt.chunks)]
        missing = [i for i in wanted if not 0 <= i < len(rt.chunks)]
    else:
        if not 0 <= offset <= len(rt.chunks):
            raise HTTPException(416, f"This index has {len(rt.chunks)} chunks, so offset {offset} is outside it.")
        picked, missing = rt.chunks[offset:offset + max(1, min(limit, 1000))], []
    return {"total": len(rt.chunks), "offset": offset, "missing": missing,
            "chunks": [{"i": c["i"], "doc_id": c["doc_id"], "title": c["meta"].get("title", ""),
                        "section": c["meta"].get("section", ""), "text": c["text"], "context": c["context"],
                        "tokens": len(c["text"]) // 4, "parent": c["parent"], "start": c["start"], "end": c["end"]}
                       for c in picked]}


@app.get("/api/indexes/{iid}/umap")
async def index_umap(iid: str):
    rt = _rt(iid)
    try:
        _, pts = await asyncio.to_thread(rt.umap)
    except ValueError as e:
        raise HTTPException(501, str(e)) from e
    return {"projection": [{"i": c["i"], "xy": xy, "doc_id": c["doc_id"]} for c, xy in zip(rt.chunks, pts)]}


class ProbeIn(BaseModel):
    text: str = ""
    i: int | None = None
    k: int = 6
    method: str = "pca"


@app.post("/api/indexes/{iid}/probe")
async def probe(iid: str, body: ProbeIn):
    """Nearest neighbours (exact cosine) of a chunk or of free text embedded as a query."""
    rt = _rt(iid, embedding=body.i is None)  # only the free-text path embeds with today's model
    if body.i is not None and not 0 <= body.i < len(rt.chunks):
        raise HTTPException(404, f"This index has {len(rt.chunks)} chunks, so #{body.i} is not one of them.")
    try:
        if body.i is not None:
            vec, xy = rt.matrix[body.i], None
        else:
            vec = (await providers.embed(MODELS["embedding"], [body.text], "query"))[0]
            xy = (await asyncio.to_thread(rt.project, vec[None, :], body.method))[0]
    except (providers.ProviderError, ValueError) as e:
        raise HTTPException(502, str(e)) from e
    hits = [(i, s) for i, s in rt.vectors.exact(vec, body.k + 1) if i != body.i][:body.k]
    return {"xy": xy, "neighbours": [{"i": i, "score": round(s, 4)} for i, s in hits]}


class HnswIn(BaseModel):
    M: int = 16
    ef_construction: int = 200


@app.post("/api/indexes/{iid}/hnsw")
async def rebuild_hnsw(iid: str, body: HnswIn):
    rt = _rt(iid)
    _validated_index({**rt.config, "hnsw": body.model_dump()})
    return await asyncio.to_thread(indexer.rebuild_hnsw, store, iid, body.model_dump())


class PreviewIn(BaseModel):
    doc_id: str
    chunking: dict


@app.post("/api/chunk-preview")
async def chunk_preview(body: PreviewIn):
    doc = store.document(body.doc_id)
    if not doc:
        raise HTTPException(404)
    _validated_index(indexer.merged_config({"chunking": body.chunking}))
    try:
        res = await chunk_text(doc["text"], body.chunking, lambda t: providers.embed(MODELS["embedding"], t, "doc"))
    except providers.ProviderError as e:
        raise HTTPException(502, str(e)) from e
    return {"text": doc["text"], **res}


# ------------------------------------------------------------ query
class QueryIn(BaseModel):
    index_id: str
    query: str
    config: dict = {}
    history: list[dict] = []
    control: bool = False  # bypass the cache and stop routing rewriting settings, for A/B runs


@app.post("/api/query")
async def query(body: QueryIn):
    rt = _rt(body.index_id, embedding=True)
    _validated(body.config)
    cfg = pipeline_controlled(body.config) if body.control else body.config
    pipe = Pipeline(rt, MODELS, cfg, store=store, history=body.history)

    async def stream():
        stages = []
        try:
            async for ev in pipe.run(body.query):
                if ev["type"] == "stage":
                    stages.append(ev)
                if ev["type"] == "done":
                    tid = uuid.uuid4().hex[:10]
                    store.x("INSERT INTO traces VALUES (?,?,?,?,?,?)", (tid, rt.id, body.query, json.dumps(pipe.cfg),
                                                                        json.dumps(stages), time.time()))
                    ev = {"type": "done", "trace_id": tid, "total_ms": ev["total_ms"], "usage": ev.get("usage", {})}
                yield f"data: {json.dumps(ev)}\n\n"
        except Exception as e:  # noqa: BLE001
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)[:800]})}\n\n"
    return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


@app.get("/api/traces")
def traces(index_id: str | None = None):
    rows = store.q("SELECT id,index_id,query,created FROM traces " + ("WHERE index_id=? " if index_id else "") +
                   "ORDER BY created DESC LIMIT 50", (index_id,) if index_id else ())
    return rows


@app.get("/api/traces/{tid}")
def trace(tid: str):
    rows = store.q("SELECT * FROM traces WHERE id=?", (tid,))
    if not rows:
        raise HTTPException(404)
    r = rows[0]
    return {**r, "config": json.loads(r["config"]), "trace": json.loads(r["trace"])}


# ------------------------------------------------------------ evaluation
class EvalGenIn(BaseModel):
    index_id: str
    n: int = 12
    name: str = "generated"
    mix: dict = {"simple": 1.0}


@app.post("/api/evalsets/generate")
async def evalset_generate(body: EvalGenIn):
    try:
        return {"id": await evals.generate_set(store, _rt(body.index_id), MODELS, body.n, body.name, body.mix)}
    except providers.ProviderError as e:
        raise HTTPException(502, str(e)) from e


@app.post("/api/evalsets/golden")
def evalset_golden(name: str = "halcyon"):
    f = SAMPLES / name / "golden.json"
    if not f.exists():
        raise HTTPException(404, "That sample has no golden set")
    sid = uuid.uuid4().hex[:8]
    store.x("INSERT INTO evalsets VALUES (?,?,?,?)", (sid, f"{name} golden set", f.read_text(), time.time()))
    return {"id": sid}


class EvalSetIn(BaseModel):
    name: str
    items: list[dict]


@app.post("/api/evalsets")
def evalset_create(body: EvalSetIn):
    sid = uuid.uuid4().hex[:8]
    store.x("INSERT INTO evalsets VALUES (?,?,?,?)", (sid, body.name, json.dumps(body.items), time.time()))
    return {"id": sid}


@app.get("/api/evalsets")
def evalsets():
    return [{**r, "items": json.loads(r["items"])} for r in store.q("SELECT * FROM evalsets ORDER BY created DESC")]


@app.delete("/api/evalsets/{sid}")
def evalset_delete(sid: str):
    for row in store.q("SELECT id FROM evalruns WHERE evalset_id=?", (sid,)):
        evalrun_delete(row["id"])
    store.x("DELETE FROM evalsets WHERE id=?", (sid,))
    store.x("DELETE FROM evalruns WHERE evalset_id=?", (sid,))
    return {"ok": True}


class EvalRunIn(BaseModel):
    evalset_id: str
    index_id: str
    label: str = "run"
    config: dict = {}
    judge: bool = False
    k: int = 10
    control: bool = False  # pin the cache off and stop routing rewriting settings per question
    repeats: int = 1       # time each question this many times and report the median
    warmup: bool = False   # discard the first timing of each question


@app.post("/api/evalruns")
async def evalrun_start(body: EvalRunIn):
    sets = [s for s in evalsets() if s["id"] == body.evalset_id]
    if not sets:
        raise HTTPException(404, "Unknown question set")
    _validated(body.config)
    rid = uuid.uuid4().hex[:8]
    _track(_batch(evals.run_eval(store, rid, _rt(body.index_id, embedding=True), _models_snapshot(),
                                 sets[0], body.config, body.label, body.judge, body.k, body.control,
                                 body.repeats, body.warmup)), rid)
    return {"id": rid}


@app.get("/api/evalruns")
def evalruns(evalset_id: str):
    rows = store.q("SELECT * FROM evalruns WHERE evalset_id=? ORDER BY created", (evalset_id,))
    done = [{**r, "config": json.loads(r["config"]), "metrics": json.loads(r["metrics"]), "rows": json.loads(r["rows"]),
             "manifest": json.loads(r["manifest"]) if r.get("manifest") else None} for r in rows]
    return {"runs": done, "jobs": evals.RUNS}


@app.get("/api/evalruns/{rid}/manifest")
def evalrun_manifest(rid: str):
    """What produced this run, and what has changed since — so a disagreement between two runs can
    be attributed instead of guessed at."""
    rows = store.q("SELECT index_id, manifest FROM evalruns WHERE id=?", (rid,))
    if not rows:
        raise HTTPException(404)
    saved = json.loads(rows[0]["manifest"]) if rows[0]["manifest"] else None
    try:
        rt = indexer.runtime(store, rows[0]["index_id"])
        # Compare against the same indexes the run named, so a sweep across several is not reported
        # as having lost the ones today's single-index manifest happens not to mention.
        named = [i.get("id") for i in (saved or {}).get("indexes", []) if i.get("id")] or [rt.id]
        loaded = []
        for iid in named:
            try:
                loaded.append(indexer.runtime(store, iid))
            except ValueError:
                pass
        now = evals.manifest(store, rt, MODELS, {}, control=False, indexes=loaded or [rt])
    except ValueError:
        return {"manifest": saved, "drift": ["The index this run used no longer exists."], "reproducible": False}
    notes = evals.drift(saved, now)
    return {"manifest": saved, "drift": notes, "reproducible": not notes}


@app.delete("/api/evalruns/{rid}")
def evalrun_delete(rid: str):
    """Delete a run and the per-question traces it stored, so they do not outlive their results."""
    rows = store.q("SELECT rows FROM evalruns WHERE id=?", (rid,))
    traces = [r.get("trace_id") for r in json.loads(rows[0]["rows"] or "[]")] if rows else []
    for tid in filter(None, traces):
        store.x("DELETE FROM traces WHERE id=?", (tid,))
    store.x("DELETE FROM evalruns WHERE id=?", (rid,))
    return {"ok": True, "traces_removed": len(list(filter(None, traces)))}


@app.post("/api/evalruns/{rid}/stop")
def evalrun_stop(rid: str):
    """Stop a running evaluation. Questions already scored stay: they were checkpointed as they finished."""
    return {"stopped": _stop(rid)}


def _stop(key: str) -> bool:
    task = TASKS.get(key)
    if task is None or task.done():
        return False
    task.cancel()
    return True


# ------------------------------------------------------------ parameter sweeps
class SweepIn(BaseModel):
    evalset_id: str
    index_id: str = ""
    param: str
    values: list
    config: dict = {}
    k: int = 10
    label: str = ""
    repeats: int = 1
    warmup: bool = False
    shared_rules: bool = False  # index sweeps: hold the first index's rules for every index


@app.post("/api/sweeps")
async def sweep_start(body: SweepIn):
    sets = [s for s in evalsets() if s["id"] == body.evalset_id]
    if not sets:
        raise HTTPException(404, "Unknown question set")
    for iid in (body.values if body.param == "index" else [body.index_id]):
        _rt(iid, embedding=True)
    _validated(body.config)
    # Every point, not just the base config: the swept value is the one most likely to be wrong,
    # and an invalid one used to run to completion and plot as a real measurement.
    errors = evals.validate_sweep(body.model_dump())
    if errors:
        raise HTTPException(422, {"message": "This sweep would not measure what it claims.", "errors": errors})
    sid = uuid.uuid4().hex[:8]
    _track(_batch(evals.run_sweep(store, sid, _models_snapshot(), sets[0], body.model_dump())), sid)
    return {"id": sid}


@app.get("/api/sweeps")
def sweeps(evalset_id: str):
    rows = store.q("SELECT * FROM sweeps WHERE evalset_id=? ORDER BY created DESC", (evalset_id,))
    return {"sweeps": [{**r, "spec": json.loads(r["spec"]), "points": json.loads(r["points"]),
                        "manifest": json.loads(r["manifest"]) if r.get("manifest") else None} for r in rows],
            "jobs": evals.SWEEPS}


@app.delete("/api/sweeps/{sid}")
def sweep_delete(sid: str):
    store.x("DELETE FROM sweeps WHERE id=?", (sid,))
    return {"ok": True}


@app.post("/api/sweeps/{sid}/stop")
def sweep_stop(sid: str):
    """Stop a running sweep. Points already measured stay: each is saved as it finishes."""
    return {"stopped": _stop(sid)}


# ------------------------------------------------------------ presets
class PresetIn(BaseModel):
    config: dict


@app.get("/api/presets")
def presets():
    return [{"name": r["name"], "config": json.loads(r["config"])} for r in store.q("SELECT * FROM presets ORDER BY name")]


@app.put("/api/presets/{name}")
def preset_save(name: str, body: PresetIn):
    store.x("INSERT OR REPLACE INTO presets VALUES (?,?,?)", (name, json.dumps(body.config), time.time()))
    return {"ok": True}


@app.delete("/api/presets/{name}")
def preset_delete(name: str):
    store.x("DELETE FROM presets WHERE name=?", (name,))
    return {"ok": True}


# ------------------------------------------------------------ normalisation lab
def _rules(index_id: str | None = None) -> dict:
    """The rules for an index, falling back to the shared set when it has none of its own."""
    for key in filter(None, (normalize.rules_key(index_id) if index_id else None, "normalization")):
        rows = store.q("SELECT value FROM kv WHERE key=?", (key,))
        if rows:
            return {**normalize.EMPTY_RULES, **json.loads(rows[0]["value"])}
    return {**normalize.EMPTY_RULES}


def _rules_scope(index_id: str | None) -> str:
    """Whether an index has its own rules, or is using the shared set."""
    if index_id and store.q("SELECT 1 FROM kv WHERE key=?", (normalize.rules_key(index_id),)):
        return "index"
    return "shared"


@app.get("/api/normalization")
def norm_get(index_id: str | None = None):
    return {"rules": _rules(index_id), "presets": ["halcyon", "empty"], "scope": _rules_scope(index_id),
            "index_id": index_id}


class RulesIn(BaseModel):
    rules: dict
    index_id: str | None = None  # None writes the shared set; an id scopes the rules to that index


@app.put("/api/normalization")
def norm_put(body: RulesIn):
    store.x("INSERT OR REPLACE INTO kv VALUES (?, ?)",
            (normalize.rules_key(body.index_id), json.dumps({**normalize.EMPTY_RULES, **body.rules})))
    return norm_get(body.index_id)


@app.delete("/api/normalization")
def norm_unscope(index_id: str):
    """Drop an index's own rules so it follows the shared set again."""
    store.x("DELETE FROM kv WHERE key=?", (normalize.rules_key(index_id),))
    return norm_get(index_id)


@app.post("/api/normalization/preset")
def norm_preset(name: str, index_id: str | None = None):
    return norm_put(RulesIn(rules=normalize.HALCYON_RULES if name == "halcyon" else normalize.EMPTY_RULES,
                            index_id=index_id))


class NormTestIn(BaseModel):
    index_id: str
    query: str
    rules: dict | None = None


@app.post("/api/normalization/test")
def norm_test(body: NormTestIn):
    rt = _rt(body.index_id)
    res = normalize.normalize(body.query, body.rules or _rules(body.index_id), rt.vocab)
    before, _ = rt.bm25.search(body.query, 5)
    after, _ = rt.bm25.search(res["normalized"], 5)
    show = lambda hits: [{"i": i, "score": round(s, 2), "title": rt.chunks[i]["meta"].get("title", ""),  # noqa: E731
                          "section": rt.chunks[i]["meta"].get("section", ""), "doc_id": rt.chunks[i]["doc_id"]} for i, s in hits]
    return {**res, "bm25_before": show(before), "bm25_after": show(after), "vocabulary": len(rt.vocab)}


@app.post("/api/normalization/suggest")
async def norm_suggest(index_id: str):
    """Ask the router model to propose rules from the corpus itself. Nothing is saved until the user accepts."""
    rt = _rt(index_id)
    step = max(1, len(rt.chunks) // 14)
    sample = "\n---\n".join(f"{c['meta'].get('title', '')}: {c['text'][:420]}" for c in rt.chunks[::step][:14])
    try:
        out = await providers.chat_json(MODELS["router"], [{"role": "user", "content": normalize.SUGGEST.format(sample=sample)}],
                                        task="suggest_rules", max_tokens=1200, default={})
    except providers.ProviderError as e:
        raise HTTPException(502, str(e)) from e
    return normalize.clean_suggestion(out)


@app.delete("/api/cache")
def cache_clear(index_id: str):
    store.x("DELETE FROM qcache WHERE index_id=?", (index_id,))
    return {"ok": True}


# ------------------------------------------------------------ UI
if STATIC.exists():
    app.mount("/assets", StaticFiles(directory=STATIC / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str):
        # The path arrives percent-decoded, so "/..%2f..%2fpyproject.toml" reaches here as
        # "../../pyproject.toml". Joined unchecked, that served any file the user can read.
        f = (STATIC / path).resolve()
        inside = f.is_relative_to(STATIC.resolve())
        return FileResponse(f if path and inside and f.is_file() else STATIC / "index.html")


def cli():
    import argparse, uvicorn
    p = argparse.ArgumentParser(description="RAG Lab Advanced (local only)")
    p.add_argument("--port", type=int, default=8600)
    p.add_argument("--host", default="127.0.0.1")
    a = p.parse_args()
    print(f"RAG Lab Advanced  ->  http://{a.host}:{a.port}")
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    cli()
