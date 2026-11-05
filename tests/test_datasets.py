"""Named document datasets: creating them, importing into them, and building an index from one.

A dataset exists in its own right so an empty one survives a restart. Membership itself stays in
`metadata.corpus`, which is the single answer for sample corpora, user datasets and documents that
predate both — a second membership field would be a second thing to disagree with.
"""
import asyncio
import io

import pytest
from fastapi.testclient import TestClient
from pypdf import PdfWriter

from raglab import indexer
from raglab.main import MODELS, app, store

client = TestClient(app)


@pytest.fixture
def own_datasets():
    """Remove datasets the test registered, so a later module sees the rows it expects."""
    before = {d["id"] for d in store.datasets()}
    yield
    for d in store.datasets():
        if d["id"] not in before:
            store.x("DELETE FROM datasets WHERE id=?", (d["id"],))


def make(name: str) -> dict:
    r = client.post("/api/datasets", json={"name": name})
    assert r.status_code == 200, r.text
    return r.json()


def upload(files, dataset: str = ""):
    url = f"/api/documents/upload{f'?dataset={dataset}' if dataset else ''}"
    return client.post(url, files=[("files", f) for f in files])


def blank_pdf() -> bytes:
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


def collection(key: str) -> dict | None:
    return next((c for c in client.get("/api/collections").json() if c["key"] == key), None)


# ------------------------------------------------------------ creating and keeping
def test_an_empty_dataset_is_kept_and_listed_before_anything_is_uploaded(own_datasets):
    """The reason it is a row of its own: you name it first and upload into it second, and it has
    to still be there in between — including after a restart."""
    made = make("Product manuals")
    assert made["docs"] == 0
    assert [d["name"] for d in client.get("/api/datasets").json() if d["id"] == made["id"]] == ["Product manuals"]
    assert collection("Product manuals") == {"key": "Product manuals", "label": "Product manuals",
                                             "kind": "dataset", "dataset_id": made["id"], "docs": 0}


def test_a_dataset_survives_reopening_the_database(own_datasets, tmp_path):
    """Opening the same file again runs the schema and the migration a second time; neither may
    drop what is already there."""
    from raglab.store import Store
    path = tmp_path / "reopen.sqlite"
    first = Store(path=path)
    did = first.add_dataset("Engineering docs")
    first.db.close()
    again = Store(path=path)
    assert [d["name"] for d in again.datasets()] == ["Engineering docs"]
    assert again.dataset(did)["name"] == "Engineering docs"
    again.db.close()


@pytest.mark.parametrize("name,expect", [
    ("", "Give the dataset a name"),
    ("   ", "Give the dataset a name"),
    ("x" * 81, "under 80 characters"),
    ("uploaded", "built-in collection"),
    ("All documents", "built-in collection"),
    ("kestrel", "built-in collection"),
    ("HALCYON", "built-in collection"),
])
def test_names_that_cannot_be_used_are_refused_with_a_reason(own_datasets, name, expect):
    r = client.post("/api/datasets", json={"name": name})
    assert r.status_code == 422
    assert expect in r.json()["detail"]


def test_a_duplicate_name_is_refused_whatever_its_case(own_datasets):
    make("Engineering docs")
    for attempt in ["Engineering docs", "engineering DOCS", "  Engineering docs  "]:
        r = client.post("/api/datasets", json={"name": attempt})
        assert r.status_code == 422, attempt
        assert "already a dataset" in r.json()["detail"]


def test_a_name_is_trimmed_before_it_is_stored(own_datasets):
    assert make("  Field notes  ")["name"] == "Field notes"


# ------------------------------------------------------------ importing into one
def test_uploads_land_in_the_selected_dataset_and_later_ones_append(own_corpus, own_datasets):
    made = make("Engineering docs")
    first = upload([("a.md", b"# A\n\nbody", "text/markdown")], made["id"]).json()
    assert first["dataset"] == "Engineering docs"
    upload([("b.md", b"# B\n\nbody", "text/markdown"),
            ("c.md", b"# C\n\nbody", "text/markdown")], made["id"])
    assert collection("Engineering docs")["docs"] == 3
    stored = client.get(f"/api/documents/{first['ids'][0]}").json()
    assert stored["metadata"]["corpus"] == "Engineering docs"


def test_a_pasted_document_uses_the_same_destination(own_corpus, own_datasets):
    made = make("Field notes")
    r = client.post("/api/documents", json={"title": "Jotting", "text": "body", "dataset": made["id"]}).json()
    assert r["dataset"] == "Field notes"
    assert client.get(f"/api/documents/{r['id']}").json()["metadata"]["corpus"] == "Field notes"


def test_the_chosen_dataset_beats_a_corpus_declared_in_front_matter(own_corpus, own_datasets):
    """Otherwise a document could quietly file itself somewhere other than where it was sent —
    including into a sample corpus it has nothing to do with."""
    made = make("Engineering docs")
    body = b"---\ncorpus: kestrel\ndoc_type: runbook\nteam: platform\n---\n# Restocking\n\nbody"
    did = upload([("r.md", body, "text/markdown")], made["id"]).json()["ids"][0]
    meta = client.get(f"/api/documents/{did}").json()["metadata"]
    assert meta["corpus"] == "Engineering docs"
    assert (meta["doc_type"], meta["team"]) == ("runbook", "platform"), "other metadata must survive"


def test_an_unknown_destination_is_refused_before_any_file_is_imported(own_corpus, own_datasets):
    before = len(client.get("/api/documents").json())
    r = upload([("a.md", b"# A\n\nbody", "text/markdown")], "nosuchid")
    assert r.status_code == 404 and "No dataset" in r.json()["detail"]
    assert len(client.get("/api/documents").json()) == before, "nothing may be stored on a bad id"
    r = client.post("/api/documents", json={"text": "body", "dataset": "nosuchid"})
    assert r.status_code == 404


def test_a_partly_failed_upload_files_only_what_worked(own_corpus, own_datasets):
    made = make("Mixed")
    r = upload([("scan.pdf", blank_pdf(), "application/pdf"),
                ("good.md", b"# Good\n\nbody", "text/markdown"),
                ("empty.txt", b"  ", "text/plain")], made["id"]).json()
    assert [x["ok"] for x in r["results"]] == [False, True, False]
    assert r["dataset"] == "Mixed" and len(r["ids"]) == 1
    assert collection("Mixed")["docs"] == 1


def test_an_import_with_no_destination_still_works(own_corpus):
    """Every caller written before datasets existed sends no destination, and must be unaffected."""
    r = upload([("a.md", b"# A\n\nbody", "text/markdown")]).json()
    assert r["dataset"] is None
    assert client.get(f"/api/documents/{r['ids'][0]}").json()["metadata"].get("corpus") is None
    assert collection("uploaded")["kind"] == "legacy"


# ------------------------------------------------------------ what the pages read
def test_collections_cover_datasets_samples_and_what_came_before(own_corpus, own_datasets):
    made = make("Engineering docs")
    upload([("a.md", b"# A\n\nbody", "text/markdown")], made["id"])
    upload([("b.md", b"# B\n\nbody", "text/markdown")])
    client.post("/api/documents", json={"text": "legacy", "metadata": {"corpus": "old pile"}})
    client.post("/api/documents/sample?name=kestrel")
    by_key = {c["key"]: c for c in client.get("/api/collections").json()}
    assert by_key["Engineering docs"]["kind"] == "dataset" and by_key["Engineering docs"]["docs"] == 1
    assert by_key["kestrel"]["kind"] == "sample" and by_key["kestrel"]["docs"] == 8
    assert by_key["old pile"]["kind"] == "legacy" and by_key["old pile"]["docs"] == 1
    assert by_key["uploaded"]["label"] == "Uploaded / unassigned" and by_key["uploaded"]["docs"] == 1


def test_a_dataset_adopts_documents_already_carrying_its_name(own_corpus, own_datasets):
    """One membership model: the collection is whatever `metadata.corpus` says. A dataset created
    with the same name takes those documents over rather than shadowing them with an empty group."""
    client.post("/api/documents", json={"text": "body", "metadata": {"corpus": "Engineering docs"}})
    made = make("Engineering docs")
    rows = [c for c in client.get("/api/collections").json() if c["key"] == "Engineering docs"]
    assert len(rows) == 1
    assert rows[0] == {"key": "Engineering docs", "label": "Engineering docs", "kind": "dataset",
                       "dataset_id": made["id"], "docs": 1}


# ------------------------------------------------------------ building from one
def test_an_index_built_from_a_dataset_holds_only_its_documents(own_corpus, own_datasets):
    made = make("Engineering docs")
    mine = upload([("a.md", b"# Widget torque\n\nTighten to 9 Nm.", "text/markdown")], made["id"]).json()["ids"]
    client.post("/api/documents/sample?name=kestrel")
    docs = client.get("/api/documents").json()
    ids = [d["id"] for d in docs if d["metadata"].get("corpus") == "Engineering docs"]
    assert ids == mine

    iid = store.create_index("dataset-only", indexer.merged_config({"doc_ids": ids}))
    asyncio.run(indexer.build(store, iid, MODELS))
    assert store.index(iid)["status"] == "ready", store.index(iid)["stats"]
    indexed = {c["doc_id"] for c in store.q("SELECT DISTINCT doc_id FROM chunks WHERE index_id=?", (iid,))}
    assert indexed == set(mine), "documents outside the dataset must not be in the index"
    store.delete_index(iid)


def test_an_empty_selection_is_refused_rather_than_indexing_everything(own_corpus, own_datasets):
    """`doc_ids: []` used to pass a truthiness check and build over the whole corpus, so choosing
    an empty dataset produced an index of every document in the lab."""
    client.post("/api/documents/sample?name=kestrel")
    r = client.post("/api/indexes", json={"name": "from-nothing", "config": {"doc_ids": []}})
    assert r.status_code == 422
    assert any(e["field"] == "doc_ids" and "selects no documents" in e["message"]
               for e in r.json()["detail"]["errors"]), r.text
    assert not [i for i in store.indexes() if i["name"] == "from-nothing"]


def test_omitting_doc_ids_still_means_the_whole_corpus(own_corpus):
    client.post("/api/documents/sample?name=kestrel")
    assert indexer.validate_index(indexer.merged_config({})) == []
    assert indexer.validate_index(indexer.merged_config({"doc_ids": None})) == []


@pytest.mark.parametrize("bad", [["a", 2], "abc", 7])
def test_doc_ids_must_be_a_list_of_ids(bad):
    errors = indexer.validate_index(indexer.merged_config({"doc_ids": bad}))
    assert any(e["field"] == "doc_ids" for e in errors), errors


def test_a_dataset_adopts_its_documents_whatever_case_they_were_written_in(own_corpus, own_datasets):
    """Dataset names are unique without regard to case, so `Engineering docs` and
    `engineering docs` cannot then appear as two collections: you would upload into one, filter by
    the other, and get a different set with nothing to explain it."""
    client.post("/api/documents", json={"text": "a", "metadata": {"corpus": "engineering docs"}})
    client.post("/api/documents", json={"text": "b", "metadata": {"corpus": "ENGINEERING DOCS"}})
    made = make("Engineering docs")
    rows = [c for c in client.get("/api/collections").json() if c["key"].lower() == "engineering docs"]
    assert len(rows) == 1, rows
    assert rows[0]["key"] == "Engineering docs", "the dataset's own spelling is the one shown"
    assert rows[0]["docs"] == 2 and rows[0]["dataset_id"] == made["id"]
    assert next(d for d in client.get("/api/datasets").json() if d["id"] == made["id"])["docs"] == 2


def test_samples_keep_their_own_label_when_counted(own_corpus, own_datasets):
    client.post("/api/documents/sample?name=kestrel")
    assert collection("kestrel")["label"] == "kestrel"
