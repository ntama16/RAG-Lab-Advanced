"""Editing a document's title and metadata after it has been ingested.

An uploaded PDF carries no front matter, so it arrives with nothing to filter on and nothing to
offer self-query. Everything here exists so that is fixable without re-uploading the file.
"""
import asyncio

import pytest
from fastapi.testclient import TestClient

from raglab import indexer
from raglab.main import MODELS, app, metadata_value, store

client = TestClient(app)


def a_document(text: str = "body text", **meta) -> str:
    """A document with no metadata beyond what is asked for, standing in for an uploaded PDF."""
    return client.post("/api/documents", json={"title": "Scanned guide", "text": text, "metadata": meta}).json()["id"]


def edit(did: str, title: str = "Scanned guide", **fields):
    return client.patch(f"/api/documents/{did}", json={"title": title, "metadata": fields})


def meta_of(did: str) -> dict:
    return client.get(f"/api/documents/{did}").json()["metadata"]


# ------------------------------------------------------------ how a typed value is read
@pytest.mark.parametrize("typed,stored", [
    ("2025", 2025),
    ("3.5", 3.5),
    ("true", True),
    ("False", False),
    ("guide", "guide"),
    ("early childhood teachers", "early childhood teachers"),
    ("guide, evidence", ["guide", "evidence"]),
    ("[guide, evidence]", ["guide", "evidence"]),
    ("  spaced  ", "spaced"),
    ("2025-03-01", "2025-03-01"),
    ("v2.0", "v2.0"),
])
def test_a_typed_value_is_read_the_way_the_editor_promises(typed, stored):
    assert metadata_value(typed) == stored


@pytest.mark.parametrize("typed", ["no", "on", "off", "yes", "null", "~"])
def test_words_yaml_would_turn_into_booleans_stay_text(typed):
    """A full YAML read makes `no` False and `on` True. Defensible in front matter, startling in
    a text box — so the editor's rule is narrower than YAML's on purpose."""
    assert metadata_value(typed) == typed


# ------------------------------------------------------------ editing
def test_metadata_can_be_added_to_a_document_that_had_none(own_corpus):
    did = a_document()
    assert meta_of(did) == {"source": "Scanned guide"}
    r = edit(did, doc_type="guide", year="2024", audience="early childhood teachers, principals")
    assert r.status_code == 200, r.text
    assert meta_of(did) == {"source": "Scanned guide", "doc_type": "guide", "year": 2024,
                            "audience": ["early childhood teachers", "principals"]}


def test_an_existing_field_can_be_changed_and_another_renamed(own_corpus):
    did = a_document(doc_type="manual", team="platform")
    edit(did, doc_type="guide", squad="platform")
    stored = meta_of(did)
    assert stored["doc_type"] == "guide"
    assert stored["squad"] == "platform" and "team" not in stored


def test_emptying_a_value_removes_the_field(own_corpus):
    did = a_document(doc_type="manual", year=2024)
    edit(did, doc_type="manual", year="")
    assert meta_of(did) == {"source": "Scanned guide", "doc_type": "manual"}


def test_a_field_left_out_is_deleted(own_corpus):
    """The editor sends every field it has, so absence is a deletion rather than an omission."""
    did = a_document(doc_type="manual", year=2024)
    edit(did, doc_type="manual")
    assert "year" not in meta_of(did)


def test_the_title_is_saved_and_cannot_be_emptied(own_corpus):
    did = a_document()
    assert edit(did, title="  NSW evidence guide  ").status_code == 200
    assert client.get(f"/api/documents/{did}").json()["title"] == "NSW evidence guide"
    r = edit(did, title="   ")
    assert r.status_code == 422
    assert any(e["field"] == "title" for e in r.json()["detail"]["errors"])


# ------------------------------------------------------------ what the editor may not touch
def test_source_and_corpus_are_kept_and_refused_as_edits(own_corpus):
    """`source` is where the document came from and `corpus` is which collection it is in, which
    follows the dataset it was added to. Two ways to move a document would be two ways to
    disagree about where it is."""
    did = client.post("/api/documents", json={"title": "Doc", "text": "body",
                                              "metadata": {"corpus": "old pile"}}).json()["id"]
    assert edit(did, title="Doc", doc_type="guide").status_code == 200
    kept = meta_of(did)
    assert kept["corpus"] == "old pile" and kept["source"] == "Doc"

    for reserved in ("source", "corpus"):
        r = client.patch(f"/api/documents/{did}", json={"title": "Doc", "metadata": {reserved: "sneaky"}})
        assert r.status_code == 422, reserved
        assert any(e["field"] == reserved for e in r.json()["detail"]["errors"])
    assert meta_of(did)["corpus"] == "old pile"


# ------------------------------------------------------------ refusals
def test_two_fields_with_the_same_trimmed_name_are_refused(own_corpus):
    did = a_document()
    r = client.patch(f"/api/documents/{did}", json={"title": "x", "metadata": {"doc_type": "a", " doc_type ": "b"}})
    assert r.status_code == 422
    assert any("twice" in e["message"] for e in r.json()["detail"]["errors"])


def test_an_unnamed_field_is_refused(own_corpus):
    did = a_document()
    r = client.patch(f"/api/documents/{did}", json={"title": "x", "metadata": {"   ": "value"}})
    assert r.status_code == 422
    assert any("needs a name" in e["message"] for e in r.json()["detail"]["errors"])


@pytest.mark.parametrize("payload,expect", [
    ({"x" * 61: "v"}, "under 60 characters"),
    ({"k": "v" * 401}, "under 400 characters"),
    ({"k": 7}, "must be text"),
    ({f"k{i}": "v" for i in range(41)}, "at most 40 fields"),
])
def test_oversized_or_mistyped_input_is_refused(own_corpus, payload, expect):
    did = a_document()
    r = client.patch(f"/api/documents/{did}", json={"title": "x", "metadata": payload})
    assert r.status_code == 422
    assert any(expect in e["message"] for e in r.json()["detail"]["errors"]), r.text


def test_editing_a_document_that_is_gone_says_so(own_corpus):
    r = client.patch("/api/documents/nosuchid", json={"title": "x", "metadata": {}})
    assert r.status_code == 404 and "No such document" in r.json()["detail"]


def test_nothing_is_written_when_any_field_is_refused(own_corpus):
    did = a_document(doc_type="manual")
    client.patch(f"/api/documents/{did}", json={"title": "New", "metadata": {"doc_type": "guide", "": "x"}})
    stored = client.get(f"/api/documents/{did}").json()
    assert (stored["title"], stored["metadata"]["doc_type"]) == ("Scanned guide", "manual")


# ------------------------------------------------------------ what an index already holds
def test_an_index_keeps_the_metadata_it_was_built_with_until_it_is_rebuilt(own_corpus):
    """A build copies each document's metadata onto its chunks, and that copy is what self-query
    filters on. Saying an edit took effect everywhere would be untrue."""
    did = a_document("Tighten the widget bracket to nine newton metres.", doc_type="manual")
    iid = store.create_index("before-edit", indexer.merged_config({"doc_ids": [did]}))
    asyncio.run(indexer.build(store, iid, MODELS))
    assert store.index(iid)["status"] == "ready"

    saved = edit(did, doc_type="guide", year="2024").json()
    assert saved["stale"] == [{"index_id": iid, "name": "before-edit", "chunks": 1}]
    assert indexer.runtime(store, iid).schema()["doc_type"] == ["manual"], "the index is a snapshot"

    asyncio.run(indexer.build(store, iid, MODELS))
    schema = indexer.runtime(store, iid).schema()
    assert schema["doc_type"] == ["guide"] and schema["year"] == [2024]
    store.delete_index(iid)


def test_a_document_in_no_index_reports_nothing_stale(own_corpus):
    did = a_document()
    assert edit(did, doc_type="guide").json()["stale"] == []


def test_edited_metadata_becomes_filterable_in_a_new_index(own_corpus):
    """The whole point: an uploaded file with no front matter can be given fields that self-query
    can then filter on."""
    did = a_document("Evidence of practice for early childhood teachers in New South Wales.")
    edit(did, doc_type="guide", audience="early childhood, principals")
    iid = store.create_index("after-edit", indexer.merged_config({"doc_ids": [did]}))
    asyncio.run(indexer.build(store, iid, MODELS))
    runtime = indexer.runtime(store, iid)
    assert runtime.schema()["doc_type"] == ["guide"]
    assert runtime.allowed([{"field": "audience", "op": "eq", "value": "principals"}]).all()
    assert not runtime.allowed([{"field": "doc_type", "op": "eq", "value": "manual"}]).any()
    store.delete_index(iid)


# ------------------------------------------------------------ carrying values through untouched
ROUND_TRIP = [
    ("a quoted version", "2.0"),
    ("a leading zero", "007"),
    ("a string holding a comma", "teachers, principals"),
    ("a list member holding a comma", ["Smith, J.", "Lee, K."]),
    ("an ordinary list", ["API", "sync"]),
    ("a year", 2025),
    ("an ISO date", "2025-05-30"),
    ("a boolean", True),
]


@pytest.mark.parametrize("label,value", ROUND_TRIP, ids=[label for label, _ in ROUND_TRIP])
def test_a_field_nobody_typed_in_is_stored_exactly_as_it_was(own_corpus, label, value):
    """The editor shows every field as text, so re-reading that text on save used to re-guess the
    type of fields nobody had touched: `version: "2.0"` became the float 2.0, `"007"` became 7,
    and one string holding a comma became two. Fixing one typo must not rewrite the rest."""
    did = a_document(field=value)
    r = client.patch(f"/api/documents/{did}", json={
        "title": "Scanned guide", "metadata": {"field": {"keep": value}, "doc_type": "guide"}})
    assert r.status_code == 200, r.text
    stored = meta_of(did)
    assert stored["field"] == value and type(stored["field"]) is type(value)
    assert stored["doc_type"] == "guide", "the field that was edited still changes"


def test_a_long_existing_value_does_not_block_editing_other_fields(own_corpus):
    """A document ingested with a long front-matter field used to be uneditable, and the refusal
    named a field the user had not opened."""
    summary = "x" * 500
    did = a_document(summary=summary)
    r = client.patch(f"/api/documents/{did}", json={
        "title": "Scanned guide", "metadata": {"summary": {"keep": summary}, "doc_type": "guide"}})
    assert r.status_code == 200, r.text
    assert meta_of(did)["summary"] == summary
    # but newly typed text is still held to the editor's ceiling
    r = client.patch(f"/api/documents/{did}", json={"title": "Scanned guide", "metadata": {"summary": summary}})
    assert r.status_code == 422


def test_a_carried_value_still_has_to_be_something_storable(own_corpus):
    did = a_document()
    for bad in [{"nested": 1}, [{"nested": 1}], "x" * 20_001]:
        r = client.patch(f"/api/documents/{did}", json={"title": "x", "metadata": {"f": {"keep": bad}}})
        assert r.status_code == 422, bad
