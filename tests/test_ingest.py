"""Getting documents in: front-matter reading, and telling the user what an import actually did.

Both halves here are the same defect seen from two sides. Front matter that parsed wrong produced
metadata nobody could filter on, and a file that produced no text was dropped without a word. In
each case the lab went on looking like it had worked.
"""
import io
import json

import pytest
from fastapi.testclient import TestClient
from pypdf import PdfWriter

from raglab.main import _front_matter, app

client = TestClient(app)


def blank_pdf() -> bytes:
    """A valid PDF holding one page and no text layer — what a scan looks like to pypdf."""
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


# ------------------------------------------------------------ front matter
def test_nested_front_matter_becomes_dotted_keys():
    """A nested map used to produce an empty `author` plus invented top-level `name` and `team`."""
    meta, body = _front_matter("---\nauthor:\n  name: Ana\n  team: platform\n---\nbody\n")
    assert meta == {"author.name": "Ana", "author.team": "platform"}
    assert body == "body\n"


def test_folded_scalar_keeps_its_text_not_its_indicator():
    """`summary: >` used to be stored as the string `>`."""
    meta, _ = _front_matter("---\nsummary: >\n  spans two\n  lines\n---\nbody")
    assert meta == {"summary": "spans two lines"}


def test_null_is_dropped_and_booleans_stay_booleans():
    """`owner: null` is the absence of an owner, so it should not become a filterable value."""
    meta, _ = _front_matter("---\ndraft: true\nowner: null\n---\nbody")
    assert meta == {"draft": True}


def test_a_date_is_stored_as_an_iso_string():
    """YAML reads an unquoted date as a `datetime.date`, which `json.dumps` cannot write, so a
    document with an ordinary `date:` field would fail to store at all."""
    meta, _ = _front_matter("---\ndate: 2025-03-01\n---\nbody")
    assert meta == {"date": "2025-03-01"}
    assert json.loads(json.dumps(meta)) == meta


def test_a_real_sample_document_parses_to_the_expected_fields():
    """Hand-written expectation against a file in the shipped corpus, so a future change to the
    reader that breaks the documents everyone actually loads is caught here."""
    from raglab.main import SAMPLES
    meta, body = _front_matter((SAMPLES / "halcyon" / "ENG-API-01.md").read_text())
    assert meta == {"title": "TrailPulse Sync API — Rate Limits", "doc_key": "ENG-API-01",
                    "doc_type": "Engineering", "date": "2025-05-30", "year": 2025,
                    "tags": ["API", "TrailPulse", "sync", "rate limit"]}
    assert body.startswith("# TrailPulse Sync API")


def test_a_quoted_version_stays_a_string():
    """`version: "2.0"` must not become the float 2.0 — the corpus compares versions as text."""
    meta, _ = _front_matter('---\nversion: "2.0"\n---\nbody')
    assert meta == {"version": "2.0"}


def test_malformed_yaml_falls_back_to_reading_line_by_line():
    """An unquoted title with a colon in it is not valid YAML. Losing every other field over it
    would be worse than the looser reading that used to be the only one."""
    meta, _ = _front_matter("---\ntitle: Returns: v3\ndoc_type: policy\n---\nbody")
    assert meta == {"title": "Returns: v3", "doc_type": "policy"}


def test_a_block_that_is_not_a_mapping_yields_no_metadata():
    meta, body = _front_matter("---\n- just\n- a list\n---\nbody")
    assert meta == {} and body == "body"


def test_no_front_matter_leaves_the_document_untouched():
    assert _front_matter("# Heading\n\nbody") == ({}, "# Heading\n\nbody")


# ------------------------------------------------------------ import feedback
def test_upload_says_why_an_empty_file_was_skipped(own_corpus):
    r = client.post("/api/documents/upload", files=[("files", ("empty.txt", b"   \n", "text/plain"))]).json()
    assert r["ids"] == []
    assert r["results"] == [{"filename": "empty.txt", "ok": False, "reason": "The file holds no text."}]


def test_upload_names_a_pdf_with_no_text_layer(own_corpus):
    """The case that made this worth building: a scanned PDF imports as nothing at all, and the
    only honest explanation is that it needs OCR."""
    r = client.post("/api/documents/upload", files=[("files", ("scan.pdf", blank_pdf(), "application/pdf"))]).json()
    assert r["ids"] == []
    assert "OCR" in r["results"][0]["reason"]


def test_one_unreadable_file_does_not_stop_the_others(own_corpus):
    r = client.post("/api/documents/upload", files=[
        ("files", ("broken.pdf", b"this is not a pdf", "application/pdf")),
        ("files", ("good.md", b"---\ndoc_type: runbook\n---\n# Restocking\n\nbody", "text/markdown")),
    ]).json()
    assert len(r["ids"]) == 1
    broken, good = r["results"]
    assert broken["ok"] is False and broken["reason"].startswith("Could not be read:")
    assert good == {"filename": "good.md", "ok": True, "id": r["ids"][0]}
    stored = client.get(f"/api/documents/{r['ids'][0]}").json()
    assert stored["title"] == "Restocking" and stored["metadata"]["doc_type"] == "runbook"


def test_sample_load_reports_what_was_already_there(own_corpus):
    """Pressing Load a second time is a no-op, and has to say so rather than looking broken."""
    first = client.post("/api/documents/sample?name=kestrel").json()
    assert first["added"] + first["skipped"] == 8
    again = client.post("/api/documents/sample?name=kestrel").json()
    assert (again["added"], again["skipped"], again["ids"]) == (0, 8, [])


@pytest.mark.parametrize("title", ["Returns: v3", 'He said "no"', "plain"])
def test_a_pasted_title_survives_punctuation(own_corpus, title):
    """The title used to be injected into the text as front matter, so a colon in it made a block
    that no parser reads the same way."""
    did = client.post("/api/documents", json={"title": title, "text": "body text"}).json()["id"]
    assert client.get(f"/api/documents/{did}").json()["title"] == title


def test_a_document_with_unparseable_front_matter_does_not_take_the_batch_down(own_corpus):
    """`yaml.safe_load` raises `RecursionError` — not a `YAMLError` — on deeply nested flow
    collections. Letting it escape returned 500 for the whole request and lost the healthy files
    sent alongside it, which is precisely what the per-file reporting is here to prevent."""
    poison = ("---\nk: " + "[" * 5000 + "]" * 5000 + "\n---\nbody").encode()
    r = client.post("/api/documents/upload", files=[
        ("files", ("poison.md", poison, "text/markdown")),
        ("files", ("fine.md", b"# Fine\n\nbody", "text/markdown")),
    ])
    assert r.status_code == 200, r.text
    assert [x["ok"] for x in r.json()["results"]] == [True, True]
    assert client.get(f"/api/documents/{r.json()['ids'][1]}").json()["title"] == "Fine"


def test_samples_report_how_many_of_their_own_documents_are_loaded(own_corpus):
    """Counted against the sample's filenames, not against documents tagged with its name: a
    document of the user's own carrying `corpus: kestrel` must not make the page believe the
    sample is complete and disable its button while documents are still missing."""
    def kestrel():
        return next(s for s in client.get("/api/samples").json() if s["name"] == "kestrel")

    assert kestrel()["loaded"] == 0
    client.post("/api/documents", json={"title": "Mine", "text": "---\ncorpus: kestrel\n---\nbody"})
    assert kestrel()["loaded"] == 0, "a document of the user's own must not count towards a sample"
    client.post("/api/documents/sample?name=kestrel")
    loaded = kestrel()
    assert loaded["loaded"] == loaded["docs"] == 8
