"""Unit tests for the generated index names and the build metadata shown in Inspect.

The convention is mirrored in frontend/src/pages/Indexes.tsx (suggestName); these tests pin
the backend side, which is authoritative.
"""
import os, tempfile

os.environ.setdefault("RAGLAB_MOCK", "1")
os.environ.setdefault("RAGLAB_DATA", tempfile.mkdtemp())

import pytest  # noqa: E402

from raglab import indexer  # noqa: E402


def name_for(cfg=None, model="nomic-embed-text", existing=()):
    """Generate a name the way the API does: merge the config first, then suggest."""
    return indexer.suggest_name(indexer.merged_config(cfg or {}), model, existing)


# ----------------------------------------------------------------- model_tag

@pytest.mark.parametrize("model,expected", [
    ("nomic-embed-text", "nomic"),
    ("nomic-embed-text:latest", "nomic"),          # ollama version tag is dropped
    ("openai/text-embedding-3-large", "te3-large"),  # provider prefix is dropped
    ("text-embedding-3-large", "te3-large"),       # only a version survives, so keep initials
    ("text-embedding-3-small", "te3-small"),
    ("bge-m3", "bge"),
    ("mxbai-embed-large", "mxbai"),
    ("all-minilm", "minilm"),                      # leading generic word is dropped
    ("e5-large-v2", "e5"),
    ("snowflake-arctic-embed", "snowflake"),
    ("mock", "mock"),
])
def test_model_tag_shortens_known_embedding_models(model, expected):
    assert indexer.model_tag(model) == expected


@pytest.mark.parametrize("model", ["", None, "   ", "///", ":::"])
def test_model_tag_returns_unknown_for_missing_or_unusable_model(model):
    assert indexer.model_tag(model) == "unknown"


def test_model_tag_keeps_the_word_when_every_token_is_generic():
    assert indexer.model_tag("embed") == "embed"
    assert indexer.model_tag("text-embedding") == "text"


# ----------------------------------------------------------------- suggest_name

def test_should_name_a_default_index_after_its_model_strategy_and_size():
    assert name_for() == "nomic-recursive-320x40"


def test_should_use_the_breakpoint_percentile_instead_of_size_for_semantic_chunking():
    assert name_for({"chunking": {"strategy": "semantic"}}) == "nomic-semantic-p90"


def test_should_append_a_flag_for_each_enabled_option_in_a_fixed_order():
    cfg = {"chunking": {"strategy": "fixed", "size": 426, "overlap": 40,
                        "parent_child": {"enabled": True, "parent_size": 1200},
                        "contextual": {"enabled": True}},
           "graph": {"enabled": True, "communities": True},
           "doc_metadata": {"enabled": True}}
    assert name_for(cfg) == "nomic-fixed-426x40-pc-ctx-graph-comm-meta"


def test_should_mark_graph_without_communities_with_a_single_flag():
    assert name_for({"graph": {"enabled": True}}) == "nomic-recursive-320x40-graph"


def test_should_flag_dedupe_only_when_it_is_turned_off():
    assert "nodedupe" not in name_for({"dedupe": {"enabled": True}})
    assert name_for({"dedupe": {"enabled": False}}) == "nomic-recursive-320x40-nodedupe"


def test_should_record_a_document_subset_as_a_document_count():
    assert name_for({"doc_ids": ["a", "b", "c"]}) == "nomic-recursive-320x40-3d"


def test_should_not_flag_a_subset_when_doc_ids_is_empty_or_absent():
    # Both names are unchanged, but for different reasons. None means the whole corpus, which is
    # not a subset. An empty list means an empty selection, which `validate_index` refuses
    # outright -- so no index is ever built from one, and there is nothing for a flag to describe.
    # (It used to mean "every document" here and in the build, which is what let an empty
    # collection quietly produce an index over the entire corpus.)
    assert name_for({"doc_ids": []}) == "nomic-recursive-320x40"
    assert name_for({"doc_ids": None}) == "nomic-recursive-320x40"


def test_should_prefix_the_shortened_embedding_model():
    assert name_for(model="text-embedding-3-large") == "te3-large-recursive-320x40"
    assert name_for(model="") == "unknown-recursive-320x40"


def test_should_ignore_hnsw_parameters_because_they_can_be_rebuilt_later():
    # Rebuilding HNSW on an existing index rewrites its config, so a name carrying M or
    # ef_construction would silently stop matching the index it describes.
    assert name_for({"hnsw": {"M": 16, "ef_construction": 200}}) == \
           name_for({"hnsw": {"M": 64, "ef_construction": 400}})


def test_should_be_deterministic_for_the_same_parameters():
    cfg = {"chunking": {"strategy": "fixed", "size": 300, "overlap": 20, "contextual": {"enabled": True}}}
    assert name_for(cfg) == name_for(cfg)


def test_should_give_different_names_to_configs_that_differ():
    names = {name_for({"chunking": {"strategy": s, "size": size, "overlap": 40}})
             for s, size in [("recursive", 320), ("recursive", 426), ("fixed", 320), ("markdown", 320)]}
    assert len(names) == 4, names


# ----------------------------------------------------------------- collisions

def test_should_add_a_numeric_suffix_when_the_same_parameters_are_built_again():
    built = []
    for _ in range(4):
        built.append(name_for(existing=built))
    assert built == ["nomic-recursive-320x40", "nomic-recursive-320x40-2",
                     "nomic-recursive-320x40-3", "nomic-recursive-320x40-4"]


def test_should_skip_suffixes_that_a_user_has_already_taken():
    existing = ["nomic-recursive-320x40", "nomic-recursive-320x40-2", "nomic-recursive-320x40-3"]
    assert name_for(existing=existing) == "nomic-recursive-320x40-4"


def test_should_not_collide_with_an_unrelated_name():
    assert name_for(existing=["baseline", "my index"]) == "nomic-recursive-320x40"


def test_should_always_return_a_non_empty_name():
    assert name_for({"chunking": {"strategy": "", "size": 0, "overlap": 0}}, model="").strip()


# ----------------------------------------------------------------- name_parts

def test_name_parts_exposes_the_pieces_used_to_assemble_the_name():
    parts = indexer.name_parts(indexer.merged_config({"chunking": {"parent_child": {"enabled": True}}}),
                               "nomic-embed-text")
    assert parts == ["nomic", "recursive", "320x40", "pc"]
    assert "-".join(parts) == name_for({"chunking": {"parent_child": {"enabled": True}}})
