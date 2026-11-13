"""Lightweight unit tests; do not require downloading embeddings."""
import sys
import types
from pathlib import Path

# Dependency-light unit tests: ingestion functions are tested without a DB.
sys.modules.setdefault("chromadb", types.ModuleType("chromadb"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ingest import chunk_text, build_chunks, load_documents
from evaluate import precision_recall_f1, TEST_QUERIES
from search import semantic_search, relevance_label


def test_chunking_invariants():
    chunks = chunk_text("one two three four five six seven eight", 18, 5)
    assert len(chunks) >= 2
    assert all(0 < len(chunk) <= 18 for chunk in chunks)
    assert chunk_text("  ", 10, 2) == []


def test_invalid_chunk_settings():
    import pytest
    with pytest.raises(ValueError):
        chunk_text("example", 0, 0)
    with pytest.raises(ValueError):
        chunk_text("example", 10, 10)
    with pytest.raises(ValueError):
        chunk_text("example", 10, -1)


def test_chunk_metadata():
    docs = [{"document_id": "demo", "source": "demo.txt", "text": "alpha beta gamma delta epsilon"}]
    chunks = build_chunks(docs, chunk_size=15, overlap=4)
    assert chunks[0]["metadata"]["source"] == "demo.txt"
    assert chunks[0]["metadata"]["chunk_index"] == 0
    assert len({chunk["id"] for chunk in chunks}) == len(chunks)


def test_source_level_metrics():
    p, r, f1 = precision_recall_f1(["good.txt", "good.txt", "extra.txt"], {"good.txt"})
    assert p == 0.5 and r == 1.0
    assert round(f1, 3) == 0.667


def test_query_set_and_corpus():
    assert len(TEST_QUERIES) >= 5
    corpus = load_documents(Path(__file__).resolve().parents[1] / "docs")
    assert len(corpus) >= 8
    assert all(doc["text"] for doc in corpus)


def test_empty_query_short_circuits():
    assert semantic_search(" ") == []
    assert semantic_search("test", n_results=0) == []


def test_relevance_bands():
    assert "High" in relevance_label(0.2)
    assert "Low" in relevance_label(1.2)
