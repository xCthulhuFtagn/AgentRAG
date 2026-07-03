"""Tests for vectordb helpers: RRF fusion and per-project settings resolution."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.vectordb.client import get_async_db, invalidate_db_cache
from src.vectordb.indexer import resolve_index_settings
from src.vectordb.tools import _rrf_merge


# ── _rrf_merge: Reciprocal Rank Fusion of vector + full-text results ────────

def test_rrf_merge_prefers_items_ranked_high_in_both_lists():
    vec = [{"seq": 1, "text": "a"}, {"seq": 2, "text": "b"}, {"seq": 3, "text": "c"}]
    fts = [{"seq": 3, "text": "c"}, {"seq": 1, "text": "a"}, {"seq": 4, "text": "d"}]

    merged = _rrf_merge(vec, fts, top_k=4)
    seqs = [r["seq"] for r in merged]

    # seq=1 (rank 1 in both) and seq=3 (rank 3 vec / rank 1 fts) score highest;
    # seq=2 appears only in vec, seq=4 only in fts.
    assert seqs[0] in (1, 3)
    assert set(seqs) == {1, 2, 3, 4}


def test_rrf_merge_caps_at_top_k():
    vec = [{"seq": i, "text": str(i)} for i in range(5)]
    fts = []
    merged = _rrf_merge(vec, fts, top_k=2)
    assert len(merged) == 2
    assert [r["seq"] for r in merged] == [0, 1]  # vec-only, rank order preserved


def test_rrf_merge_dedups_by_seq_keeping_vector_row():
    # The same seq appears in both lists — the vector-side row (carries
    # _distance) must be kept, not silently duplicated or replaced.
    vec = [{"seq": 1, "text": "a", "_distance": 0.05}]
    fts = [{"seq": 1, "text": "a", "_score": 9.9}]
    merged = _rrf_merge(vec, fts, top_k=5)
    assert len(merged) == 1
    assert merged[0]["_distance"] == 0.05


def test_rrf_merge_legacy_dedups_by_text_when_no_seq():
    vec = [{"text": "same"}]
    fts = [{"text": "same"}, {"text": "different"}]
    merged = _rrf_merge(vec, fts, top_k=5)
    assert len(merged) == 2
    assert {r["text"] for r in merged} == {"same", "different"}


# ── resolve_index_settings: new per-project override keys ──────────────────

def test_resolve_index_settings_includes_hybrid_search_toggle():
    from src.vectordb.config import vdb_settings

    resolved = resolve_index_settings(None)
    assert resolved["hybrid_search_enabled"] == vdb_settings.hybrid_search_enabled

    overridden = resolve_index_settings({"hybrid_search_enabled": False})
    assert overridden["hybrid_search_enabled"] is False
    # Untouched keys still fall back to the global defaults.
    assert overridden["search_top_k"] == vdb_settings.search_top_k


# ── Async connection cache ───────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_get_async_db_is_cached_and_invalidated(tmp_path):
    path = str(tmp_path / "lancedb_cache_test")

    db1 = await get_async_db(path)
    db2 = await get_async_db(path)
    assert db1 is db2  # same event loop, same path → cached connection reused

    invalidate_db_cache(path)
    db3 = await get_async_db(path)
    assert db3 is not db1  # a fresh connection is opened after invalidation


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
