# Real FAISS pipeline for custom syllabi using deterministic fake embeddings
# (no API key needed): proves chapter-restricted retrieval works end to end.

import pytest
from langchain_community.embeddings import DeterministicFakeEmbedding
from unittest.mock import patch

from app.config import settings
from app.custom_syllabus import create_custom_syllabus, load_custom_syllabus, delete_custom_syllabus
from app.retrieval import retrieve_relevant_chunks

TEXT = """Unit 1: Fractions
- Halves and quarters
- Comparing fractions
Unit 2: Geometry
- Triangles
- Circles
Unit 3: Data
- Bar graphs
"""


@pytest.fixture
def fake_embeddings(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    emb = DeterministicFakeEmbedding(size=32)
    with patch("app.retrieval.get_embeddings", return_value=emb):
        yield


def test_retrieval_only_returns_selected_chapters(fake_embeddings):
    syl = create_custom_syllabus(name="Real", text=TEXT, owner_id="t1", grade="4")
    assert syl.has_index and syl.num_chunks >= 3
    assert (settings.indexes_dir / syl.syllabus_id / "index.faiss").exists()

    got = retrieve_relevant_chunks(
        document_id=syl.syllabus_id,
        query="Triangles, Circles",
        permitted_units=["Unit 2: Geometry"],
    )
    assert got
    assert {d.metadata["chapter"] for d in got} == {"Unit 2: Geometry"}

    got2 = retrieve_relevant_chunks(
        document_id=syl.syllabus_id,
        query="anything",
        permitted_units=["Unit 1: Fractions", "Unit 3: Data"],
    )
    assert {d.metadata["chapter"] for d in got2} == {"Unit 1: Fractions", "Unit 3: Data"}


def test_saved_syllabus_survives_reload_and_delete_cleans_up(fake_embeddings):
    syl = create_custom_syllabus(name="Persist", text=TEXT, owner_id="t1")
    again = load_custom_syllabus(syl.syllabus_id, "t1")
    assert [c.title for c in again.chapters] == [c.title for c in syl.chapters]
    assert delete_custom_syllabus(syl.syllabus_id, "t1") is True
    assert not (settings.indexes_dir / syl.syllabus_id).exists()
