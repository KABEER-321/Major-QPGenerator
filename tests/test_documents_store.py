# tests/test_documents_store.py
# Tests for document metadata persistence — no API key needed.

import json
import pytest
from pathlib import Path

from app.schemas import DocumentRecord, DocumentSummary
from app.documents_store import (
    save_document_record,
    load_document_record,
    list_document_records,
    delete_document_record,
)
from app.config import settings


def _make_record(**overrides):
    """Create a valid DocumentRecord for testing."""
    base = {
        "document_id": "testdoc123",
        "original_filename": "algebra_chapter.pdf",
        "safe_filename": "testdoc123.pdf",
        "num_chunks": 8,
        "num_pages": 4,
        "file_size_bytes": 25600,
        "has_index": True,
        "text_preview": "Introduction to quadratic equations...",
    }
    base.update(overrides)
    return DocumentRecord(**base)


class TestDocumentStoreSave:
    def test_save_creates_json_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
        rec = _make_record()
        path = save_document_record(rec)
        assert path.exists()
        assert path.suffix == ".json"

    def test_saved_file_is_valid_json(self, tmp_path, monkeypatch):
        monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
        rec = _make_record()
        path = save_document_record(rec)
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["document_id"] == "testdoc123"
        assert data["num_chunks"] == 8


class TestDocumentStoreLoad:
    def test_load_returns_correct_record(self, tmp_path, monkeypatch):
        monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
        rec = _make_record()
        save_document_record(rec)
        loaded = load_document_record("testdoc123")
        assert loaded.document_id == rec.document_id
        assert loaded.original_filename == rec.original_filename
        assert loaded.num_chunks == rec.num_chunks

    def test_load_missing_raises_file_not_found(self, tmp_path, monkeypatch):
        monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
        with pytest.raises(FileNotFoundError):
            load_document_record("nonexistent")

    def test_load_malformed_raises_value_error(self, tmp_path, monkeypatch):
        monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
        uploads_dir = tmp_path / "uploads"
        uploads_dir.mkdir(parents=True, exist_ok=True)
        (uploads_dir / "bad.json").write_text("not valid json", encoding="utf-8")
        with pytest.raises(ValueError):
            load_document_record("bad")

    def test_path_traversal_prevented(self, tmp_path, monkeypatch):
        monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
        with pytest.raises(FileNotFoundError):
            load_document_record("../../../etc/passwd")


class TestDocumentStoreList:
    def test_list_empty(self, tmp_path, monkeypatch):
        monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
        summaries = list_document_records()
        assert summaries == []

    def test_list_returns_summaries(self, tmp_path, monkeypatch):
        monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
        for i in range(3):
            rec = _make_record(
                document_id=f"doc_{i}",
                original_filename=f"file_{i}.pdf",
                safe_filename=f"doc_{i}.pdf",
            )
            save_document_record(rec)
        summaries = list_document_records()
        assert len(summaries) == 3
        assert all(isinstance(s, DocumentSummary) for s in summaries)

    def test_list_skips_malformed_json(self, tmp_path, monkeypatch):
        monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
        uploads_dir = tmp_path / "uploads"
        uploads_dir.mkdir(parents=True, exist_ok=True)

        # One valid, one malformed
        rec = _make_record(document_id="good")
        save_document_record(rec)
        (uploads_dir / "bad.json").write_text("{invalid", encoding="utf-8")

        summaries = list_document_records()
        assert len(summaries) == 1
        assert summaries[0].document_id == "good"


class TestDocumentStoreDelete:
    def test_delete_existing_record(self, tmp_path, monkeypatch):
        monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
        rec = _make_record()
        save_document_record(rec)
        assert delete_document_record("testdoc123") is True
        # Should be gone now
        with pytest.raises(FileNotFoundError):
            load_document_record("testdoc123")

    def test_delete_nonexistent_returns_false(self, tmp_path, monkeypatch):
        monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
        assert delete_document_record("nope") is False
