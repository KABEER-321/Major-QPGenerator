# tests/test_api.py
# Tests for FastAPI endpoints using httpx TestClient.
# All model/embedding calls are mocked — no API key needed.

import io
import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from fastapi.testclient import TestClient

from app.main import app
from app.config import settings
from app.schemas import (
    Question,
    QuestionType,
    Difficulty,
    SectionBlueprint,
    ExamBlueprint,
    MathVerificationResult,
    MathCheckStatus,
)


client = TestClient(app)


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------

class TestHealth:
    def test_health_returns_ok(self):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_root_dashboard_returns_html(self):
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "QPGen" in response.text



# ---------------------------------------------------------------------------
# Document upload endpoint
# ---------------------------------------------------------------------------

class TestDocumentUpload:
    def test_rejects_unsupported_file_type(self):
        fake_file = io.BytesIO(b"hello world")
        response = client.post(
            "/documents/upload",
            files={"file": ("test.txt", fake_file, "text/plain")},
        )
        assert response.status_code == 400
        assert "Unsupported" in response.json()["detail"]

    def test_rejects_oversized_file(self, monkeypatch):
        """Simulate a file larger than the max upload size."""
        monkeypatch.setattr(settings, "MAX_UPLOAD_SIZE_BYTES", 100)  # 100 bytes
        fake_file = io.BytesIO(b"x" * 200)
        response = client.post(
            "/documents/upload",
            files={"file": ("test.pdf", fake_file, "application/pdf")},
        )
        assert response.status_code == 413

    @patch("app.main.save_document_record")
    @patch("app.main.count_pages", return_value=5)
    @patch("app.main.build_faiss_index")
    @patch("app.main.extract_text", return_value="Some math content about algebra.")
    def test_successful_upload(self, mock_extract, mock_index, mock_pages, mock_save_doc, tmp_path, monkeypatch):
        """Test a successful upload with mocked extraction and indexing."""
        monkeypatch.setattr(settings, "DATA_DIR", tmp_path)

        fake_file = io.BytesIO(b"%PDF-1.4 fake pdf content")
        response = client.post(
            "/documents/upload",
            files={"file": ("chapter1.pdf", fake_file, "application/pdf")},
        )
        assert response.status_code == 200
        data = response.json()
        assert "document_id" in data
        assert data["num_chunks"] > 0
        assert "successfully" in data["message"].lower()
        # Verify document record was saved
        mock_save_doc.assert_called_once()


# ---------------------------------------------------------------------------
# Text Document upload endpoint
# ---------------------------------------------------------------------------

class TestTextDocumentUpload:
    @patch("app.main.save_document_record")
    @patch("app.main.build_faiss_index")
    def test_successful_text_upload(self, mock_index, mock_save_doc, tmp_path, monkeypatch):
        monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
        payload = {
            "title": "Grade 10 Algebra Notes",
            "content": "A linear equation in two variables is of the form ax + by + c = 0.",
        }
        response = client.post("/documents/text", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "document_id" in data
        assert data["num_chunks"] > 0
        assert "Grade 10 Algebra Notes.txt" in data["filename"]
        mock_index.assert_called_once()
        mock_save_doc.assert_called_once()

    def test_rejects_empty_content(self):
        payload = {
            "title": "Notes",
            "content": "   ",
        }
        response = client.post("/documents/text", json=payload)
        assert response.status_code in (400, 422)

    def test_rejects_short_content(self):
        payload = {
            "title": "Notes",
            "content": "short",
        }
        response = client.post("/documents/text", json=payload)
        assert response.status_code == 422

    def test_rejects_missing_title(self):
        payload = {
            "content": "Valid length syllabus content here.",
        }
        response = client.post("/documents/text", json=payload)
        assert response.status_code == 422

    @patch("app.main.build_faiss_index", side_effect=Exception("Embedding quota exceeded"))
    def test_handles_embedding_failure(self, mock_index, tmp_path, monkeypatch):
        monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
        payload = {
            "title": "Chapter Notes",
            "content": "Valid mathematical notes content here.",
        }
        response = client.post("/documents/text", json=payload)
        assert response.status_code == 502
        assert "Failed to create embeddings" in response.json()["detail"]


# ---------------------------------------------------------------------------
# Documents endpoints
# ---------------------------------------------------------------------------

class TestDocumentsEndpoints:
    def test_list_documents_empty(self, tmp_path, monkeypatch):
        monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
        response = client.get("/documents")
        assert response.status_code == 200
        assert response.json() == []

    def test_get_nonexistent_document(self, tmp_path, monkeypatch):
        monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
        response = client.get("/documents/nonexistent123")
        assert response.status_code == 404

    def test_save_and_retrieve_document(self, tmp_path, monkeypatch):
        """Save a document record JSON manually and retrieve it via the API."""
        monkeypatch.setattr(settings, "DATA_DIR", tmp_path)

        uploads_dir = tmp_path / "uploads"
        uploads_dir.mkdir(parents=True, exist_ok=True)

        doc_data = {
            "document_id": "doc_abc",
            "original_filename": "textbook.pdf",
            "safe_filename": "doc_abc.pdf",
            "num_chunks": 12,
            "num_pages": 5,
            "file_size_bytes": 102400,
            "uploaded_at": "2026-01-01T00:00:00",
            "has_index": True,
            "text_preview": "Chapter 1: Introduction to Algebra...",
        }
        (uploads_dir / "doc_abc.json").write_text(
            json.dumps(doc_data), encoding="utf-8"
        )

        # GET single document
        response = client.get("/documents/doc_abc")
        assert response.status_code == 200
        data = response.json()
        assert data["document_id"] == "doc_abc"
        assert data["original_filename"] == "textbook.pdf"
        assert data["num_chunks"] == 12
        assert data["has_index"] is True

        # GET document list
        response = client.get("/documents")
        assert response.status_code == 200
        docs = response.json()
        assert len(docs) == 1
        assert docs[0]["document_id"] == "doc_abc"

    def test_list_documents_skips_malformed_json(self, tmp_path, monkeypatch):
        monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
        uploads_dir = tmp_path / "uploads"
        uploads_dir.mkdir(parents=True, exist_ok=True)

        # Write a malformed JSON file
        (uploads_dir / "bad.json").write_text("not valid json", encoding="utf-8")

        # Should return empty list, not crash
        response = client.get("/documents")
        assert response.status_code == 200
        assert response.json() == []

    def test_list_documents_with_multiple(self, tmp_path, monkeypatch):
        """Multiple document records should all appear in the listing."""
        monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
        uploads_dir = tmp_path / "uploads"
        uploads_dir.mkdir(parents=True, exist_ok=True)

        for i in range(3):
            doc_data = {
                "document_id": f"doc_{i}",
                "original_filename": f"file_{i}.pdf",
                "safe_filename": f"doc_{i}.pdf",
                "num_chunks": 5 + i,
                "file_size_bytes": 1000 * (i + 1),
                "uploaded_at": "2026-01-01T00:00:00",
                "has_index": True,
            }
            (uploads_dir / f"doc_{i}.json").write_text(
                json.dumps(doc_data), encoding="utf-8"
            )

        response = client.get("/documents")
        assert response.status_code == 200
        assert len(response.json()) == 3

    def test_delete_document_success(self, tmp_path, monkeypatch):
        monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
        uploads_dir = tmp_path / "uploads"
        uploads_dir.mkdir(parents=True, exist_ok=True)
        (uploads_dir / "doc_to_del.json").write_text(json.dumps({"document_id": "doc_to_del"}), encoding="utf-8")

        response = client.delete("/documents/doc_to_del")
        assert response.status_code == 200
        assert "deleted successfully" in response.json()["message"]
        assert not (uploads_dir / "doc_to_del.json").exists()

    def test_delete_document_not_found(self, tmp_path, monkeypatch):
        monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
        response = client.delete("/documents/nonexistent")
        assert response.status_code == 404



# ---------------------------------------------------------------------------
# Papers endpoints
# ---------------------------------------------------------------------------

class TestPapersEndpoints:
    def test_list_papers_empty(self, tmp_path, monkeypatch):
        monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
        response = client.get("/papers")
        assert response.status_code == 200
        assert response.json() == []

    def test_get_nonexistent_paper(self, tmp_path, monkeypatch):
        monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
        response = client.get("/papers/nonexistent123")
        assert response.status_code == 404

    def test_get_paper_pdf_nonexistent(self, tmp_path, monkeypatch):
        monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
        response = client.get("/papers/nonexistent123/pdf")
        assert response.status_code == 404

    def test_save_and_retrieve_paper(self, tmp_path, monkeypatch):
        """Save a paper JSON manually and retrieve it via the API."""
        monkeypatch.setattr(settings, "DATA_DIR", tmp_path)

        papers_dir = tmp_path / "papers"
        papers_dir.mkdir(parents=True, exist_ok=True)

        paper_data = {
            "paper_id": "abc123",
            "title": "Test Paper",
            "grade": "8",
            "subject": "Mathematics",
            "topic": "Fractions",
            "difficulty": "easy",
            "created_at": "2026-01-01T00:00:00",
            "document_id": "doc1",
            "num_questions": 1,
            "total_marks": 5,
            "questions": [{
                "question_text": "Simplify 2/4",
                "question_type": "short_answer",
                "marks": 5,
                "difficulty": "easy",
                "topic": "Fractions",
                "answer": "1/2",
                "solution": "Divide numerator and denominator by 2",
            }],
        }
        (papers_dir / "abc123.json").write_text(
            json.dumps(paper_data), encoding="utf-8"
        )

        # GET single paper
        response = client.get("/papers/abc123")
        assert response.status_code == 200
        assert response.json()["paper_id"] == "abc123"

        # GET paper list
        response = client.get("/papers")
        assert response.status_code == 200
        assert len(response.json()) == 1

        # GET paper PDF
        response = client.get("/papers/abc123/pdf")
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/pdf"
        assert len(response.content) > 0

    def test_delete_paper_success(self, tmp_path, monkeypatch):
        monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
        papers_dir = tmp_path / "papers"
        papers_dir.mkdir(parents=True, exist_ok=True)
        (papers_dir / "del123.json").write_text(json.dumps({"paper_id": "del123"}), encoding="utf-8")

        response = client.delete("/papers/del123")
        assert response.status_code == 200
        assert "deleted successfully" in response.json()["message"]
        assert not (papers_dir / "del123.json").exists()

    def test_delete_paper_not_found(self, tmp_path, monkeypatch):
        monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
        response = client.delete("/papers/nonexistent123")
        assert response.status_code == 404


# ---------------------------------------------------------------------------
# Generation API Tests: POST /papers/generate & POST /papers/generate-batch
# ---------------------------------------------------------------------------

class TestSinglePaperGeneration:
    """Tests for POST /papers/generate endpoint."""

    @patch("app.main.save_paper")
    @patch("app.main.generate_questions")
    @patch("app.main.retrieve_relevant_chunks")
    def test_generate_single_paper_success(self, mock_retrieve, mock_gen, mock_save, tmp_path, monkeypatch):
        monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
        mock_retrieve.return_value = []
        mock_q = Question(
            question_id="q1",
            question_text="What is 3 + 5?",
            question_type=QuestionType.SHORT_ANSWER,
            marks=2,
            difficulty=Difficulty.EASY,
            topic="Arithmetic",
            answer="8",
            solution="3 + 5 = 8",
        )
        mock_gen.return_value = [mock_q]

        payload = {
            "document_id": "doc123",
            "grade": "10",
            "subject": "Mathematics",
            "topic": "Arithmetic",
            "difficulty": "easy",
            "num_questions": 1,
            "total_marks": 2,
        }
        response = client.post("/papers/generate", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["num_questions"] == 1
        assert data["total_marks"] == 2
        assert len(data["questions"]) == 1
        assert data["questions"][0]["question_text"] == "What is 3 + 5?"
        mock_save.assert_called_once()

    @patch("app.main.retrieve_relevant_chunks", side_effect=FileNotFoundError("Document not found"))
    def test_generate_single_paper_doc_not_found(self, mock_retrieve):
        payload = {
            "document_id": "missing_doc",
            "grade": "10",
            "topic": "Algebra",
            "num_questions": 2,
            "total_marks": 4,
        }
        response = client.post("/papers/generate", json=payload)
        assert response.status_code == 404


class TestBatchPaperGeneration:
    """Tests for POST /papers/generate-batch endpoint."""

    @patch("app.main.save_paper")
    @patch("app.main.generate_candidate_pool")
    @patch("app.main.retrieve_relevant_chunks")
    def test_generate_batch_two_versions_success(
        self, mock_retrieve, mock_gen_pool, mock_save, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
        mock_retrieve.return_value = []

        # Candidate pool with 4 MCQs and 2 SA (exact amount for 2 versions)
        candidates = [
            Question(
                question_id=f"mcq_{i}",
                question_text=f"MCQ candidate {i} text",
                question_type=QuestionType.MULTIPLE_CHOICE,
                marks=1,
                difficulty=Difficulty.EASY,
                topic="Algebra",
                answer="A",
                options=["A", "B", "C", "D"],
                correct_option="A",
                math_verification=MathVerificationResult(status=MathCheckStatus.VERIFIED),
            )
            for i in range(4)
        ] + [
            Question(
                question_id=f"sa_{i}",
                question_text=f"SA candidate {i} text",
                question_type=QuestionType.SHORT_ANSWER,
                marks=2,
                difficulty=Difficulty.MEDIUM,
                topic="Algebra",
                answer="42",
                solution="40 + 2 = 42",
                math_verification=MathVerificationResult(status=MathCheckStatus.VERIFIED),
            )
            for i in range(2)
        ]

        mock_gen_pool.return_value = (candidates, 0, 0)

        # 2 sections: Section A (2 MCQs, 2 marks), Section B (1 SA, 2 marks)
        # Total per paper: 3 questions, 4 marks.
        payload = {
            "document_id": "doc_batch",
            "grade": "10",
            "subject": "Mathematics",
            "topic": "Algebra",
            "difficulty": "medium",
            "num_versions": 2,
            "sections": [
                {
                    "section_id": "A",
                    "title": "Section A: MCQs",
                    "question_type": "multiple_choice",
                    "num_questions": 2,
                    "section_marks": 2,
                    "marks_per_question": 1,
                },
                {
                    "section_id": "B",
                    "title": "Section B: Short Answer",
                    "question_type": "short_answer",
                    "num_questions": 1,
                    "section_marks": 2,
                    "marks_per_question": 2,
                },
            ],
        }

        response = client.post("/papers/generate-batch", json=payload)
        assert response.status_code == 200
        data = response.json()

        assert data["num_versions_requested"] == 2
        assert data["num_versions_generated"] == 2
        assert len(data["papers"]) == 2
        assert len(data["paper_ids"]) == 2
        assert data["validation_summary"]["is_unique"] is True
        assert data["validation_summary"]["all_versions_valid"] is True
        assert data["validation_summary"]["math_verified_count"] == 6

        # Check versions share blueprint and have distinct questions
        p1 = data["papers"][0]
        p2 = data["papers"][1]
        assert p1["version_number"] == 1
        assert p2["version_number"] == 2
        assert p1["num_questions"] == 3
        assert p2["num_questions"] == 3
        assert p1["total_marks"] == 4
        assert p2["total_marks"] == 4

        p1_qids = {q["question_id"] for q in p1["questions"]}
        p2_qids = {q["question_id"] for q in p2["questions"]}
        assert p1_qids.isdisjoint(p2_qids)

        # Verify save_paper called twice (once per paper)
        assert mock_save.call_count == 2

    @patch("app.main.generate_candidate_pool")
    @patch("app.main.retrieve_relevant_chunks")
    def test_generate_batch_candidate_pool_shortage(self, mock_retrieve, mock_gen_pool):
        mock_retrieve.return_value = []
        # Return only 1 question when 2 versions require at least 2
        candidates = [
            Question(
                question_id="sa_single",
                question_text="Only one question available",
                question_type=QuestionType.SHORT_ANSWER,
                marks=2,
                difficulty=Difficulty.MEDIUM,
                topic="Algebra",
                answer="1",
                solution="1",
            )
        ]
        mock_gen_pool.return_value = (candidates, 0, 0)

        payload = {
            "document_id": "doc123",
            "grade": "10",
            "topic": "Algebra",
            "num_questions": 1,
            "total_marks": 2,
            "num_versions": 2,
            "question_types": ["short_answer"],
        }
        response = client.post("/papers/generate-batch", json=payload)
        assert response.status_code == 422
        detail = response.json()["detail"]
        assert detail["error"] == "CandidatePoolShortageError"
        assert detail["needed"] == 2
        assert detail["available"] == 1

    def test_generate_batch_invalid_blueprint_rejected(self):
        # sum of section marks != total marks
        payload = {
            "document_id": "doc123",
            "grade": "10",
            "topic": "Algebra",
            "num_versions": 2,
            "num_questions": 2,
            "total_marks": 10,
            "sections": [
                {
                    "section_id": "A",
                    "title": "Section A",
                    "question_type": "short_answer",
                    "num_questions": 2,
                    "section_marks": 8,  # 8 != 10
                    "marks_per_question": 4,
                }
            ],
        }
        response = client.post("/papers/generate-batch", json=payload)
        assert response.status_code == 422


