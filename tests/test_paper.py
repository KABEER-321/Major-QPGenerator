# tests/test_paper.py
# Tests for paper assembly and JSON persistence — no API key needed.

import json
import pytest
from pathlib import Path

from app.schemas import (
    Question,
    QuestionType,
    Difficulty,
    Paper,
    PaperGenerateRequest,
    PaperSection,
)
from app.paper import assemble_paper, save_paper, load_paper, list_papers
from app.pdf_export import generate_pdf
from app.config import settings


def _make_questions(count=3, marks_each=5):
    """Create a list of valid Question objects for testing."""
    questions = []
    for i in range(count):
        questions.append(Question(
            question_text=f"Test question {i + 1}",
            question_type=QuestionType.SHORT_ANSWER,
            marks=marks_each,
            difficulty=Difficulty.MEDIUM,
            topic="Algebra",
            answer=f"Answer {i + 1}",
            solution=f"Solution {i + 1}",
        ))
    return questions


def _make_request(count=3, marks_each=5):
    """Create a matching PaperGenerateRequest."""
    return PaperGenerateRequest(
        document_id="testdoc123",
        grade="10",
        subject="Mathematics",
        topic="Algebra",
        difficulty=Difficulty.MEDIUM,
        num_questions=count,
        total_marks=count * marks_each,
    )


class TestPaperAssembly:
    def test_assemble_paper_creates_valid_paper(self):
        questions = _make_questions(3, 5)
        request = _make_request(3, 5)
        paper = assemble_paper(request, questions)

        assert paper.num_questions == 3
        assert paper.total_marks == 15
        assert paper.grade == "10"
        assert paper.topic == "Algebra"
        assert len(paper.paper_id) > 0

    def test_assemble_paper_mismatched_count_raises(self):
        """If questions don't match the request, Paper validation should fail."""
        questions = _make_questions(2, 5)  # 2 questions
        request = _make_request(3, 5)      # but request says 3
        with pytest.raises(Exception):
            assemble_paper(request, questions)


class TestPaperPersistence:
    def test_save_and_load_paper(self, tmp_path, monkeypatch):
        """Save a paper to a temp dir and load it back."""
        # Point papers_dir to the temp directory
        monkeypatch.setattr(settings, "DATA_DIR", tmp_path)

        questions = _make_questions(2, 4)
        request = _make_request(2, 4)
        paper = assemble_paper(request, questions)

        save_paper(paper)
        loaded = load_paper(paper.paper_id)

        assert loaded.paper_id == paper.paper_id
        assert loaded.total_marks == 8
        assert len(loaded.questions) == 2

    def test_load_missing_paper_raises(self, tmp_path, monkeypatch):
        monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
        with pytest.raises(FileNotFoundError):
            load_paper("nonexistent_id")

    def test_list_papers_returns_summaries(self, tmp_path, monkeypatch):
        monkeypatch.setattr(settings, "DATA_DIR", tmp_path)

        # Save two papers
        for i in range(2):
            questions = _make_questions(2, 3)
            request = _make_request(2, 3)
            paper = assemble_paper(request, questions)
            save_paper(paper)

        summaries = list_papers()
        assert len(summaries) == 2
        for s in summaries:
            assert s.num_questions == 2
            assert s.total_marks == 6

    def test_list_papers_skips_malformed_json(self, tmp_path, monkeypatch):
        monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
        papers_dir = tmp_path / "papers"
        papers_dir.mkdir(parents=True, exist_ok=True)

        # Write a malformed JSON file
        (papers_dir / "bad.json").write_text("not valid json", encoding="utf-8")

        # Should return empty list, not crash
        summaries = list_papers()
        assert len(summaries) == 0

    def test_path_traversal_prevented(self, tmp_path, monkeypatch):
        """Paper IDs like '../../../etc/passwd' should not escape the papers dir."""
        monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
        with pytest.raises(FileNotFoundError):
            load_paper("../../../etc/passwd")

    def test_assemble_versioned_paper_with_sections(self):
        """Test assembly with explicit sections, batch ID, and version number."""
        q1 = Question(
            question_id="q1",
            question_text="Q1 text",
            question_type=QuestionType.SHORT_ANSWER,
            marks=2,
            difficulty=Difficulty.EASY,
            topic="Algebra",
            answer="Ans 1",
            solution="Sol 1",
        )
        sec_a = PaperSection(
            section_id="A",
            title="Section A",
            question_type=QuestionType.SHORT_ANSWER,
            num_questions=1,
            section_marks=2,
            questions=[q1],
        )
        req = PaperGenerateRequest(
            document_id="doc1",
            grade="10",
            topic="Algebra",
            num_questions=1,
            total_marks=2,
        )
        paper = assemble_paper(
            request=req,
            questions=[q1],
            sections=[sec_a],
            batch_id="batch999",
            version_number=2,
            num_versions=4,
            blueprint_id="bp123",
        )
        assert paper.version_number == 2
        assert paper.num_versions == 4
        assert paper.batch_id == "batch999"
        assert paper.blueprint_id == "bp123"
        assert len(paper.sections) == 1
        assert "Version 2" in paper.title

    def test_backward_compatible_loading_legacy_paper(self, tmp_path, monkeypatch):
        """A legacy paper JSON saved without sections or batch_id should load cleanly."""
        monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
        papers_dir = tmp_path / "papers"
        papers_dir.mkdir(parents=True, exist_ok=True)

        legacy_data = {
            "paper_id": "legacy_001",
            "title": "Legacy Math Paper",
            "grade": "9",
            "subject": "Mathematics",
            "topic": "Geometry",
            "difficulty": "medium",
            "created_at": "2025-12-01T10:00:00",
            "num_questions": 1,
            "total_marks": 3,
            "document_id": "doc_legacy",
            "questions": [
                {
                    "question_id": "leg_q1",
                    "question_text": "What is the angle sum of a triangle?",
                    "question_type": "short_answer",
                    "marks": 3,
                    "difficulty": "medium",
                    "topic": "Geometry",
                    "answer": "180 degrees",
                    "solution": "Angle sum is always 180 degrees",
                }
            ],
        }
        (papers_dir / "legacy_001.json").write_text(json.dumps(legacy_data), encoding="utf-8")

        loaded = load_paper("legacy_001")
        assert loaded.paper_id == "legacy_001"
        assert loaded.version_number == 1
        assert loaded.num_versions == 1
        assert loaded.batch_id is None
        assert loaded.sections == []
        assert loaded.blueprint_id is None

        # Verify list_papers also handles it cleanly
        summaries = list_papers()
        assert len(summaries) == 1
        assert summaries[0].paper_id == "legacy_001"
        assert summaries[0].version_number == 1


class TestPdfExport:
    """Tests for PDF export functionality."""

    @pytest.fixture
    def sample_sectioned_paper(self):
        q1 = Question(
            question_id="q1",
            question_text="Which of the following is an irrational number?",
            question_type=QuestionType.MULTIPLE_CHOICE,
            marks=1,
            difficulty=Difficulty.EASY,
            topic="Real Numbers",
            answer="sqrt(2)",
            options=["2", "3/4", "sqrt(2)", "0.5"],
            correct_option="sqrt(2)",
            solution="sqrt(2) cannot be written as a fraction.",
        )
        q2 = Question(
            question_id="q2",
            question_text="Find the HCF of 12 and 18 using Euclid's division lemma.",
            question_type=QuestionType.SHORT_ANSWER,
            marks=2,
            difficulty=Difficulty.MEDIUM,
            topic="Real Numbers",
            answer="6",
            solution="18 = 12 * 1 + 6, 12 = 6 * 2 + 0. Hence HCF is 6.",
        )
        sec_a = PaperSection(
            section_id="A",
            title="Section A: Multiple Choice",
            question_type=QuestionType.MULTIPLE_CHOICE,
            num_questions=1,
            section_marks=1,
            questions=[q1],
        )
        sec_b = PaperSection(
            section_id="B",
            title="Section B: Short Answer",
            question_type=QuestionType.SHORT_ANSWER,
            num_questions=1,
            section_marks=2,
            questions=[q2],
        )
        return Paper(
            paper_id="sec_paper_1",
            title="Mathematics Exam - Real Numbers",
            grade="10",
            subject="Mathematics",
            topic="Real Numbers",
            difficulty=Difficulty.MEDIUM,
            questions=[q1, q2],
            num_questions=2,
            total_marks=3,
            sections=[sec_a, sec_b],
            version_number=1,
            num_versions=3,
            batch_id="batch_pdf_test",
        )

    def test_teacher_pdf_generation(self, sample_sectioned_paper):
        """Teacher PDF should include questions, answers, and solutions."""
        pdf_bytes = generate_pdf(sample_sectioned_paper, include_answers=True)
        assert isinstance(pdf_bytes, bytes)
        assert len(pdf_bytes) > 1000
        # Valid PDF file starts with %PDF-
        assert pdf_bytes.startswith(b"%PDF-")

    def test_student_pdf_generation(self, sample_sectioned_paper):
        """Student PDF should generate successfully without answers."""
        pdf_bytes = generate_pdf(sample_sectioned_paper, include_answers=False)
        assert isinstance(pdf_bytes, bytes)
        assert len(pdf_bytes) > 1000
        assert pdf_bytes.startswith(b"%PDF-")
        # Student PDF is smaller because Answer Key page is omitted
        teacher_bytes = generate_pdf(sample_sectioned_paper, include_answers=True)
        assert len(pdf_bytes) < len(teacher_bytes)


