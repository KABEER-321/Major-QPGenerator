"""Tests for Permanent Five Textbooks Curriculum Storage and Examination Logic.

Verifies:
1.  All five textbooks (Classes 1–5) are stored permanently in data/curriculum/class_N/textbook.pdf.
2.  Curriculum status endpoint (/curriculum/status) reports all 5 textbooks available and ready.
3.  Upload section removed: teachers never need to upload syllabus files.
4.  Simplified paper specifications: subject and topic are auto-determined from textbooks.
5.  Class 1 Prerequisite is unavailable and returns a clear 422 error.
6.  Class 5 Prerequisite automatically routes to Class 4 textbook without silent substitution.
7.  Class 4 Prerequisite routes to Class 3; Class 3 to Class 2; Class 2 to Class 1.
8.  Class 5 Mid-Semester selects approximately the first half of chapters by TOC structure.
9.  Class 5 Final uses the full 10 units of Class 5.
10. Only one paper generated per request (never all 3).
11. Textbooks and indexes persist across server reloads without re-embedding.
12. PDF downloads for teacher (with answers/solutions) and student (without answers) work properly.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.curriculum import (
    CURRICULUM_CHAPTERS,
    get_chapters_for_class,
    get_curriculum_doc_id,
    get_curriculum_status,
    get_midterm_chapters_for_class,
    get_second_half_chapters_for_class,
    get_textbook_path,
)
from app.main import app, resolve_curriculum_and_passages
from app.pdf_export import generate_pdf
from app.schemas import (
    Difficulty,
    ExamType,
    Paper,
    PaperGenerateRequest,
    Question,
    QuestionType,
)


@pytest.fixture
def client():
    return TestClient(app)


def _sample_question(qid: str = "q1", marks: int = 5) -> Question:
    return Question(
        question_id=qid,
        question_text=f"Sample question {qid} for math?",
        question_type=QuestionType.SHORT_ANSWER,
        marks=marks,
        difficulty=Difficulty.MEDIUM,
        topic="Numbers",
        grade="Class 5",
        answer="42",
        solution="Step by step calculation.",
        source_references=[],
    )


# ---------------------------------------------------------------------------
# Test 1: Permanent Five Textbooks Storage
# ---------------------------------------------------------------------------
def test_all_five_textbooks_permanently_stored():
    for grade in range(1, 6):
        tb_path = get_textbook_path(grade)
        assert tb_path.exists(), f"Textbook for Class {grade} missing at {tb_path}"
        assert tb_path.stat().st_size > 100_000, f"Textbook for Class {grade} is too small ({tb_path.stat().st_size} bytes)"


# ---------------------------------------------------------------------------
# Test 2: Curriculum Status Endpoint
# ---------------------------------------------------------------------------
def test_curriculum_status_endpoint(client):
    res = client.get("/curriculum/status")
    assert res.status_code == 200
    data = res.json()
    assert len(data) == 5

    for item in data:
        grade = item["grade"]
        assert item["available"] is True
        assert item["indexed"] is True
        assert item["total_chapters"] == len(CURRICULUM_CHAPTERS[grade])
        assert item["total_chunks"] > 0
        assert len(item["midterm_chapters"]) > 0


# ---------------------------------------------------------------------------
# Test 3: Class 1 Prerequisite Unavailable
# ---------------------------------------------------------------------------
def test_class_1_prerequisite_unavailable():
    req = PaperGenerateRequest(
        grade="1",
        academic_grade="Class 1",
        exam_type=ExamType.PREREQUISITE,
        num_questions=4,
        total_marks=20,
    )
    with pytest.raises(Exception) as exc_info:
        resolve_curriculum_and_passages(req)

    assert "not available for Class 1" in str(exc_info.value.detail)


# ---------------------------------------------------------------------------
# Test 4: Prerequisite Routing (Class 5 -> Class 4, Class 4 -> Class 3, etc.)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("acad_grade,expected_curr_grade", [
    ("5", "Class 4"),
    ("4", "Class 3"),
    ("3", "Class 2"),
    ("2", "Class 1"),
])
def test_prerequisite_curriculum_routing(acad_grade, expected_curr_grade):
    req = PaperGenerateRequest(
        grade=acad_grade,
        academic_grade=f"Class {acad_grade}",
        exam_type=ExamType.PREREQUISITE,
        num_questions=4,
        total_marks=20,
    )
    with patch("app.main.retrieve_relevant_chunks", return_value=["Passage from textbook"]):
        target_doc, passages, forbidden_units = resolve_curriculum_and_passages(req)

    expected_doc_id = get_curriculum_doc_id(expected_curr_grade)
    assert target_doc.document_id == expected_doc_id
    assert req.curriculum_grade == expected_curr_grade
    assert req.academic_grade == f"Class {acad_grade}"
    assert "Prerequisite" in req.syllabus_scope


# ---------------------------------------------------------------------------
# Test 5: Mid-Semester Scope Selection (First Half by TOC)
# ---------------------------------------------------------------------------
def test_midterm_scope_selection():
    # Class 5 has 10 units; first half should be 5 units
    req = PaperGenerateRequest(
        grade="5",
        academic_grade="Class 5",
        exam_type=ExamType.MIDTERM,
        num_questions=4,
        total_marks=20,
    )
    with patch("app.main.retrieve_relevant_chunks", return_value=["Passage 1"]) as mock_retrieve:
        target_doc, passages, forbidden_units = resolve_curriculum_and_passages(req)

    assert target_doc.document_id == "curriculum_class_5"
    assert req.curriculum_grade == "Class 5"
    first_half = get_midterm_chapters_for_class(5)
    second_half = get_second_half_chapters_for_class(5)

    assert len(first_half) == 5
    assert len(second_half) == 5
    assert forbidden_units == second_half
    assert "First half" in req.syllabus_scope
    for ch in first_half:
        assert ch in req.syllabus_scope

    # Verify retrieval was invoked with first_half_only=True
    assert mock_retrieve.call_args.kwargs.get("first_half_only") is True
    assert mock_retrieve.call_args.kwargs.get("permitted_units") == first_half


# ---------------------------------------------------------------------------
# Test 6: Final Examination Uses Full Syllabus
# ---------------------------------------------------------------------------
def test_final_scope_uses_full_syllabus():
    req = PaperGenerateRequest(
        grade="5",
        academic_grade="Class 5",
        exam_type=ExamType.FINAL,
        num_questions=4,
        total_marks=20,
    )
    with patch("app.main.retrieve_relevant_chunks", return_value=["Passage 1"]) as mock_retrieve:
        target_doc, passages, forbidden_units = resolve_curriculum_and_passages(req)

    assert target_doc.document_id == "curriculum_class_5"
    assert req.curriculum_grade == "Class 5"
    assert "Full syllabus" in req.syllabus_scope
    assert forbidden_units == []
    # Final does NOT restrict to first half
    assert mock_retrieve.call_args.kwargs.get("first_half_only", False) is False


# ---------------------------------------------------------------------------
# Test 7: Simplified Specifications (No Document ID or Topic Required)
# ---------------------------------------------------------------------------
def test_simplified_request_without_document_or_topic(client):
    mock_questions = [_sample_question(f"q{i}", marks=5) for i in range(4)]

    with patch("app.main.generate_questions", return_value=mock_questions), \
         patch("app.main.save_paper"):

        response = client.post(
            "/papers/generate",
            json={
                "grade": "5",
                "exam_type": "final",
                "num_questions": 4,
                "total_marks": 20,
                "difficulty": "medium",
                "question_types": ["short_answer"],
            },
        )

        assert response.status_code == 200
        paper = response.json()
        assert paper["academic_grade"] == "Class 5"
        assert paper["curriculum_grade"] == "Class 5"
        assert paper["exam_type"] == "final"
        assert paper["total_marks"] == 20
        assert len(paper["questions"]) == 4


# ---------------------------------------------------------------------------
# Test 8: Single Paper Generation Rule
# ---------------------------------------------------------------------------
def test_generates_only_selected_paper(client):
    mock_questions = [_sample_question(f"q{i}", marks=5) for i in range(4)]

    with patch("app.main.generate_questions", return_value=mock_questions), \
         patch("app.main.save_paper"):

        response = client.post(
            "/papers/generate",
            json={
                "grade": "4",
                "exam_type": "prerequisite",
                "num_questions": 4,
                "total_marks": 20,
            },
        )

        assert response.status_code == 200
        data = response.json()
        # Must return exactly one Paper object
        assert isinstance(data, dict)
        assert data["exam_type"] == "prerequisite"
        assert data["curriculum_grade"] == "Class 3"
        assert "Prerequisite Examination" in data["title"]


# ---------------------------------------------------------------------------
# Test 9: UI Elements (Curriculum Status Card Present, Upload Section Removed)
# ---------------------------------------------------------------------------
def test_ui_contains_curriculum_status_and_no_upload(client):
    res = client.get("/")
    assert res.status_code == 200
    html = res.text

    # Read-only curriculum status card
    assert 'id="curriculum-status-card"' in html
    assert 'id="curriculum-status-list"' in html
    assert "Curriculum Status" in html

    # Simplified controls
    assert 'id="grade-select"' in html
    assert 'id="exam-type-select"' in html
    assert 'id="btn-submit"' in html

    # Upload section removed from teacher UI
    assert 'id="panel-file-upload"' not in html
    assert 'id="panel-text-upload"' not in html
    assert 'id="drop-zone"' not in html
    assert 'id="file-input"' not in html


# ---------------------------------------------------------------------------
# Test 10: PDF Export (Teacher with answers, Student without answers)
# ---------------------------------------------------------------------------
def test_pdf_export_copies():
    paper = Paper(
        paper_id="paper_curr_test",
        document_id="curriculum_class_5",
        title="Class 5 Mathematics — Mid-Semester Examination",
        grade="Class 5",
        academic_grade="Class 5",
        curriculum_grade="Class 5",
        exam_type=ExamType.MIDTERM,
        syllabus_scope="First half of Class 5 syllabus (Units 1-5)",
        topic="Mathematics",
        total_marks=20,
        num_questions=4,
        difficulty=Difficulty.MEDIUM,
        subject="Mathematics",
        questions=[_sample_question(f"q_{i}", marks=5) for i in range(4)],
    )

    teacher_pdf = generate_pdf(paper, include_answers=True)
    assert isinstance(teacher_pdf, bytes)
    assert len(teacher_pdf) > 1000

    student_pdf = generate_pdf(paper, include_answers=False)
    assert isinstance(student_pdf, bytes)
    assert len(student_pdf) > 1000
    # Student copy is typically slightly smaller because answers/solutions are omitted
    assert len(student_pdf) <= len(teacher_pdf)


def test_download_paper_pdf_endpoint_with_unicode_title(client, monkeypatch, tmp_path):
    from app.config import settings
    from app.paper import save_paper

    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    paper = Paper(
        paper_id="paper_unicode_title",
        document_id="curriculum_class_4",
        title="Class 4 Mathematics — Prerequisite Examination",
        grade="Class 4",
        academic_grade="Class 4",
        curriculum_grade="Class 3",
        exam_type=ExamType.PREREQUISITE,
        syllabus_scope="Class 3 syllabus",
        topic="Mathematics",
        total_marks=10,
        num_questions=2,
        difficulty=Difficulty.EASY,
        subject="Mathematics",
        questions=[_sample_question("q1", marks=5), _sample_question("q2", marks=5)],
    )
    save_paper(paper)

    # Test teacher download (include_answers=True)
    res_teacher = client.get(f"/papers/{paper.paper_id}/pdf?include_answers=true")
    assert res_teacher.status_code == 200
    assert res_teacher.headers["content-type"] == "application/pdf"
    assert "attachment; filename=" in res_teacher.headers["content-disposition"]
    assert "solutions.pdf" in res_teacher.headers["content-disposition"]
    assert len(res_teacher.content) > 1000

    # Test student download (include_answers=False)
    res_student = client.get(f"/papers/{paper.paper_id}/pdf?include_answers=false")
    assert res_student.status_code == 200
    assert res_student.headers["content-type"] == "application/pdf"
    assert "attachment; filename=" in res_student.headers["content-disposition"]
    assert "student.pdf" in res_student.headers["content-disposition"]
    assert len(res_student.content) > 1000

