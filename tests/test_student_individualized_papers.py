# tests/test_student_individualized_papers.py
# Feature: Individualized Question Papers for Every Student in a Classroom
# Verifies:
# 1. Common blueprint invariance across all student papers (marks, count, sections, difficulty).
# 2. Genuinely different questions with numerical & visual diversity (zero duplicate questions).
# 3. Figure-based questions diversity across student papers.
# 4. Student assignment with names & roll numbers.
# 5. Student PDFs (no answers) vs Teacher PDFs (solutions) & batch ZIP export.

import io
import re
import zipfile
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.duplicates import normalize_question_text, validate_batch_papers_uniqueness
from app.figures import create_figure_question
from app.main import app
from app.paper import assemble_paper, load_paper
from app.pdf_export import generate_pdf
from app.schemas import (
    Difficulty,
    MathCheckStatus,
    MathVerificationResult,
    PaperGenerateRequest,
    Question,
    QuestionType,
    StudentEntry,
)

client = TestClient(app)

TOPICS = ["rivers", "markets", "gardens", "libraries", "buses", "kites", "rooms", "farms", "ponds", "trains"]


def _make_candidate_pool(n_mcq: int, n_sa: int):
    """Generate a mock pool of diverse candidate questions with SymPy verification."""
    pool = []
    for i in range(n_mcq):
        val = 10 + i * 3
        pool.append(Question(
            question_id=f"cand_mcq_{i}",
            question_text=f"Find the value of {val} + 15 in scenario {i} regarding {TOPICS[i % len(TOPICS)]}:",
            question_type=QuestionType.MULTIPLE_CHOICE,
            marks=1,
            difficulty=Difficulty.MEDIUM,
            topic="Addition & Operations",
            chapter="Addition & Operations",
            options=[str(val + 15), str(val + 14), str(val + 16), str(val + 10)],
            correct_option=str(val + 15),
            answer=str(val + 15),
            solution=f"Adding {val} and 15 gives {val + 15}.",
            math_verification=MathVerificationResult(
                status=MathCheckStatus.VERIFIED,
                expression=f"{val} + 15",
                evaluated_value=str(val + 15),
            ),
        ))
    for i in range(n_sa):
        w = 3 + i
        h = 2 + i
        perim = 2 * (w + h)
        pool.append(Question(
            question_id=f"cand_sa_{i}",
            question_text=f"Calculate the perimeter of a rectangle with length {w} cm and breadth {h} cm in {TOPICS[i % len(TOPICS)]}:",
            question_type=QuestionType.SHORT_ANSWER,
            marks=2,
            difficulty=Difficulty.MEDIUM,
            topic="Perimeter and Area",
            chapter="Perimeter and Area",
            answer=f"Perimeter = {perim} cm",
            solution=f"Perimeter = 2 * ({w} + {h}) = {perim} cm.",
            math_verification=MathVerificationResult(
                status=MathCheckStatus.VERIFIED,
                expression=f"2 * ({w} + {h})",
                evaluated_value=str(perim),
            ),
        ))
    return pool


SECTIONS_MOCK = [
    {
        "section_id": "A",
        "title": "Section A: Multiple Choice Questions",
        "question_type": "multiple_choice",
        "num_questions": 2,
        "section_marks": 2,
        "marks_per_question": 1,
    },
    {
        "section_id": "B",
        "title": "Section B: Short Answer Questions",
        "question_type": "short_answer",
        "num_questions": 2,
        "section_marks": 4,
        "marks_per_question": 2,
    },
]


@pytest.fixture
def mock_batch_env(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    with patch("app.main.retrieve_relevant_chunks", return_value=[]), \
         patch("app.main.generate_candidate_pool") as gen:
        yield gen


def test_individual_student_papers_with_roster(mock_batch_env):
    """Test generating individualized papers for specific students in a roster."""
    roster = [
        {"student_name": "Aarav Sharma", "roll_number": "101"},
        {"student_name": "Diya Patel", "roll_number": "102"},
        {"student_name": "Mohammed Zaid", "roll_number": "103"},
    ]
    # Provide candidate pool of 6 MCQs and 6 Short Answers (enough for 3 students * 2 each)
    mock_batch_env.return_value = (_make_candidate_pool(6, 6), 0, 0)

    payload = {
        "document_id": "doc_student_test",
        "grade": "5",
        "subject": "Mathematics",
        "topic": "Mathematics Curriculum",
        "num_versions": 3,
        "sections": SECTIONS_MOCK,
        "generation_mode": "individual",
        "students": roster,
    }

    resp = client.post("/papers/generate-batch", json=payload)
    assert resp.status_code == 200, resp.text
    data = resp.json()

    assert data["num_versions_generated"] == 3
    assert data["generation_mode"] == "individual"
    assert len(data["papers"]) == 3

    # Check student assignment & titles
    papers = data["papers"]
    for idx, (p, exp_st) in enumerate(zip(papers, roster)):
        assert p["student_name"] == exp_st["student_name"]
        assert p["roll_number"] == exp_st["roll_number"]
        assert exp_st["student_name"] in p["title"]
        assert p["version_number"] == idx + 1

    # 1. Verify Common Blueprint Invariance:
    # All student papers must have identical total marks, question count, and section distribution
    for p in papers:
        assert p["total_marks"] == 6
        assert p["num_questions"] == 4
        assert len(p["sections"]) == 2
        assert p["sections"][0]["section_marks"] == 2
        assert p["sections"][1]["section_marks"] == 4

    # 2. Verify Cross-Paper Uniqueness (Different questions for every student)
    all_q_ids = [q["question_id"] for p in papers for q in p["questions"]]
    assert len(all_q_ids) == len(set(all_q_ids)) == 12  # 3 students * 4 questions = 12 unique questions

    all_texts = [normalize_question_text(q["question_text"]) for p in papers for q in p["questions"]]
    assert len(all_texts) == len(set(all_texts)) == 12

    # Validation summary check
    assert data["validation_summary"]["is_unique"] is True
    assert data["validation_summary"]["all_versions_valid"] is True


def test_individual_student_papers_without_roster(mock_batch_env):
    """Test generating individualized papers when teacher specifies student count without names."""
    mock_batch_env.return_value = (_make_candidate_pool(4, 4), 0, 0)

    payload = {
        "document_id": "doc_student_test",
        "grade": "4",
        "subject": "Mathematics",
        "topic": "Mathematics Curriculum",
        "num_versions": 2,
        "sections": SECTIONS_MOCK,
        "generation_mode": "individual",
    }

    resp = client.post("/papers/generate-batch", json=payload)
    assert resp.status_code == 200, resp.text
    data = resp.json()

    assert data["num_versions_generated"] == 2
    papers = data["papers"]
    assert papers[0]["set_label"] == "Student 1"
    assert papers[1]["set_label"] == "Student 2"
    assert "Student 1" in papers[0]["title"]
    assert "Student 2" in papers[1]["title"]
    assert papers[0]["student_name"] is None


def test_figure_questions_generate_diverse_valid_variations():
    """Verify that create_figure_question produces mathematically distinct variations."""
    # Test fractions variations across student indices 0..3
    frac_qs = [
        create_figure_question(grade=5, chapter="fractions", question_type=QuestionType.MULTIPLE_CHOICE, marks=1, variation_index=i)
        for i in range(4)
    ]
    assert all(q is not None and q.figure is not None for q in frac_qs)
    frac_answers = [q.answer for q in frac_qs]
    # Answers must not all be identical
    assert len(set(frac_answers)) >= 3
    # All fractions must be verified
    assert all(q.math_verification.status == MathCheckStatus.VERIFIED for q in frac_qs)

    # Test perimeter grid variations across student indices 0..3
    perim_qs = [
        create_figure_question(grade=4, chapter="perimeter", question_type=QuestionType.SHORT_ANSWER, marks=2, variation_index=i)
        for i in range(4)
    ]
    assert all(q is not None and q.figure is not None for q in perim_qs)
    perim_values = [q.figure.parameters["perimeter"] for q in perim_qs]
    assert len(set(perim_values)) >= 2
    assert all(q.math_verification.status == MathCheckStatus.VERIFIED for q in perim_qs)

    # Test spatial counting variations for Class 1 across student indices 0..3
    count_qs = [
        create_figure_question(grade=1, chapter="digits", question_type=QuestionType.SHORT_ANSWER, marks=1, variation_index=i)
        for i in range(4)
    ]
    assert all(q is not None and q.figure is not None for q in count_qs)
    inside_counts = [q.figure.parameters["inside_count"] for q in count_qs]
    assert len(set(inside_counts)) >= 2


def test_student_and_teacher_pdfs_and_zip_export(mock_batch_env):
    """Test PDF generation for individual students and ZIP bundle downloads."""
    roster = [
        {"student_name": "Aarav Sharma", "roll_number": "101"},
        {"student_name": "Diya Patel", "roll_number": "102"},
    ]
    mock_batch_env.return_value = (_make_candidate_pool(4, 4), 0, 0)

    payload = {
        "document_id": "doc_student_test",
        "grade": "5",
        "subject": "Mathematics",
        "num_versions": 2,
        "sections": SECTIONS_MOCK,
        "generation_mode": "individual",
        "students": roster,
    }

    resp = client.post("/papers/generate-batch", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    batch_id = data["batch_id"]

    # 1. Student PDF (questions only, student info box)
    p1_id = data["paper_ids"][0]
    p1_pdf = client.get(f"/papers/{p1_id}/pdf?include_answers=false")
    assert p1_pdf.status_code == 200
    assert p1_pdf.content.startswith(b"%PDF")

    # 2. Teacher PDF (solutions and answers)
    p1_sol_pdf = client.get(f"/papers/{p1_id}/pdf?include_answers=true")
    assert p1_sol_pdf.status_code == 200
    assert p1_sol_pdf.content.startswith(b"%PDF")
    # Teacher PDF and Student PDF must differ in content
    assert p1_pdf.content != p1_sol_pdf.content

    # 3. Student PDFs ZIP download
    zip_resp = client.get(f"/papers/batch/{batch_id}/pdfs?include_answers=false")
    assert zip_resp.status_code == 200
    assert zip_resp.headers["content-type"] == "application/zip"
    with zipfile.ZipFile(io.BytesIO(zip_resp.content)) as zf:
        names = sorted(zf.namelist())
        assert len(names) == 2
        assert "Student_01_Aarav_Sharma_Roll_101_student.pdf" in names
        assert "Student_02_Diya_Patel_Roll_102_student.pdf" in names
        assert all(zf.read(n).startswith(b"%PDF") for n in names)

    # 4. Teacher Solutions ZIP download
    sol_zip_resp = client.get(f"/papers/batch/{batch_id}/pdfs?include_answers=true")
    assert sol_zip_resp.status_code == 200
    with zipfile.ZipFile(io.BytesIO(sol_zip_resp.content)) as zf:
        names = zf.namelist()
        assert "Student_01_Aarav_Sharma_Roll_101_solutions.pdf" in names
        assert "Student_02_Diya_Patel_Roll_102_solutions.pdf" in names
        assert "Batch_Master_Index.txt" in names
        # Check index content
        index_text = zf.read("Batch_Master_Index.txt").decode("utf-8")
        assert "Aarav Sharma" in index_text
        assert "Diya Patel" in index_text


def test_classroom_scale_uniqueness():
    """Verify that 10 distinct student papers are properly assembled and cross-validated."""
    req = PaperGenerateRequest(
        grade="5",
        subject="Mathematics",
        topic="General Math",
        num_questions=4,
        total_marks=6,
        question_types=["multiple_choice", "short_answer"],
    )

    # Build pool of 40 questions (20 MCQs, 20 Short Answers) for 10 students
    pool = _make_candidate_pool(20, 20)
    papers = []
    for s_idx in range(10):
        # Deal 2 MCQs and 2 SAs to each student
        student_qs = [pool[s_idx * 2], pool[s_idx * 2 + 1], pool[20 + s_idx * 2], pool[20 + s_idx * 2 + 1]]
        paper = assemble_paper(
            request=req,
            questions=student_qs,
            version_number=s_idx + 1,
            num_versions=10,
            student_name=f"Student {s_idx + 1}",
            roll_number=str(101 + s_idx),
            generation_mode="individual",
        )
        papers.append(paper)

    # Validate batch uniqueness
    is_unique, violations = validate_batch_papers_uniqueness(papers)
    assert is_unique is True
    assert len(violations) == 0

    # Ensure every paper has exact same total marks and count
    assert all(p.total_marks == 6 for p in papers)
    assert all(p.num_questions == 4 for p in papers)
    assert len({p.paper_id for p in papers}) == 10
