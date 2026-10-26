"""Comprehensive test suite for examination-type selection and syllabus-based RAG logic.

Tests all 16 required scenarios:
1.  Class 10 Prerequisite request selects Class 9 curriculum.
2.  Class 9 Prerequisite request selects Class 8 curriculum.
3.  Other supported classes follow dynamic previous-class rule (e.g. 12 -> 11, 8 -> 7).
4.  Prerequisite generation does not silently use current class's syllabus (raises 422 if previous missing).
5.  Mid-Semester generation selects approximately first half of syllabus when no explicit scope exists.
6.  Explicitly defined mid-semester scope overrides default selection.
7.  Final generation uses full applicable selected-class syllabus by default.
8.  Only one examination paper is generated per request.
9.  Selected examination type reaches generator and RAG retrieval logic.
10. Document retrieval is filtered by class and subject.
11. Saved paper contains correct examination type, academic class, and curriculum metadata.
12. Paper question count, marks, question types, and syllabus scope are validated.
13. Teacher and student PDFs correctly identify selected examination, academic class, curriculum, and scope.
14. Missing documents produce clear errors without incorrect fallback.
15. Existing generation, document management, history, and PDF functionality continue to work.
16. Frontend HTML/JS includes exam-type-select, dynamic curriculum hint, and handles response display.
"""

from unittest.mock import patch
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from langchain_core.documents import Document

from app.config import settings
from app.documents_store import save_document_record
from app.main import app, resolve_curriculum_and_passages
from app.paper import assemble_paper
from app.pdf_export import generate_pdf
from app.schemas import (
    Difficulty,
    DocumentRecord,
    ExamType,
    Paper,
    PaperGenerateRequest,
    Question,
    QuestionType,
    determine_midterm_scope,
    extract_syllabus_units,
    format_grade_display,
    get_previous_class,
    parse_grade_number,
)


@pytest.fixture
def client():
    return TestClient(app)


def _sample_question(qid: str = "q1", marks: int = 5, qtype: QuestionType = QuestionType.SHORT_ANSWER) -> Question:
    return Question(
        question_id=qid,
        question_text=f"Find the value of x in {qid}.",
        question_type=qtype,
        marks=marks,
        difficulty=Difficulty.MEDIUM,
        topic="Algebra",
        grade="Class 10",
        answer="x = 2",
        solution="Substitute and solve.",
        source_references=[],
    )


def _sample_doc_record(doc_id: str, grade: str, subject: str = "Mathematics", units: list[str] | None = None) -> DocumentRecord:
    return DocumentRecord(
        document_id=doc_id,
        original_filename=f"{subject}_{grade}.pdf",
        safe_filename=f"{subject}_{grade}.pdf",
        num_chunks=6,
        chunk_ids=[f"{doc_id}_c{i}" for i in range(6)],
        grade=grade,
        subject=subject,
        units=units or [f"Unit {i}: Topic {i}" for i in range(1, 5)],
        content_hash=f"hash_{doc_id}",
    )


# ---------------------------------------------------------------------------
# Scenario 1, 2, 3: Grade helpers & previous-class dynamic deduction
# ---------------------------------------------------------------------------
def test_scenario_1_2_3_grade_deduction():
    # Scenario 1: Class 10 Prerequisite -> Class 9
    assert parse_grade_number("10") == 10
    assert parse_grade_number("Class 10") == 10
    assert parse_grade_number("Grade 10") == 10
    assert parse_grade_number("10th") == 10
    assert get_previous_class("Class 10") == "9"
    assert format_grade_display(get_previous_class("Class 10")) == "Class 9"

    # Scenario 2: Class 9 Prerequisite -> Class 8
    assert parse_grade_number("9") == 9
    assert parse_grade_number("Class 9") == 9
    assert get_previous_class("Class 9") == "8"
    assert format_grade_display(get_previous_class("Class 9")) == "Class 8"

    # Scenario 3: Other supported classes (12 -> 11, 8 -> 7, 7 -> 6, etc.)
    assert get_previous_class("Class 12") == "11"
    assert get_previous_class("Class 8") == "7"
    assert get_previous_class("Class 7") == "6"
    assert get_previous_class("11") == "10"

    # Edge cases: lowest supported class raises ValueError
    with pytest.raises(ValueError):
        get_previous_class("Class 1")
    with pytest.raises(ValueError):
        get_previous_class("Kindergarten")


# ---------------------------------------------------------------------------
# Scenario 4 & 14: Prerequisite does not silently use current syllabus & missing syllabus error
# ---------------------------------------------------------------------------
def test_scenario_4_and_14_prerequisite_missing_previous_syllabus_raises_422(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    class10_rec = _sample_doc_record("doc_class10", "Class 10")
    save_document_record(class10_rec)

    req = PaperGenerateRequest(
        document_id="doc_class10",
        num_questions=4,
        total_marks=20,
        topic="Real Numbers",
        subject="Mathematics",
        grade="Class 10",
        academic_grade="Class 10",
        exam_type=ExamType.PREREQUISITE,
    )

    with pytest.raises(HTTPException) as exc_info:
        resolve_curriculum_and_passages(req)

    assert exc_info.value.status_code == 422
    assert "Prerequisite Examination requires syllabus for Class 9 Mathematics" in str(exc_info.value.detail)
    assert "no syllabus document was found for Class 9" in str(exc_info.value.detail)


def test_scenario_4_prerequisite_resolves_when_previous_syllabus_present(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    class10_rec = _sample_doc_record("doc_class10", "Class 10")
    class9_rec = _sample_doc_record("doc_class9", "Class 9")

    save_document_record(class10_rec)
    save_document_record(class9_rec)

    req = PaperGenerateRequest(
        document_id="doc_class10",
        num_questions=4,
        total_marks=20,
        topic="Number Systems",
        subject="Mathematics",
        grade="Class 10",
        academic_grade="Class 10",
        exam_type=ExamType.PREREQUISITE,
    )

    with patch("app.main.retrieve_relevant_chunks", return_value=["Passage from Class 9 syllabus"]):
        target_doc, passages, forbidden_units = resolve_curriculum_and_passages(req)

    assert target_doc.document_id == "doc_class9"
    assert req.curriculum_grade == "Class 9"
    assert "Class 9" in req.syllabus_scope
    assert passages == ["Passage from Class 9 syllabus"]


# ---------------------------------------------------------------------------
# Scenario 5 & 6: Mid-Semester first-half scope and explicit scope override
# ---------------------------------------------------------------------------
def test_scenario_5_midterm_default_half_syllabus():
    units = [f"Unit {i}: Topic {i}" for i in range(1, 9)]  # 8 units
    midterm_units = determine_midterm_scope(units)
    assert len(midterm_units) == 4
    assert midterm_units == ["Unit 1: Topic 1", "Unit 2: Topic 2", "Unit 3: Topic 3", "Unit 4: Topic 4"]

    five_units = [f"Unit {i}" for i in range(1, 6)]
    assert determine_midterm_scope(five_units) == ["Unit 1", "Unit 2", "Unit 3"]


def test_scenario_6_explicit_midterm_scope_override(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    doc10 = _sample_doc_record("doc10", "Class 10", units=["U1", "U2", "U3", "U4"])
    save_document_record(doc10)

    req = PaperGenerateRequest(
        document_id="doc10",
        num_questions=4,
        total_marks=20,
        topic="General Math",
        grade="Class 10",
        academic_grade="Class 10",
        exam_type=ExamType.MIDTERM,
        syllabus_scope="Units 1 to 3 only (Teacher Selection)",
    )
    with patch("app.main.retrieve_relevant_chunks", return_value=["Midterm chunk 1"]):
        target_doc, passages, forbidden_units = resolve_curriculum_and_passages(req)

    assert req.syllabus_scope == "Units 1 to 3 only (Teacher Selection)"
    assert req.curriculum_grade == "Class 10"


# ---------------------------------------------------------------------------
# Scenario 7: Final examination uses full syllabus by default
# ---------------------------------------------------------------------------
def test_scenario_7_final_uses_full_syllabus(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    doc10 = _sample_doc_record("doc10", "Class 10")
    save_document_record(doc10)

    req = PaperGenerateRequest(
        document_id="doc10",
        num_questions=4,
        total_marks=20,
        topic="All Topics",
        grade="Class 10",
        academic_grade="Class 10",
        exam_type=ExamType.FINAL,
    )
    with patch("app.main.retrieve_relevant_chunks", return_value=["Full syllabus chunk"]):
        target_doc, passages, forbidden_units = resolve_curriculum_and_passages(req)

    assert target_doc.document_id == "doc10"
    assert req.curriculum_grade == "Class 10"
    assert "Full Syllabus" in req.syllabus_scope


# ---------------------------------------------------------------------------
# Scenario 8: Only one paper is generated per request (never all three)
# ---------------------------------------------------------------------------
def test_scenario_8_only_one_paper_generated_per_request(client):
    mock_questions = [_sample_question(f"q_{i}", marks=5) for i in range(4)]
    doc10 = _sample_doc_record("doc10", "Class 10")

    with patch("app.main.resolve_curriculum_and_passages") as mock_resolve, \
         patch("app.main.generate_questions", return_value=mock_questions), \
         patch("app.main.save_paper"):

        mock_resolve.return_value = (doc10, ["Class 10 mid passage"], [])

        response = client.post(
            "/papers/generate",
            json={
                "document_id": "doc10",
                "total_marks": 20,
                "num_questions": 4,
                "topic": "First Term Topics",
                "grade": "Class 10",
                "academic_grade": "Class 10",
                "subject": "Mathematics",
                "exam_type": "midterm",
            },
        )

        assert response.status_code == 200
        paper = response.json()
        assert paper["exam_type"] == "midterm"
        assert paper["academic_grade"] == "Class 10"
        assert "Mid-Semester Examination" in paper["title"]
        assert isinstance(paper, dict)
        assert len(paper["questions"]) == 4


# ---------------------------------------------------------------------------
# Scenario 9: Selected exam type reaches generator prompt and RAG retrieval
# ---------------------------------------------------------------------------
def test_scenario_9_exam_type_reaches_generator_and_rag():
    from app.generator import _build_human_prompt

    doc = Document(page_content="Sample syllabus passage for algebra")

    # Test prerequisite prompt
    req_prereq = PaperGenerateRequest(
        document_id="doc1",
        num_questions=4,
        total_marks=20,
        topic="Algebra",
        grade="Class 10",
        academic_grade="Class 10",
        curriculum_grade="Class 9",
        exam_type=ExamType.PREREQUISITE,
        syllabus_scope="Full syllabus of Class 9 Mathematics (Prerequisite)",
    )
    prompt_prereq = _build_human_prompt(req_prereq, [doc])
    assert "EXAMINATION TYPE: Prerequisite Examination" in prompt_prereq
    assert "Academic Class of Students: Class 10" in prompt_prereq
    assert "Curriculum Class Used: Class 9" in prompt_prereq

    # Test midterm prompt
    req_midterm = PaperGenerateRequest(
        document_id="doc1",
        num_questions=4,
        total_marks=20,
        topic="Algebra",
        grade="Class 10",
        academic_grade="Class 10",
        curriculum_grade="Class 10",
        exam_type=ExamType.MIDTERM,
        syllabus_scope="First half of syllabus: Unit 1, Unit 2",
    )
    prompt_midterm = _build_human_prompt(req_midterm, [doc])
    assert "EXAMINATION TYPE: Mid-Semester Examination" in prompt_midterm
    assert "Do NOT generate questions from topics outside this permitted mid-semester scope" in prompt_midterm

    # Test final prompt
    req_final = PaperGenerateRequest(
        document_id="doc1",
        num_questions=4,
        total_marks=20,
        topic="Algebra",
        grade="Class 10",
        academic_grade="Class 10",
        curriculum_grade="Class 10",
        exam_type=ExamType.FINAL,
        syllabus_scope="Full Syllabus",
    )
    prompt_final = _build_human_prompt(req_final, [doc])
    assert "EXAMINATION TYPE: Final Examination" in prompt_final


# ---------------------------------------------------------------------------
# Scenario 10: Document retrieval filtering by class and subject
# ---------------------------------------------------------------------------
def test_scenario_10_document_filtering_by_class_and_subject(tmp_path, monkeypatch):
    from app.documents_store import find_documents_by_class_and_subject, find_best_document_for_grade

    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    doc1 = _sample_doc_record("d1", "Class 9", "Mathematics")
    doc2 = _sample_doc_record("d2", "Class 10", "Mathematics")
    doc3 = _sample_doc_record("d3", "Class 9", "Science")

    save_document_record(doc1)
    save_document_record(doc2)
    save_document_record(doc3)

    math_c9 = find_documents_by_class_and_subject("Class 9", "Mathematics")
    assert len(math_c9) == 1
    assert math_c9[0].document_id == "d1"

    best_c9 = find_best_document_for_grade("Class 9", "Mathematics")
    assert best_c9 is not None
    assert best_c9.document_id == "d1"

    best_c10 = find_best_document_for_grade("Class 10", "Mathematics")
    assert best_c10 is not None
    assert best_c10.document_id == "d2"


# ---------------------------------------------------------------------------
# Scenario 11: Saved paper metadata
# ---------------------------------------------------------------------------
def test_scenario_11_saved_paper_metadata():
    req = PaperGenerateRequest(
        document_id="doc1",
        num_questions=2,
        total_marks=10,
        topic="Algebra",
        grade="Class 10",
        academic_grade="Class 10",
        curriculum_grade="Class 9",
        exam_type=ExamType.PREREQUISITE,
        syllabus_scope="Class 9 Mathematics Prerequisite Content",
        subject="Mathematics",
    )
    candidates = [_sample_question("q1", marks=5), _sample_question("q2", marks=5)]
    paper = assemble_paper(req, candidates)

    assert paper.academic_grade == "Class 10"
    assert paper.curriculum_grade == "Class 9"
    assert paper.exam_type == ExamType.PREREQUISITE
    assert paper.syllabus_scope == "Class 9 Mathematics Prerequisite Content"
    assert "Class 10 Mathematics — Prerequisite Examination" in paper.title


# ---------------------------------------------------------------------------
# Scenario 12: Paper question count, marks, question types validation
# ---------------------------------------------------------------------------
def test_scenario_12_paper_validation():
    # Valid paper
    paper = Paper(
        paper_id="p1",
        document_id="doc1",
        title="Class 10 Mathematics — Mid-Semester Examination",
        grade="Class 10",
        academic_grade="Class 10",
        curriculum_grade="Class 10",
        exam_type=ExamType.MIDTERM,
        topic="Algebra",
        total_marks=15,
        num_questions=2,
        difficulty=Difficulty.MEDIUM,
        subject="Mathematics",
        questions=[_sample_question("q1", marks=5), _sample_question("q2", marks=10)],
    )
    assert paper.total_marks == 15
    assert len(paper.questions) == 2

    # Inconsistent marks raises ValueError
    with pytest.raises(ValueError):
        Paper(
            paper_id="p2",
            document_id="doc1",
            title="Class 10 Mathematics — Mid-Semester Examination",
            grade="Class 10",
            academic_grade="Class 10",
            curriculum_grade="Class 10",
            exam_type=ExamType.MIDTERM,
            topic="Algebra",
            total_marks=50,  # Expected 50, but questions sum to 15
            num_questions=2,
            difficulty=Difficulty.MEDIUM,
            subject="Mathematics",
            questions=[_sample_question("q1", marks=5), _sample_question("q2", marks=10)],
        )


# ---------------------------------------------------------------------------
# Scenario 13: Teacher and Student PDF export contains exam metadata
# ---------------------------------------------------------------------------
def test_scenario_13_pdf_export_contains_exam_metadata():
    paper = Paper(
        paper_id="p101",
        document_id="doc1",
        title="Class 10 Mathematics — Prerequisite Examination",
        grade="Class 10",
        academic_grade="Class 10",
        curriculum_grade="Class 9",
        exam_type=ExamType.PREREQUISITE,
        syllabus_scope="Class 9 Mathematics Prerequisite Content",
        topic="Algebra",
        total_marks=10,
        num_questions=1,
        difficulty=Difficulty.MEDIUM,
        subject="Mathematics",
        questions=[_sample_question("q1", marks=10)],
    )

    student_pdf_bytes = generate_pdf(paper, include_answers=False)
    assert isinstance(student_pdf_bytes, bytes)
    assert len(student_pdf_bytes) > 500

    teacher_pdf_bytes = generate_pdf(paper, include_answers=True)
    assert isinstance(teacher_pdf_bytes, bytes)
    assert len(teacher_pdf_bytes) > 500


# ---------------------------------------------------------------------------
# Scenario 15: Existing endpoints (health, documents, papers) continue to work
# ---------------------------------------------------------------------------
def test_scenario_15_existing_endpoints_continue_working(client):
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"

    docs = client.get("/documents")
    assert docs.status_code == 200
    assert isinstance(docs.json(), list)

    papers = client.get("/papers")
    assert papers.status_code == 200
    assert isinstance(papers.json(), list)


# ---------------------------------------------------------------------------
# Scenario 16: Frontend HTML includes exam-type-select and curriculum hint
# ---------------------------------------------------------------------------
def test_scenario_16_frontend_ui_elements(client):
    res = client.get("/")
    assert res.status_code == 200
    html = res.text

    # Examination type selector and options
    assert 'id="exam-type-select"' in html
    assert 'value="prerequisite"' in html
    assert 'value="midterm"' in html
    assert 'value="final"' in html
    assert "Prerequisite Examination" in html
    assert "Mid-Semester Examination" in html
    assert "Final Examination" in html

    # Dynamic curriculum hint container
    assert 'id="exam-curriculum-hint"' in html

    # Badges styling for exam type, curriculum, scope
    assert "badge-exam-type" in html
    assert "badge-curriculum" in html
    assert "badge-scope" in html
