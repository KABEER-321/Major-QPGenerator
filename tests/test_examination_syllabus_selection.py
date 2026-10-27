"""Comprehensive automated test suite for examination-wise syllabus selection in QPGen.

Validates all requirements:
1. Prerequisite Examination:
   - Uses 100% of the previous class's textbook (Class 2 -> Class 1, Class 3 -> Class 2, Class 4 -> Class 3, Class 5 -> Class 4).
   - Class 1 Prerequisite is unavailable unless a previous-level textbook is provided.
2. Mid-Semester Examination:
   - Uses the first half of the selected class's textbook starting from chapter 1.
   - Uses consistent rule: number_of_chapters = ceiling(total_chapters / 2).
   - Odd and even chapter count splitting (9 -> 5, 6 -> 3, 5 -> 3, 10 -> 5).
3. Final Examination:
   - Uses 100% of the selected class's textbook syllabus, including all identified chapters.
   - Does not restrict retrieval to the first few chapters.
4. Authoritative Syllabus Scope:
   - Consistent scope description, retrieval, and validation.
   - Retrieval filtering ensures eligible chunks only.
5. Balanced Question Coverage:
   - Chapter-wise blueprint distribution across sections.
   - Transparent coverage explanation when N < M.
6. Validation & Error Handling:
   - Rejects questions touching forbidden chapters or missing chapters.
   - Rejects missing/ambiguous chapter metadata.
"""

import math
from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document

from app.curriculum import (
    CURRICULUM_CHAPTERS,
    assign_blueprint_chapter_coverage,
    get_chapters_for_class,
    get_curriculum_doc_id,
    get_midterm_chapters_for_class,
    get_second_half_chapters_for_class,
    order_chapters_by_textbook,
    resolve_authoritative_syllabus_scope,
)
from app.generator import (
    _enrich_question,
    generate_questions,
    is_question_in_forbidden_units,
)
from app.main import app, resolve_curriculum_and_passages
from app.retrieval import is_chunk_in_permitted_scope, retrieve_relevant_chunks
from app.schemas import (
    Difficulty,
    DocumentRecord,
    ExamBlueprint,
    ExamType,
    Paper,
    PaperGenerateRequest,
    PaperSection,
    Question,
    QuestionType,
    SectionBlueprint,
    SourceReference,
    create_default_blueprint,
    determine_midterm_scope,
    determine_second_half_scope,
)
from app.validator import validate_full_paper


@pytest.fixture
def client():
    return TestClient(app)


# ---------------------------------------------------------------------------
# Test 1: Strict Textbook Chapter Ordering & Odd/Even Splitting
# ---------------------------------------------------------------------------
def test_textbook_chapter_ordering_and_odd_even_splitting():
    """Verify chapter ordering is non-alphabetical and uses ceiling(total / 2)."""
    # Class 1: 9 chapters -> 5 midterm, 4 second-half
    ch1 = get_chapters_for_class(1)
    assert len(ch1) == 9
    assert ch1[0] == "Lesson 1: Spatial Understanding"
    assert ch1[8] == "Lesson 9: Number 11 to 20"
    m1 = get_midterm_chapters_for_class(1)
    s1 = get_second_half_chapters_for_class(1)
    assert len(m1) == math.ceil(9 / 2) == 5
    assert len(s1) == 4
    assert m1 == ch1[:5]
    assert s1 == ch1[5:]

    # Class 2: 6 chapters -> 3 midterm, 3 second-half
    ch2 = get_chapters_for_class(2)
    assert len(ch2) == 6
    m2 = get_midterm_chapters_for_class(2)
    s2 = get_second_half_chapters_for_class(2)
    assert len(m2) == math.ceil(6 / 2) == 3
    assert len(s2) == 3
    assert m2 == ch2[:3]
    assert s2 == ch2[3:]

    # Class 3: 5 chapters -> 3 midterm, 2 second-half
    ch3 = get_chapters_for_class(3)
    assert len(ch3) == 5
    m3 = get_midterm_chapters_for_class(3)
    s3 = get_second_half_chapters_for_class(3)
    assert len(m3) == math.ceil(5 / 2) == 3
    assert len(s3) == 2
    assert m3 == ch3[:3]
    assert s3 == ch3[3:]

    # Class 4: 9 chapters -> 5 midterm, 4 second-half
    ch4 = get_chapters_for_class(4)
    assert len(ch4) == 9
    m4 = get_midterm_chapters_for_class(4)
    s4 = get_second_half_chapters_for_class(4)
    assert len(m4) == math.ceil(9 / 2) == 5
    assert len(s4) == 4
    assert m4 == ch4[:5]
    assert s4 == ch4[5:]

    # Class 5: 10 chapters -> 5 midterm, 5 second-half
    # Verify Unit 10 is at the end, not before Unit 2
    ch5 = get_chapters_for_class(5)
    assert len(ch5) == 10
    assert ch5[0] == "Unit 1: 5-digit numbers"
    assert ch5[1] == "Unit 2: Addition"
    assert ch5[9] == "Unit 10: Data Handling"
    m5 = get_midterm_chapters_for_class(5)
    s5 = get_second_half_chapters_for_class(5)
    assert len(m5) == math.ceil(10 / 2) == 5
    assert len(s5) == 5
    assert m5 == ch5[:5]
    assert s5 == ch5[5:]


# ---------------------------------------------------------------------------
# Test 2: Authoritative Syllabus Scope for All Supported Classes & Exam Types
# ---------------------------------------------------------------------------
def test_authoritative_syllabus_scope_prerequisite_all_classes():
    """Verify Prerequisite rules for Classes 1 to 5."""
    # Class 1 Prerequisite: must fail without a previous-level textbook
    with pytest.raises(ValueError, match="Prerequisite Examination is not available for Class 1"):
        resolve_authoritative_syllabus_scope(ExamType.PREREQUISITE, "Class 1")

    # Class 2 Prerequisite: 100% of Class 1 (all 9 chapters)
    scope2 = resolve_authoritative_syllabus_scope(ExamType.PREREQUISITE, "Class 2")
    assert scope2.curriculum_grade == "Class 1"
    assert scope2.academic_grade == "Class 2"
    assert scope2.total_chapters == 9
    assert len(scope2.permitted_chapters) == 9
    assert len(scope2.forbidden_chapters) == 0
    assert scope2.permitted_chapters == get_chapters_for_class(1)

    # Class 3 Prerequisite: 100% of Class 2 (all 6 chapters)
    scope3 = resolve_authoritative_syllabus_scope(ExamType.PREREQUISITE, "Class 3")
    assert scope3.curriculum_grade == "Class 2"
    assert scope3.total_chapters == 6
    assert len(scope3.permitted_chapters) == 6
    assert scope3.permitted_chapters == get_chapters_for_class(2)

    # Class 4 Prerequisite: 100% of Class 3 (all 5 chapters)
    scope4 = resolve_authoritative_syllabus_scope(ExamType.PREREQUISITE, "Class 4")
    assert scope4.curriculum_grade == "Class 3"
    assert scope4.total_chapters == 5
    assert len(scope4.permitted_chapters) == 5
    assert scope4.permitted_chapters == get_chapters_for_class(3)

    # Class 5 Prerequisite: 100% of Class 4 (all 9 chapters)
    scope5 = resolve_authoritative_syllabus_scope(ExamType.PREREQUISITE, "Class 5")
    assert scope5.curriculum_grade == "Class 4"
    assert scope5.total_chapters == 9
    assert len(scope5.permitted_chapters) == 9
    assert scope5.permitted_chapters == get_chapters_for_class(4)


def test_authoritative_syllabus_scope_midterm_all_classes():
    """Verify Mid-Semester first-half boundaries for Classes 1 to 5."""
    expected_midterm_counts = {1: 5, 2: 3, 3: 3, 4: 5, 5: 5}
    for grade, expected_count in expected_midterm_counts.items():
        scope = resolve_authoritative_syllabus_scope(ExamType.MIDTERM, f"Class {grade}")
        assert scope.academic_grade == f"Class {grade}"
        assert scope.curriculum_grade == f"Class {grade}"
        assert len(scope.permitted_chapters) == expected_count
        assert len(scope.forbidden_chapters) == (scope.total_chapters - expected_count)
        assert scope.permitted_chapters == get_chapters_for_class(grade)[:expected_count]
        assert scope.forbidden_chapters == get_chapters_for_class(grade)[expected_count:]


def test_authoritative_syllabus_scope_final_all_classes():
    """Verify Final Examination uses 100% of the textbook for Classes 1 to 5."""
    for grade in range(1, 6):
        scope = resolve_authoritative_syllabus_scope(ExamType.FINAL, f"Class {grade}")
        assert scope.academic_grade == f"Class {grade}"
        assert scope.curriculum_grade == f"Class {grade}"
        assert len(scope.permitted_chapters) == len(get_chapters_for_class(grade))
        assert len(scope.forbidden_chapters) == 0
        assert scope.permitted_chapters == get_chapters_for_class(grade)


def test_missing_or_ambiguous_chapter_metadata_reports_problem():
    """Verify that if chapter metadata is missing, an error is reported instead of guessing."""
    empty_doc = DocumentRecord(
        document_id="empty_doc",
        original_filename="unknown.pdf",
        safe_filename="unknown.pdf",
        num_chunks=0,
        units=[],
        has_index=True,
    )
    with pytest.raises(ValueError, match="missing or ambiguous chapter metadata"):
        resolve_authoritative_syllabus_scope(ExamType.FINAL, "Class 10", custom_doc=empty_doc)

    single_ch_doc = DocumentRecord(
        document_id="single_ch_doc",
        original_filename="single.pdf",
        safe_filename="single.pdf",
        num_chunks=1,
        units=["Only One Chapter"],
        has_index=True,
    )
    with pytest.raises(ValueError, match="At least 2 chapters are required"):
        resolve_authoritative_syllabus_scope(ExamType.MIDTERM, "Class 10", custom_doc=single_ch_doc)


# ---------------------------------------------------------------------------
# Test 3: Balanced Question Blueprint Allocation
# ---------------------------------------------------------------------------
def test_balanced_question_blueprint_when_n_greater_than_or_equal_m():
    """When N >= M, every single chapter must receive at least one question slot."""
    chapters = get_chapters_for_class(5)  # 10 chapters
    bp = create_default_blueprint(
        num_questions=10,
        total_marks=20,
        question_types=[QuestionType.MULTIPLE_CHOICE, QuestionType.SHORT_ANSWER, QuestionType.LONG_ANSWER],
    )
    bp, explanation = assign_blueprint_chapter_coverage(bp, chapters)

    assert "All 10 eligible curriculum chapters are assessed" in explanation
    assert len(bp.allocated_chapters) == 10

    # Collect all allocated chapters across sections
    all_assigned = []
    for sec in bp.sections:
        assert sec.chapter_allocation is not None
        assert len(sec.chapter_allocation) == sec.num_questions
        all_assigned.extend(sec.chapter_allocation)

    assert len(all_assigned) == 10
    # Every chapter is represented
    assert set(all_assigned) == set(chapters)


def test_balanced_question_blueprint_when_n_less_than_m():
    """When N < M, select N representative chapters and explain actual coverage."""
    chapters = get_chapters_for_class(5)  # 10 chapters
    bp = create_default_blueprint(
        num_questions=5,
        total_marks=10,
        question_types=[QuestionType.MULTIPLE_CHOICE, QuestionType.SHORT_ANSWER],
    )
    bp, explanation = assign_blueprint_chapter_coverage(bp, chapters)

    assert "Balanced representative coverage: 5 chapters selected" in explanation
    assert "because requested question count (5) is smaller than available chapters (10)" in explanation
    assert len(bp.allocated_chapters) == 5

    all_assigned = []
    for sec in bp.sections:
        all_assigned.extend(sec.chapter_allocation)
    assert len(all_assigned) == 5
    # First and last chapters are spanned
    assert all_assigned[0] == chapters[0]
    assert all_assigned[-1] == chapters[-1]


# ---------------------------------------------------------------------------
# Test 4: Retrieval Filtering & Full Scope Eligibility
# ---------------------------------------------------------------------------
def test_retrieval_filtering_midterm_vs_final():
    """Verify FAISS retrieval filtering for Midterm excludes second-half chunks."""
    doc_id = "curriculum_class_1"
    all_chapters = get_chapters_for_class(1)
    midterm_chapters = get_midterm_chapters_for_class(1)
    second_half_chapters = get_second_half_chapters_for_class(1)

    # Simulated chunks
    test_chunks = [
        Document(
            page_content=f"Content for {ch}",
            metadata={"chunk_id": f"c_{i}", "chapter": ch, "unit": ch, "is_first_half": i < 5},
        )
        for i, ch in enumerate(all_chapters)
    ]

    # Check is_chunk_in_permitted_scope
    for chk in test_chunks[:5]:
        assert is_chunk_in_permitted_scope(chk, permitted_units=midterm_chapters, first_half_only=True) is True
    for chk in test_chunks[5:]:
        assert is_chunk_in_permitted_scope(chk, permitted_units=midterm_chapters, first_half_only=True) is False

    # For Final Exam, all chunks are eligible
    for chk in test_chunks:
        assert is_chunk_in_permitted_scope(chk, permitted_units=all_chapters, first_half_only=False) is True


# ---------------------------------------------------------------------------
# Test 5: Forbidden Units Question Detection & Validation
# ---------------------------------------------------------------------------
def test_forbidden_units_detection():
    """Verify that candidate questions touching second-half topics are caught."""
    ch1_second_half = get_second_half_chapters_for_class(1)

    # Subtraction belongs to Lesson 6 (second half of Class 1)
    q_sub = Question(
        question_text="Subtract 3 from 7: 7 - 3 = ___",
        question_type=QuestionType.SHORT_ANSWER,
        marks=1,
        difficulty=Difficulty.EASY,
        topic="Lesson 6: Subtraction",
        chapter="Lesson 6: Subtraction",
        answer="4",
    )
    is_forb, reason = is_question_in_forbidden_units(q_sub, ch1_second_half)
    assert is_forb is True
    assert "forbidden chapter" in reason.lower()

    # Addition up to 9 belongs to Lesson 5 (first half of Class 1)
    q_add = Question(
        question_text="Add 3 and 4: 3 + 4 = ___",
        question_type=QuestionType.SHORT_ANSWER,
        marks=1,
        difficulty=Difficulty.EASY,
        topic="Lesson 5: Addition",
        chapter="Lesson 5: Addition (Sum not more than 9)",
        answer="7",
    )
    is_forb, _ = is_question_in_forbidden_units(q_add, ch1_second_half)
    assert is_forb is False


# ---------------------------------------------------------------------------
# Test 6: Full Paper Validation
# ---------------------------------------------------------------------------
def test_validate_full_paper_syllabus_scope():
    """Verify validate_full_paper checks permitted chapters and forbidden units."""
    midterm_ch = get_midterm_chapters_for_class(1)
    second_half_ch = get_second_half_chapters_for_class(1)

    q1 = Question(
        question_id="q1",
        question_text="Count the stars: ★ ★ ★. Write the number: ___",
        question_type=QuestionType.SHORT_ANSWER,
        marks=1,
        difficulty=Difficulty.EASY,
        topic="Lesson 3: Digits (1-9)",
        chapter="Lesson 3: Digits (1-9)",
        answer="3",
        solution="Count: 1, 2, 3.",
    )
    sec_a = PaperSection(
        section_id="A",
        title="Section A: Short Answer",
        question_type=QuestionType.SHORT_ANSWER,
        num_questions=1,
        section_marks=1,
        questions=[q1],
    )
    paper_valid = Paper(
        paper_id="valid_midterm_paper",
        title="Class 1 Mathematics — Mid-Semester Examination",
        grade="Class 1",
        subject="Mathematics",
        topic="Class 1 Mathematics",
        exam_type=ExamType.MIDTERM,
        academic_grade="Class 1",
        curriculum_grade="Class 1",
        syllabus_scope="First half of Class 1 syllabus (5 of 9 chapters)",
        permitted_units=midterm_ch,
        forbidden_units=second_half_ch,
        num_questions=1,
        total_marks=1,
        difficulty=Difficulty.EASY,
        sections=[sec_a],
        questions=[q1],
    )
    ok, errors = validate_full_paper(paper_valid)
    assert ok is True
    assert len(errors) == 0

    # Paper with question from forbidden chapter must fail validation
    q_bad = Question(
        question_id="q_bad",
        question_text="Find the difference: 8 - 2 = ___",
        question_type=QuestionType.SHORT_ANSWER,
        marks=1,
        difficulty=Difficulty.EASY,
        topic="Lesson 6: Subtraction",
        chapter="Lesson 6: Subtraction",
        answer="6",
        solution="8 - 2 = 6.",
    )
    sec_bad = PaperSection(
        section_id="A",
        title="Section A: Short Answer",
        question_type=QuestionType.SHORT_ANSWER,
        num_questions=1,
        section_marks=1,
        questions=[q_bad],
    )
    paper_invalid = Paper(
        paper_id="invalid_midterm_paper",
        title="Class 1 Mathematics — Mid-Semester Examination",
        grade="Class 1",
        subject="Mathematics",
        topic="Class 1 Mathematics",
        exam_type=ExamType.MIDTERM,
        academic_grade="Class 1",
        curriculum_grade="Class 1",
        syllabus_scope="First half of Class 1 syllabus",
        permitted_units=midterm_ch,
        forbidden_units=second_half_ch,
        num_questions=1,
        total_marks=1,
        difficulty=Difficulty.EASY,
        sections=[sec_bad],
        questions=[q_bad],
    )
    ok, errors = validate_full_paper(paper_invalid)
    assert ok is False
    assert any("forbidden chapter" in e.lower() or "not belong to permitted" in e.lower() for e in errors)


# ---------------------------------------------------------------------------
# Test 7: API Endpoint Behavior for All Exam Types
# ---------------------------------------------------------------------------
def test_api_generate_class_1_prerequisite_fails(client):
    """Class 1 Prerequisite must return HTTP 422 with a clear explanation."""
    req_data = {
        "grade": "1",
        "subject": "Mathematics",
        "exam_type": "prerequisite",
        "num_questions": 5,
        "total_marks": 10,
        "question_types": ["short_answer"],
    }
    resp = client.post("/papers/generate", json=req_data)
    assert resp.status_code == 422
    assert "Prerequisite Examination is not available for Class 1" in resp.json()["detail"]


@patch("app.generator.get_llm")
def test_api_generate_class_1_midterm_uses_first_5_chapters(mock_get_llm, client):
    """Class 1 Mid-Semester must use Lessons 1-5 and exclude Lessons 6-9."""
    fake_llm = MagicMock()
    mock_get_llm.return_value = fake_llm

    fake_response = MagicMock()
    fake_response.content = """[
        {
            "question_text": "Count the balls: ● ● ● ●. Write the number: ___",
            "question_type": "short_answer",
            "marks": 2,
            "difficulty": "easy",
            "topic": "Lesson 3: Digits (1-9)",
            "chapter": "Lesson 3: Digits (1-9)",
            "answer": "4",
            "solution": "1, 2, 3, 4."
        }
    ]"""
    fake_llm.invoke.return_value = fake_response

    req_data = {
        "grade": "1",
        "subject": "Mathematics",
        "exam_type": "midterm",
        "num_questions": 1,
        "total_marks": 2,
        "question_types": ["short_answer"],
    }
    resp = client.post("/papers/generate", json=req_data)
    assert resp.status_code == 200, resp.text
    data = resp.json()

    assert data["exam_type"] == "midterm"
    assert len(data["permitted_units"]) == 5
    assert len(data["forbidden_units"]) == 4
    assert data["permitted_units"] == get_midterm_chapters_for_class(1)
    assert data["forbidden_units"] == get_second_half_chapters_for_class(1)
    assert "First half of Class 1 syllabus" in data["syllabus_scope"]
    assert "5 chapters" in data["syllabus_scope"]


@patch("app.generator.get_llm")
def test_api_generate_class_2_prerequisite_uses_class_1(mock_get_llm, client):
    """Class 2 Prerequisite must use Class 1 textbook (all 9 chapters)."""
    fake_llm = MagicMock()
    mock_get_llm.return_value = fake_llm

    fake_response = MagicMock()
    fake_response.content = """[
        {
            "question_text": "Count the apples: ★ ★ ★. Write the number: ___",
            "question_type": "short_answer",
            "marks": 2,
            "difficulty": "easy",
            "topic": "Lesson 3: Digits (1-9)",
            "chapter": "Lesson 3: Digits (1-9)",
            "answer": "3",
            "solution": "Count: 3."
        }
    ]"""
    fake_llm.invoke.return_value = fake_response

    req_data = {
        "grade": "2",
        "subject": "Mathematics",
        "exam_type": "prerequisite",
        "num_questions": 1,
        "total_marks": 2,
        "question_types": ["short_answer"],
    }
    resp = client.post("/papers/generate", json=req_data)
    assert resp.status_code == 200, resp.text
    data = resp.json()

    assert data["exam_type"] == "prerequisite"
    assert data["academic_grade"] == "Class 2"
    assert data["curriculum_grade"] == "Class 1"
    assert len(data["permitted_units"]) == 9
    assert len(data["forbidden_units"]) == 0
    assert data["permitted_units"] == get_chapters_for_class(1)
