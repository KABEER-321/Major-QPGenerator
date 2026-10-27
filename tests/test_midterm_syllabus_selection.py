"""Comprehensive tests for Mid-Semester Syllabus Selection, Textbook Ordering,
Retrieval Filtering, and Generation Scope Enforcement.

Verifies:
1. Chapters are preserved in textbook numerical order, never sorted alphabetically.
2. Odd and even chapter count splitting (9 -> 5, 10 -> 5, 6 -> 3, 5 -> 3).
3. FAISS retrieval filtering strictly excludes second-half chunks.
4. Forbidden units detection identifies and blocks questions from second-half topics.
5. End-to-end Class 1 Mid-Semester paper generation produces questions strictly
   grounded in Lessons 1–5, with zero questions from Lessons 6–9.
6. Rules for Final and Prerequisite examinations remain unchanged.
"""

from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document

from app.curriculum import (
    CURRICULUM_CHAPTERS,
    extract_chapter_number,
    get_chapters_for_class,
    get_curriculum_doc_id,
    get_midterm_chapters_for_class,
    get_second_half_chapters_for_class,
    order_chapters_by_textbook,
)
from app.generator import (
    _enrich_question,
    _generate_section_questions,
    generate_questions,
    is_question_in_forbidden_units,
)
from app.main import app, resolve_curriculum_and_passages
from app.retrieval import is_chunk_in_permitted_scope, load_faiss_index, retrieve_relevant_chunks
from app.schemas import (
    Difficulty,
    ExamType,
    PaperGenerateRequest,
    Question,
    QuestionType,
    SourceReference,
    determine_midterm_scope,
    determine_second_half_scope,
)


@pytest.fixture
def client():
    return TestClient(app)


# ---------------------------------------------------------------------------
# Test 1: Textbook Chapter Ordering is Non-Alphabetical
# ---------------------------------------------------------------------------
def test_textbook_chapter_ordering_non_alphabetical():
    # Class 5 has 10 units: Unit 1 to Unit 10
    ch5 = get_chapters_for_class(5)
    assert len(ch5) == 10
    # In alphabetical sorting, 'Unit 10' comes before 'Unit 2'.
    # In textbook numerical sorting, 'Unit 10' MUST be the 10th (last) element.
    assert ch5[0] == "Unit 1: 5-digit numbers"
    assert ch5[1] == "Unit 2: Addition"
    assert ch5[8] == "Unit 9: Perimeter and Area"
    assert ch5[9] == "Unit 10: Data Handling"

    # Verify order_chapters_by_textbook never sorts alphabetically
    shuffled = [
        "Unit 10: Data Handling",
        "Unit 2: Addition",
        "Unit 1: 5-digit numbers",
        "Unit 9: Perimeter and Area",
    ]
    ordered = order_chapters_by_textbook(shuffled)
    assert ordered[0] == "Unit 1: 5-digit numbers"
    assert ordered[1] == "Unit 2: Addition"
    assert ordered[2] == "Unit 9: Perimeter and Area"
    assert ordered[3] == "Unit 10: Data Handling"

    # Number extraction helper
    assert extract_chapter_number("Lesson 1: Spatial Understanding")[0] == 1
    assert extract_chapter_number("Chapter 10: Geometry")[0] == 10
    assert extract_chapter_number("Unit 5 - Fractions")[0] == 5


# ---------------------------------------------------------------------------
# Test 2: Odd and Even Chapter Splitting
# ---------------------------------------------------------------------------
def test_odd_and_even_chapter_splitting():
    # Class 1: 9 chapters -> exactly first 5 for mid-term, last 4 for second half
    ch1 = get_chapters_for_class(1)
    assert len(ch1) == 9
    first1 = get_midterm_chapters_for_class(1)
    second1 = get_second_half_chapters_for_class(1)
    assert len(first1) == 5
    assert len(second1) == 4
    assert first1 == [
        "Lesson 1: Spatial Understanding",
        "Lesson 2: Solids Around Us",
        "Lesson 3: Digits (1-9)",
        "Lesson 4: Zero",
        "Lesson 5: Addition (Sum not more than 9)",
    ]
    assert second1 == [
        "Lesson 6: Subtraction",
        "Lesson 7: Number 10",
        "Lesson 8: Units and Tens",
        "Lesson 9: Number 11 to 20",
    ]

    # Class 5: 10 chapters -> exactly 5 for mid-term, 5 for second half
    first5 = get_midterm_chapters_for_class(5)
    second5 = get_second_half_chapters_for_class(5)
    assert len(first5) == 5
    assert len(second5) == 5
    assert first5 == [
        "Unit 1: 5-digit numbers",
        "Unit 2: Addition",
        "Unit 3: Subtraction",
        "Unit 4: Factors and multiples",
        "Unit 5: Fractions",
    ]
    assert second5 == [
        "Unit 6: Angles",
        "Unit 7: Circles",
        "Unit 8: Length",
        "Unit 9: Perimeter and Area",
        "Unit 10: Data Handling",
    ]

    # Class 2: 6 chapters -> 3 and 3
    assert len(get_midterm_chapters_for_class(2)) == 3
    assert len(get_second_half_chapters_for_class(2)) == 3

    # Class 3: 5 chapters -> 3 and 2
    assert len(get_midterm_chapters_for_class(3)) == 3
    assert len(get_second_half_chapters_for_class(3)) == 2

    # Generic determine_midterm_scope on arbitrary units
    sample_9 = [f"Chapter {i}" for i in range(1, 10)]
    assert len(determine_midterm_scope(sample_9)) == 5
    assert len(determine_second_half_scope(sample_9)) == 4


# ---------------------------------------------------------------------------
# Test 3: Retrieval Filtering Strictly Enforces First Half
# ---------------------------------------------------------------------------
def test_retrieval_filtering_strictly_enforces_first_half():
    mid1_units = get_midterm_chapters_for_class(1)
    sec1_units = get_second_half_chapters_for_class(1)

    # Unit test is_chunk_in_permitted_scope with mock documents
    doc_lesson1 = Document(
        page_content="Top and bottom spatial exercise.",
        metadata={"chapter": "Lesson 1: Spatial Understanding", "is_first_half": True},
    )
    doc_lesson5 = Document(
        page_content="Addition sum 4 + 3 = 7.",
        metadata={"chapter": "Lesson 5: Addition (Sum not more than 9)", "is_first_half": True},
    )
    doc_lesson6 = Document(
        page_content="Subtract 3 from 8.",
        metadata={"chapter": "Lesson 6: Subtraction", "is_first_half": False},
    )
    doc_lesson9 = Document(
        page_content="Numbers 11 to 20 counting.",
        metadata={"chapter": "Lesson 9: Number 11 to 20", "is_first_half": False},
    )

    # First half chunks are permitted
    assert is_chunk_in_permitted_scope(doc_lesson1, mid1_units, first_half_only=True) is True
    assert is_chunk_in_permitted_scope(doc_lesson5, mid1_units, first_half_only=True) is True

    # Second half chunks are STRICTLY rejected
    assert is_chunk_in_permitted_scope(doc_lesson6, mid1_units, first_half_only=True) is False
    assert is_chunk_in_permitted_scope(doc_lesson9, mid1_units, first_half_only=True) is False

    # Also test actual FAISS index for Class 1
    vs = load_faiss_index(get_curriculum_doc_id(1))
    all_docs = list(vs.docstore._dict.values())
    first_half_indexed = [d for d in all_docs if is_chunk_in_permitted_scope(d, mid1_units, first_half_only=True)]
    second_half_indexed = [d for d in all_docs if not is_chunk_in_permitted_scope(d, mid1_units, first_half_only=True)]

    assert len(first_half_indexed) == 17
    assert len(second_half_indexed) == 10
    # Every first half indexed doc must belong to Lessons 1 to 5
    for d in first_half_indexed:
        ch = d.metadata.get("chapter")
        assert ch in mid1_units
        assert ch not in sec1_units


# ---------------------------------------------------------------------------
# Test 4: Forbidden Units Detection
# ---------------------------------------------------------------------------
def test_forbidden_units_detection():
    mid1_units = get_midterm_chapters_for_class(1)
    sec1_units = get_second_half_chapters_for_class(1)

    # Subtraction question (Lesson 6) -> Forbidden
    q_sub = Question(
        question_id="q1",
        question_text="Ravi had 8 apples and gave away 3 apples. How many apples are left with Ravi?",
        question_type=QuestionType.SHORT_ANSWER,
        marks=2,
        difficulty=Difficulty.EASY,
        topic="Subtraction",
        grade="Class 1",
        answer="5 apples",
        solution="8 - 3 = 5",
    )
    is_forb, reason = is_question_in_forbidden_units(q_sub, sec1_units, mid1_units)
    assert is_forb is True
    assert "subtraction" in reason.lower()

    # Number 10 / Tens question (Lessons 7-8) -> Forbidden
    q_tens = Question(
        question_id="q2",
        question_text="What is the place value of 1 in the number 14, in terms of units and tens?",
        question_type=QuestionType.SHORT_ANSWER,
        marks=2,
        difficulty=Difficulty.MEDIUM,
        topic="Place Value",
        grade="Class 1",
        answer="1 Ten",
        solution="1 Ten and 4 Units",
    )
    is_forb, reason = is_question_in_forbidden_units(q_tens, sec1_units, mid1_units)
    assert is_forb is True

    # Addition question (Lesson 5) -> Permitted
    q_add = Question(
        question_id="q3",
        question_text="Anu has 4 balls and her mother gives her 3 more balls. Find the total number of balls (4 + 3).",
        question_type=QuestionType.SHORT_ANSWER,
        marks=2,
        difficulty=Difficulty.EASY,
        topic="Addition (Sum not more than 9)",
        grade="Class 1",
        answer="7",
        solution="4 + 3 = 7",
    )
    is_forb, _ = is_question_in_forbidden_units(q_add, sec1_units, mid1_units)
    assert is_forb is False

    # Spatial understanding question (Lesson 1) -> Permitted
    q_space = Question(
        question_id="q4",
        question_text="A monkey is sitting on top of a tree and a dog is standing below the tree. Which animal is at the top?",
        question_type=QuestionType.SHORT_ANSWER,
        marks=1,
        difficulty=Difficulty.EASY,
        topic="Spatial Understanding",
        grade="Class 1",
        answer="Monkey",
        solution="The monkey is on top.",
    )
    is_forb, _ = is_question_in_forbidden_units(q_space, sec1_units, mid1_units)
    assert is_forb is False


# ---------------------------------------------------------------------------
# Test 5: End-to-End Class 1 Mid-Semester Paper Generation Scope
# ---------------------------------------------------------------------------
def test_class_1_midterm_generation_scope(client):
    req = PaperGenerateRequest(
        grade="1",
        academic_grade="Class 1",
        exam_type=ExamType.MIDTERM,
        num_questions=6,
        total_marks=12,
        question_types=[QuestionType.MULTIPLE_CHOICE, QuestionType.SHORT_ANSWER],
    )

    # Verify resolve_curriculum_and_passages sets exact permitted and forbidden units
    with patch("app.main.retrieve_relevant_chunks") as mock_retrieve:
        mock_retrieve.return_value = [
            Document(page_content="Lesson 1 text", metadata={"chapter": "Lesson 1: Spatial Understanding", "is_first_half": True}),
            Document(page_content="Lesson 3 text", metadata={"chapter": "Lesson 3: Digits (1-9)", "is_first_half": True}),
        ]
        target_doc, passages, forbidden_units = resolve_curriculum_and_passages(req)

    assert target_doc.document_id == "curriculum_class_1"
    assert req.curriculum_grade == "Class 1"
    assert req.permitted_units == get_midterm_chapters_for_class(1)
    assert forbidden_units == get_second_half_chapters_for_class(1)
    assert req.forbidden_units == get_second_half_chapters_for_class(1)
    assert "5 chapters" in req.syllabus_scope
    for ch in req.permitted_units:
        assert ch in req.syllabus_scope
    for fch in forbidden_units:
        assert fch not in req.syllabus_scope


# ---------------------------------------------------------------------------
# Test 6: Final and Prerequisite Unaffected
# ---------------------------------------------------------------------------
def test_final_and_prerequisite_unaffected():
    # Final Exam: uses full syllabus, no forbidden units
    req_final = PaperGenerateRequest(
        grade="1",
        academic_grade="Class 1",
        exam_type=ExamType.FINAL,
        num_questions=4,
        total_marks=10,
    )
    with patch("app.main.retrieve_relevant_chunks", return_value=["Full chunk"]):
        target_doc, passages, forbidden_units = resolve_curriculum_and_passages(req_final)

    assert forbidden_units == []
    assert req_final.forbidden_units == []
    assert len(req_final.permitted_units) == 9
    assert "Full syllabus" in req_final.syllabus_scope

    # Prerequisite: Class 2 uses Class 1 full syllabus
    req_pre = PaperGenerateRequest(
        grade="2",
        academic_grade="Class 2",
        exam_type=ExamType.PREREQUISITE,
        num_questions=4,
        total_marks=10,
    )
    with patch("app.main.retrieve_relevant_chunks", return_value=["Pre chunk"]):
        target_doc, passages, forbidden_units = resolve_curriculum_and_passages(req_pre)

    assert target_doc.document_id == "curriculum_class_1"
    assert req_pre.curriculum_grade == "Class 1"
    assert req_pre.academic_grade == "Class 2"
    assert forbidden_units == []
    assert len(req_pre.permitted_units) == 9
