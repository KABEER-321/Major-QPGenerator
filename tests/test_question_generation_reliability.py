# tests/test_question_generation_reliability.py
# Tests for section-wise question generation reliability:
# - Blueprint feasibility for Class 2 (25 questions, 100 marks)
# - Missing slot generation on retry without restarting
# - Preservation of accepted questions
# - Clear diagnostic error reporting (no "Last error: None")
# - Handling of invalid JSON, duplicates, and incorrect marks

import json
from unittest.mock import MagicMock, patch
import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from app.schemas import (
    PaperGenerateRequest,
    QuestionType,
    Difficulty,
    ExamType,
    Question,
    ValidationStatus,
)
from app.generator import (
    _generate_section_questions,
    generate_questions,
    _build_source_references,
)
from app.paper import assemble_paper
from app.validator import validate_full_paper


@pytest.fixture
def dummy_passages():
    return [
        Document(
            page_content="Lesson 1 Shapes: Triangle has 3 sides. Rectangle has 4 sides. Circle is round.\n"
                         "Lesson 2 Numbers: Counting up to 100. Place value of tens and ones.\n"
                         "Lesson 3 Addition: Adding two 2-digit numbers.\n"
                         "Lesson 4 Subtraction: Subtracting 2-digit numbers.\n"
                         "Lesson 5 Multiplication: Repeated addition as multiplication.\n"
                         "Lesson 6 Division: Equal sharing.",
            metadata={"source": "class_2_textbook.pdf", "chunk_index": 0},
        )
    ]


@pytest.fixture
def class_2_request():
    return PaperGenerateRequest(
        grade="Class 2",
        subject="Mathematics",
        topic="General Math",
        num_questions=25,
        total_marks=100,
        difficulty=Difficulty.MEDIUM,
        exam_type=ExamType.FINAL,
        question_types=[
            QuestionType.MULTIPLE_CHOICE,
            QuestionType.SHORT_ANSWER,
            QuestionType.LONG_ANSWER,
        ],
    )


# ---------------------------------------------------------------------------
# 1. Blueprint Feasibility Test for Class 2 (25 questions, 100 marks)
# ---------------------------------------------------------------------------

def test_class_2_final_blueprint_feasibility(class_2_request):
    """Verify that Class 2 (25 questions, 100 marks) produces a balanced, feasible blueprint."""
    bp = class_2_request.get_or_create_blueprint()

    assert bp.total_questions == 25
    assert bp.total_marks == 100
    assert len(bp.sections) == 3

    sec_a, sec_b, sec_c = bp.sections[0], bp.sections[1], bp.sections[2]
    assert sec_a.question_type == QuestionType.MULTIPLE_CHOICE
    assert sec_b.question_type == QuestionType.SHORT_ANSWER
    assert sec_c.question_type == QuestionType.LONG_ANSWER

    # Total questions match
    assert sec_a.num_questions + sec_b.num_questions + sec_c.num_questions == 25
    # Total marks match
    assert sec_a.section_marks + sec_b.section_marks + sec_c.section_marks == 100

    # Section counts and integer marks per question
    assert sec_a.num_questions == 10
    assert sec_b.num_questions == 10
    assert sec_c.num_questions == 5

    assert sec_a.marks_per_question == 2
    assert sec_b.marks_per_question == 3
    assert sec_c.marks_per_question == 10


# ---------------------------------------------------------------------------
# 2. Test Section-Wise Generation with Missing Question Retry
# ---------------------------------------------------------------------------

def test_section_generation_preserves_accepted_and_fills_deficit(class_2_request, dummy_passages):
    """
    When Section B needs 10 questions and attempt 1 yields 6 valid questions,
    attempt 2 should request only the remaining 4 questions and complete the section.
    """
    bp = class_2_request.get_or_create_blueprint()
    sec_b = bp.sections[1]  # Section B (10 questions)

    # Attempt 1: Returns 6 valid questions + 2 duplicate/invalid questions
    attempt_1_questions = [
        {"question_text": f"Add {10+i} and {20+i}.", "question_type": "short_answer", "marks": 4, "answer": str(30+2*i), "solution": "Add numbers."}
        for i in range(6)
    ]
    # Add a duplicate that should be rejected
    attempt_1_questions.append(
        {"question_text": "Add 10 and 20.", "question_type": "short_answer", "marks": 4, "answer": "30", "solution": "Duplicate"}
    )
    # Add a dangling prompt that fails quality validation
    attempt_1_questions.append(
        {"question_text": "Calculate the sum of the following:", "question_type": "short_answer", "marks": 4, "answer": "0", "solution": "Dangling"}
    )

    # Attempt 2: Returns the 4 missing questions
    attempt_2_questions = [
        {"question_text": f"Subtract {5+i} from {50+i}.", "question_type": "short_answer", "marks": 4, "answer": str(45), "solution": "Subtract numbers."}
        for i in range(4)
    ]

    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = [
        AIMessage(content=json.dumps(attempt_1_questions)),
        AIMessage(content=json.dumps(attempt_2_questions)),
    ]

    source_refs = _build_source_references(dummy_passages)
    sec_qs = _generate_section_questions(
        sec=sec_b,
        request=class_2_request,
        passages=dummy_passages,
        existing_questions=[],
        source_refs=source_refs,
        llm=mock_llm,
    )

    assert len(sec_qs) == 10
    assert all(q.question_type == QuestionType.SHORT_ANSWER for q in sec_qs)
    assert sum(q.marks for q in sec_qs) == sec_b.section_marks

    # Verify that LLM was invoked twice (initial attempt + 1 retry for missing slots)
    assert mock_llm.invoke.call_count == 2


# ---------------------------------------------------------------------------
# 3. Test Diagnostic Error Reporting (No "Last error: None")
# ---------------------------------------------------------------------------

def test_section_generation_reports_detailed_rejections(class_2_request, dummy_passages):
    """When questions fail validation and attempts are exhausted, the error must detail the reasons."""
    bp = class_2_request.get_or_create_blueprint()
    sec_b = bp.sections[1]

    # Return only 2 valid questions and invalid ones in all attempts
    bad_questions = [
        {"question_text": "Add 10 and 20.", "question_type": "short_answer", "marks": 4, "answer": "30", "solution": "ok"},
        {"question_text": "Look at the figure below and find the angle.", "question_type": "short_answer", "marks": 4, "answer": "90", "solution": "bad"},
    ]

    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(content=json.dumps(bad_questions))

    source_refs = _build_source_references(dummy_passages)

    with pytest.raises(RuntimeError) as exc_info:
        _generate_section_questions(
            sec=sec_b,
            request=class_2_request,
            passages=dummy_passages,
            existing_questions=[],
            source_refs=source_refs,
            llm=mock_llm,
        )

    err_msg = str(exc_info.value)
    assert "Last error: None" not in err_msg
    assert "Obtained" in err_msg
    assert "needed" in err_msg.lower()
    assert "Quality check failed" in err_msg or "Rejection" in err_msg or "Duplicate" in err_msg


# ---------------------------------------------------------------------------
# 4. Test Complete End-to-End Generation of Class 2 Paper (25 Qs, 100 Marks)
# ---------------------------------------------------------------------------

def test_generate_full_class_2_paper_with_mocked_llm(class_2_request, dummy_passages):
    """
    Test full paper generation for Class 2 Final Examination with 25 questions and 100 marks.
    Verifies Section A (10 MCQs), Section B (10 Short Answer), Section C (5 Long Answer).
    """
    bp = class_2_request.get_or_create_blueprint()

    # Section A (10 MCQs at 2 marks each = 20 marks)
    sec_a_data = [
        {
            "question_text": f"What is {i+1} + {i+2}?",
            "question_type": "multiple_choice",
            "marks": 2,
            "answer": str(2*i+3),
            "solution": f"{i+1} + {i+2} = {2*i+3}",
            "options": [str(2*i+1), str(2*i+2), str(2*i+3), str(2*i+4)],
            "correct_option": str(2*i+3),
        }
        for i in range(10)
    ]

    # Section B (10 Short Answer at 4 marks each = 40 marks)
    sec_b_data = [
        {
            "question_text": f"Subtract {i+10} from {i+40}.",
            "question_type": "short_answer",
            "marks": 4,
            "answer": "30",
            "solution": f"({i+40}) - ({i+10}) = 30",
        }
        for i in range(10)
    ]

    # Section C (5 Long Answer at 8 marks each = 40 marks)
    sec_c_data = [
        {
            "question_text": "A fruit basket has 25 apples and 35 bananas. Find the total number of fruits in the basket.",
            "question_type": "long_answer",
            "marks": 8,
            "answer": "60",
            "solution": "25 + 35 = 60 fruits in total.",
        },
        {
            "question_text": "A primary school library had 95 storybooks. During the reading week, 42 books were issued to students. How many storybooks remain in the library?",
            "question_type": "long_answer",
            "marks": 8,
            "answer": "53",
            "solution": "95 - 42 = 53 storybooks remain.",
        },
        {
            "question_text": "In a grade 2 classroom, there are 6 tables. Each table has 4 chairs arranged around it. Calculate the total number of chairs in the classroom.",
            "question_type": "long_answer",
            "marks": 8,
            "answer": "24",
            "solution": "6 * 4 = 24 chairs in total.",
        },
        {
            "question_text": "A toy shop sold 28 toy cars on Saturday and 36 toy cars on Sunday. Find the total number of toy cars sold over the weekend.",
            "question_type": "long_answer",
            "marks": 8,
            "answer": "64",
            "solution": "28 + 36 = 64 toy cars.",
        },
        {
            "question_text": "Ravi has a ribbon of length 80 centimeters. He cuts a piece of length 35 centimeters for his art project. What is the length of ribbon left with him?",
            "question_type": "long_answer",
            "marks": 8,
            "answer": "45",
            "solution": "80 - 35 = 45 centimeters left.",
        },
    ]

    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = [
        AIMessage(content=json.dumps(sec_a_data)),
        AIMessage(content=json.dumps(sec_b_data)),
        AIMessage(content=json.dumps(sec_c_data)),
    ]

    with patch("app.generator.get_llm", return_value=mock_llm):
        questions = generate_questions(class_2_request, dummy_passages, blueprint=bp)

    assert len(questions) == 25
    assert sum(q.marks for q in questions) == 100

    paper = assemble_paper(
        request=class_2_request,
        questions=questions,
        blueprint_id=bp.blueprint_id,
    )

    assert paper.total_marks == 100
    assert paper.num_questions == 25
    assert len(paper.sections) == 3

    # Section A
    assert paper.sections[0].num_questions == 10
    assert paper.sections[0].section_marks == 20
    assert paper.sections[0].question_type == QuestionType.MULTIPLE_CHOICE

    # Section B
    assert paper.sections[1].num_questions == 10
    assert paper.sections[1].section_marks == 30
    assert paper.sections[1].question_type == QuestionType.SHORT_ANSWER

    # Section C
    assert paper.sections[2].num_questions == 5
    assert paper.sections[2].section_marks == 50
    assert paper.sections[2].question_type == QuestionType.LONG_ANSWER

    # Paper validation passes 100%
    is_valid, errors = validate_full_paper(paper)
    assert is_valid is True
    assert len(errors) == 0
