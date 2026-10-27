# tests/test_class_1_question_quality.py
# Tests for Class 1 question age-appropriateness, mathematical skill focus,
# rejection of definitions/reading comprehension, and accurate source chunk traceability.

import pytest
from langchain_core.documents import Document

from app.schemas import (
    Difficulty,
    ExamType,
    PaperGenerateRequest,
    Question,
    QuestionType,
    SourceReference,
    create_default_blueprint,
)
from app.validator import (
    check_grade_age_appropriateness,
    validate_question_quality,
)
from app.generator import (
    _build_source_references,
    _match_source_reference,
    _enrich_question,
    get_grade_pedagogical_prompt,
)
from app.pdf_export import generate_pdf
from app.schemas import Paper, PaperSection


def test_class_1_blueprint_has_age_appropriate_instructions():
    """Verify that Class 1 blueprint does not instruct 6-year-olds with adult word-problem instructions."""
    bp_class1 = create_default_blueprint(
        num_questions=10,
        total_marks=20,
        question_types=[QuestionType.MULTIPLE_CHOICE, QuestionType.SHORT_ANSWER, QuestionType.LONG_ANSWER],
        grade="Class 1",
    )
    for sec in bp_class1.sections:
        if sec.question_type == QuestionType.LONG_ANSWER:
            assert "Solve the following mathematical sums and number problems." in sec.instructions
            assert "Solve the following word problems" not in sec.instructions
        elif sec.question_type == QuestionType.SHORT_ANSWER:
            assert "Write the correct number or answer in the blank space provided." in sec.instructions
        elif sec.question_type == QuestionType.MULTIPLE_CHOICE:
            assert "Choose the correct option and write its letter in the box." in sec.instructions

    # Older classes still retain standard instructions
    bp_class5 = create_default_blueprint(
        num_questions=10,
        total_marks=20,
        question_types=[QuestionType.LONG_ANSWER],
        grade="Class 5",
    )
    assert "Solve the following word problems and show all calculation steps clearly." in bp_class5.sections[0].instructions


def test_class_1_validator_rejects_definitional_and_english_questions():
    """Class 1 questions must test mathematical competence, not English vocabulary or definitions."""
    bad_questions = [
        "What does the plus symbol mean?",
        "State vocabulary terms used in spatial relations.",
        "Explain what term is used when no monkeys dangling from the tree.",
        "Define addition and explain the rule.",
        "In your own words explain why zero means none.",
        "Write a sentence explaining what inside means.",
        "Which English word describes an object on top of another?",
    ]
    for text in bad_questions:
        q = Question(
            question_text=text,
            question_type=QuestionType.SHORT_ANSWER,
            marks=1,
            difficulty=Difficulty.EASY,
            grade="Class 1",
            chapter="Lesson 1: Spatial Understanding",
            topic="Spatial Understanding",
            answer="Answer",
            solution="Solution steps",
        )
        ok, err = check_grade_age_appropriateness(q)
        assert not ok, f"Expected '{text}' to be rejected, but passed!"
        assert "Class 1 questions must test direct math skills" in str(err)


def test_class_1_validator_rejects_adult_commercial_scenarios():
    """Class 1 questions must not test adult banking, factory production, or census demographics."""
    adult_questions = [
        "A shopkeeper deposited 450 rupees into his bank account and withdrew 120 rupees.",
        "A factory manufactured 852 boxes of soap packets in a warehouse.",
        "The population of a village census showed 500 inhabitants.",
        "A wholesale merchant purchased supplies for his company employees.",
    ]
    for text in adult_questions:
        q = Question(
            question_text=text,
            question_type=QuestionType.SHORT_ANSWER,
            marks=2,
            difficulty=Difficulty.EASY,
            grade="Class 1",
            chapter="Lesson 5: Addition",
            topic="Addition",
            answer="Answer",
            solution="Solution steps",
        )
        ok, err = check_grade_age_appropriateness(q)
        assert not ok, f"Expected adult scenario '{text}' to be rejected, but passed!"
        assert "Class 1 questions cannot use adult" in str(err)


def test_class_1_validator_accepts_age_appropriate_math():
    """Valid Class 1 math questions (counting, number sequence, simple sums) must pass."""
    valid_questions = [
        ("Count the stars: ★ ★ ★ ★. Write the number.", QuestionType.SHORT_ANSWER, 1, "4"),
        ("What number comes after 5?", QuestionType.SHORT_ANSWER, 1, "6"),
        ("Write the number that comes between 7 and 9: ___", QuestionType.SHORT_ANSWER, 1, "8"),
        ("Solve the addition: 3 + 4 = ___", QuestionType.SHORT_ANSWER, 1, "7"),
        ("Which number is greater: 6 or 2?", QuestionType.SHORT_ANSWER, 1, "6"),
        (
            "Solve the following additions: (a) 2 + 3 = ___ (b) 5 + 1 = ___ (c) 4 + 4 = ___",
            QuestionType.LONG_ANSWER,
            3,
            "(a) 5, (b) 6, (c) 8",
        ),
    ]
    for text, qtype, marks, ans in valid_questions:
        q = Question(
            question_text=text,
            question_type=qtype,
            marks=marks,
            difficulty=Difficulty.EASY,
            grade="Class 1",
            chapter="Lesson 5: Addition (Sum not more than 9)",
            topic="Addition",
            answer=ans,
            solution=f"Step: {ans}",
        )
        ok, err = validate_question_quality(q)
        assert ok, f"Expected valid question '{text}' to pass, but got error: {err}"


def test_class_1_pedagogical_prompt_present():
    """Verify that pedagogical prompts are injected for Class 1."""
    prompt_c1 = get_grade_pedagogical_prompt("Class 1", QuestionType.SHORT_ANSWER)
    assert "STUDENTS AGED 6-7 YEARS" in prompt_c1
    assert "MATHEMATICAL SKILLS ONLY - NOT ENGLISH COMPREHENSION" in prompt_c1
    assert "ABSOLUTE PROHIBITION OF DEFINITIONS & ESSAY QUESTIONS" in prompt_c1

    prompt_other = get_grade_pedagogical_prompt("Class 9")
    assert prompt_other == ""

    prompt_c5 = get_grade_pedagogical_prompt("Class 5")
    assert "CLASS 5" in prompt_c5


def test_source_reference_chunk_index_and_chapter_matching():
    """Verify that chunks have correct chunk_index and questions are matched to specific chapter chunks."""
    passages = [
        Document(
            page_content="Spatial understanding top and bottom, inside and outside, on and under.",
            metadata={
                "chunk_id": "c1_ch1_k0",
                "chunk_index": 0,
                "chapter": "Lesson 1: Spatial Understanding",
                "grade": "Class 1",
            },
        ),
        Document(
            page_content="Solid shapes around us: ball rolls, box slides, can rolls and slides.",
            metadata={
                "chunk_id": "c1_ch2_k4",
                "chunk_index": 4,
                "chapter": "Lesson 2: Solids Around Us",
                "grade": "Class 1",
            },
        ),
        Document(
            page_content="Digits 1 to 9. Count objects: 1 one, 2 two, 3 three, 4 four, 5 five.",
            metadata={
                "chunk_id": "c1_ch3_k8",
                "chunk_index": 8,
                "chapter": "Lesson 3: Digits (1-9)",
                "grade": "Class 1",
            },
        ),
    ]

    refs = _build_source_references(passages)
    assert len(refs) == 3
    assert refs[0].source_filename == "Lesson 1: Spatial Understanding"
    assert refs[0].chunk_index == 0
    assert refs[1].source_filename == "Lesson 2: Solids Around Us"
    assert refs[1].chunk_index == 4
    assert refs[2].source_filename == "Lesson 3: Digits (1-9)"
    assert refs[2].chunk_index == 8

    # Match question to specific passage
    q_counting = Question(
        question_text="Count the balls: 1, 2, 3. How many are there?",
        question_type=QuestionType.SHORT_ANSWER,
        marks=1,
        difficulty=Difficulty.EASY,
        grade="Class 1",
        chapter="Lesson 3: Digits (1-9)",
        topic="Digits",
        answer="3",
        solution="3",
    )
    matched = _match_source_reference(q_counting, passages, refs)
    assert len(matched) == 1
    assert matched[0].source_filename == "Lesson 3: Digits (1-9)"
    assert matched[0].chunk_index == 8

    # Enrich question with passages
    req = PaperGenerateRequest(
        grade="Class 1",
        academic_grade="Class 1",
        topic="Mathematics Curriculum",
        num_questions=5,
        total_marks=10,
    )
    enriched = _enrich_question(
        q_counting,
        req,
        refs,
        permitted_units=["Lesson 1: Spatial Understanding", "Lesson 2: Solids Around Us", "Lesson 3: Digits (1-9)"],
        passages=passages,
    )
    assert enriched.chapter == "Lesson 3: Digits (1-9)"
    assert len(enriched.source_references) == 1
    assert enriched.source_references[0].chunk_index == 8
    assert enriched.source_references[0].source_filename == "Lesson 3: Digits (1-9)"


def test_pdf_export_renders_chapter_and_chunk_without_generic_labels():
    """Verify that PDF generation renders the actual chapter and chunk without 'Mathematics Curriculum — Chunk 0'."""
    q1 = Question(
        question_text="Count the apples: 1, 2, 3. How many are there?",
        question_type=QuestionType.SHORT_ANSWER,
        marks=2,
        difficulty=Difficulty.EASY,
        grade="Class 1",
        chapter="Lesson 3: Digits (1-9)",
        topic="Digits",
        answer="3",
        solution="Count: 3 apples",
        source_references=[
            SourceReference(source_filename="Lesson 3: Digits (1-9)", chunk_index=7, passage_preview="...")
        ],
    )
    sec_a = PaperSection(
        section_id="A",
        title="Section A: Short Answer",
        question_type=QuestionType.SHORT_ANSWER,
        num_questions=1,
        section_marks=2,
        instructions="Answer the following.",
        questions=[q1],
    )
    paper = Paper(
        paper_id="test_c1_paper",
        title="Class 1 Mathematics Examination",
        grade="Class 1",
        academic_grade="Class 1",
        subject="Mathematics",
        topic="Mathematics Curriculum",
        difficulty=Difficulty.EASY,
        num_questions=1,
        total_marks=2,
        questions=[q1],
        sections=[sec_a],
    )

    teacher_pdf = generate_pdf(paper, include_answers=True)
    assert len(teacher_pdf) > 1000

    student_pdf = generate_pdf(paper, include_answers=False)
    assert len(student_pdf) > 1000
