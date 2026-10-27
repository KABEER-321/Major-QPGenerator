# tests/test_question_quality.py
# Tests for question quality, clarity, cross-section non-duplication,
# single-scenario coherence, and paper validation.

import pytest
from app.schemas import (
    Question,
    QuestionType,
    Difficulty,
    ValidationStatus,
    Paper,
    PaperSection,
    ExamType,
)
from app.duplicates import (
    is_duplicate,
    normalize_question_text,
    mask_numbers_and_names,
    identify_scenario_cluster,
)
from app.validator import (
    check_question_clarity,
    check_question_coherence,
    validate_question_quality,
    validate_paper_questions_uniqueness,
    validate_full_paper,
)


def _make_q(
    q_id: str,
    text: str,
    qtype: QuestionType = QuestionType.SHORT_ANSWER,
    marks: int = 2,
    options: list[str] = None,
    correct_option: str = None,
) -> Question:
    return Question(
        question_id=q_id,
        question_text=text,
        question_type=qtype,
        marks=marks,
        difficulty=Difficulty.MEDIUM,
        topic="Numbers",
        answer="Answer",
        solution="Step 1: calculate. Step 2: result.",
        options=options,
        correct_option=correct_option,
        grade="Class 5",
        subject="Mathematics",
        validation_status=ValidationStatus.VALID,
    )


# ---------------------------------------------------------------------------
# 1. Tests for Duplicate Prevention Across Sections & Scenario Reskins
# ---------------------------------------------------------------------------

def test_detect_bank_deposit_duplicate_across_sections():
    """Q2 (Sec A) and Q13 (Sec C) repeating bank-deposit problem must be flagged as duplicate."""
    q_sec_a = _make_q(
        "q2",
        "A man deposited Rs. 15,400 in January and Rs. 8,750 in February. How much money did he deposit in total?",
        QuestionType.SHORT_ANSWER,
        marks=2,
    )
    q_sec_c = _make_q(
        "q13",
        "Suresh deposited Rs. 24,500 in a bank on Monday and Rs. 12,300 on Tuesday. Find the total amount deposited by him.",
        QuestionType.LONG_ANSWER,
        marks=4,
    )

    dup, reason = is_duplicate(q_sec_c, [q_sec_a])
    assert dup is True
    assert "Banking" in reason or "template" in reason.lower() or "deposit" in reason.lower()


def test_detect_factory_production_duplicate_across_sections():
    """Q3 (Sec A) and Q12 (Sec B) repeating factory-box subtraction problem must be flagged as duplicate."""
    q_sec_a = _make_q(
        "q3",
        "A factory produced 45,000 soap boxes. 12,350 boxes were sold. How many boxes are left in the factory?",
        QuestionType.SHORT_ANSWER,
        marks=2,
    )
    q_sec_b = _make_q(
        "q12",
        "A factory manufactured 60,000 cartons. If 25,400 cartons were shipped to a warehouse, how many cartons remain in the factory?",
        QuestionType.LONG_ANSWER,
        marks=4,
    )

    dup, reason = is_duplicate(q_sec_b, [q_sec_a])
    assert dup is True
    assert "Factory" in reason or "carton" in reason.lower() or "manufacture" in reason.lower() or "template" in reason.lower()


def test_detect_village_population_duplicate_across_sections():
    """Q8 and Q13 repeating village-population breakdown problem must be flagged as duplicate."""
    q8 = _make_q(
        "q8",
        "The population of a village is 8,450. If 4,200 are men and 2,150 are women, find the number of children in the village.",
        QuestionType.SHORT_ANSWER,
        marks=2,
    )
    q13 = _make_q(
        "q13",
        "In a town of 65,000 residents, there are 32,000 males and 24,000 females. How many children live in the town?",
        QuestionType.LONG_ANSWER,
        marks=4,
    )

    dup, reason = is_duplicate(q13, [q8])
    assert dup is True
    assert "Population" in reason or "template" in reason.lower() or "children" in reason.lower()


def test_allow_distinct_mathematical_topics():
    """Distinct mathematical topics and scenarios must NOT be flagged as duplicates."""
    q_geom = _make_q("g1", "Find the perimeter of a rectangle whose length is 14 cm and breadth is 9 cm.")
    q_time = _make_q("t1", "A train departed from Delhi at 08:30 AM and reached Agra at 11:45 AM. Find the total journey time.")
    q_arith = _make_q("a1", "Find the sum of 45,892 and 37,214.")
    q_frac = _make_q("f1", "Ravi ate 2/5 of a watermelon and his sister ate 1/5. What fraction of the watermelon did they eat together?")

    assert is_duplicate(q_geom, [q_time, q_arith, q_frac])[0] is False
    assert is_duplicate(q_time, [q_geom, q_arith, q_frac])[0] is False
    assert is_duplicate(q_arith, [q_geom, q_time, q_frac])[0] is False
    assert is_duplicate(q_frac, [q_geom, q_time, q_arith])[0] is False


# ---------------------------------------------------------------------------
# 2. Tests for Never Combining Unrelated Questions (Coherence)
# ---------------------------------------------------------------------------

def test_reject_incoherent_combination_of_unrelated_domains():
    """A question mashing up factory production and bank deposit must be rejected."""
    bad_mashup = (
        "A factory manufactured 50,000 soap cartons in June. "
        "The owner deposited Rs. 35,000 into his bank account and withdrew Rs. 10,000. "
        "Find how many cartons are left and the remaining bank balance."
    )
    ok, err = check_question_coherence(bad_mashup)
    assert ok is False
    assert "Incoherently combined unrelated domain scenarios" in err
    assert "Banking" in err and "Factory" in err


def test_reject_incoherent_combination_village_and_railway():
    """A question mashing village demographics and train express travel must be rejected."""
    bad_mashup = (
        "The village population is 12,500. A passenger train departed from the railway platform at 40 km/h. "
        "Calculate the number of female residents when the train arrives."
    )
    ok, err = check_question_coherence(bad_mashup)
    assert ok is False
    assert "combined unrelated" in err.lower()


def test_allow_coherent_multi_step_problem():
    """A multi-step problem with a single coherent storyline must pass."""
    good_multi_step = (
        "A farmer harvested 3,450 kg of wheat in April and 2,850 kg in May. "
        "He packed the wheat into bags of 50 kg each. "
        "(a) Find the total weight of wheat harvested. "
        "(b) How many bags did he fill in total?"
    )
    ok, err = check_question_coherence(good_multi_step)
    assert ok is True
    assert err is None


# ---------------------------------------------------------------------------
# 3. Tests for Question Clarity & Completeness
# ---------------------------------------------------------------------------

def test_reject_missing_visual_reference():
    """Questions referencing 'as shown in the figure below' without diagrams must be rejected."""
    bad_q = "Look at the figure below and find the measure of angle ABC."
    ok, err = check_question_clarity(bad_q, QuestionType.SHORT_ANSWER)
    assert ok is False
    assert "missing visual artifact" in err.lower()

    bad_q2 = "From the given table shown below, calculate the total sales."
    ok2, err2 = check_question_clarity(bad_q2, QuestionType.SHORT_ANSWER)
    assert ok2 is False
    assert "missing visual artifact" in err2.lower()


def test_reject_dangling_prompts():
    """Dangling questions ending in 'following:' without options must be rejected."""
    dangling = "Calculate the perimeter of the following:"
    ok, err = check_question_clarity(dangling, QuestionType.SHORT_ANSWER, options=None)
    assert ok is False
    assert "dangling" in err.lower() or "following" in err.lower()


def test_reject_computation_without_numbers():
    """Computation questions that lack required numerical values must be rejected."""
    missing_nums = "Find the sum of the two numbers."
    ok, err = check_question_clarity(missing_nums, QuestionType.SHORT_ANSWER)
    assert ok is False
    assert "numerical values" in err.lower()


def test_reject_invalid_mcq_completeness():
    """MCQs with fewer than 4 options or mismatched correct_option must fail."""
    q_few_opts = _make_q(
        "mcq1",
        "What is 25 + 75?",
        QuestionType.MULTIPLE_CHOICE,
        marks=1,
        options=["50", "100", "125"],
        correct_option="100",
    )
    ok, err = validate_question_quality(q_few_opts)
    assert ok is False
    assert "at least 4 options" in err

    # Model construct bypasses Pydantic model validation to test validate_question_quality directly
    q_mismatched = Question.model_construct(
        question_id="mcq2",
        question_text="What is 25 + 75?",
        question_type=QuestionType.MULTIPLE_CHOICE,
        marks=1,
        difficulty=Difficulty.MEDIUM,
        topic="Numbers",
        answer="100",
        solution="25 + 75 = 100",
        options=["50", "75", "100", "125"],
        correct_option="99",  # not in options
        grade="Class 5",
        subject="Mathematics",
        validation_status=ValidationStatus.VALID,
    )
    ok2, err2 = validate_question_quality(q_mismatched)
    assert ok2 is False
    assert "does not match any" in err2


# ---------------------------------------------------------------------------
# 4. Tests for Paper Validation Across Sections & Marks
# ---------------------------------------------------------------------------

def test_paper_validation_detects_cross_section_duplicates():
    """Paper validation must catch duplicate questions across sections."""
    q1 = _make_q("q1", "A factory produced 45,000 soap boxes. 12,350 boxes were sold. How many are left?", marks=1)
    q2 = _make_q("q2", "Find the perimeter of a rectangle with length 10 cm and breadth 5 cm.", marks=2)
    q3 = _make_q(
        "q3",
        "A factory manufactured 60,000 cartons. 25,000 cartons were shipped. How many cartons remain in the factory?",
        qtype=QuestionType.LONG_ANSWER,
        marks=4,
    )

    sec_a = PaperSection(section_id="sec_a", title="Section A", question_type=QuestionType.SHORT_ANSWER, num_questions=1, section_marks=1, questions=[q1])
    sec_b = PaperSection(section_id="sec_b", title="Section B", question_type=QuestionType.SHORT_ANSWER, num_questions=1, section_marks=2, questions=[q2])
    sec_c = PaperSection(section_id="sec_c", title="Section C", question_type=QuestionType.LONG_ANSWER, num_questions=1, section_marks=4, questions=[q3])

    paper = Paper(
        paper_id="test_dup_paper",
        title="Class 5 Mathematics — Final Examination",
        grade="Class 5",
        subject="Mathematics",
        topic="General Math",
        difficulty=Difficulty.MEDIUM,
        exam_type=ExamType.FINAL,
        total_marks=7,
        num_questions=3,
        sections=[sec_a, sec_b, sec_c],
        questions=[q1, q2, q3],
    )

    valid, errors = validate_full_paper(paper)
    assert valid is False
    assert any("Cross-section duplication error" in e for e in errors)


def test_paper_validation_passes_valid_structured_paper():
    """A well-formed structured paper with unique, clear questions and matching marks must pass."""
    q1 = _make_q(
        "q1",
        "Which of the following is an acute angle?",
        QuestionType.MULTIPLE_CHOICE,
        marks=1,
        options=["45 degrees", "90 degrees", "120 degrees", "180 degrees"],
        correct_option="45 degrees",
    )
    q2 = _make_q(
        "q2",
        "Find the perimeter of a square park with side 25 meters.",
        QuestionType.SHORT_ANSWER,
        marks=2,
    )
    q3 = _make_q(
        "q3",
        "A library has 4,500 books. In the first month, 1,200 books were borrowed and 850 books were returned. How many books were in the library at the end of the month?",
        QuestionType.LONG_ANSWER,
        marks=4,
    )

    sec_a = PaperSection(section_id="sec_a", title="Section A (MCQs)", question_type=QuestionType.MULTIPLE_CHOICE, num_questions=1, section_marks=1, questions=[q1])
    sec_b = PaperSection(section_id="sec_b", title="Section B (Short Answer)", question_type=QuestionType.SHORT_ANSWER, num_questions=1, section_marks=2, questions=[q2])
    sec_c = PaperSection(section_id="sec_c", title="Section C (Long Answer)", question_type=QuestionType.LONG_ANSWER, num_questions=1, section_marks=4, questions=[q3])

    paper = Paper(
        paper_id="test_valid_paper",
        title="Class 5 Mathematics — Final Examination",
        grade="Class 5",
        subject="Mathematics",
        topic="General Math",
        difficulty=Difficulty.MEDIUM,
        exam_type=ExamType.FINAL,
        total_marks=7,
        num_questions=3,
        sections=[sec_a, sec_b, sec_c],
        questions=[q1, q2, q3],
    )

    valid, errors = validate_full_paper(paper)
    assert valid is True
    assert len(errors) == 0
