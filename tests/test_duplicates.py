# tests/test_duplicates.py
# Unit tests for text normalization, duplicate detection, and candidate selection.

import pytest
from app.schemas import (
    Question,
    QuestionType,
    Difficulty,
    SectionBlueprint,
    ExamBlueprint,
    MathVerificationResult,
    MathCheckStatus,
)
from app.duplicates import (
    normalize_question_text,
    calculate_similarity,
    is_duplicate,
    deduplicate_candidates,
    select_version_questions,
    CandidatePoolShortageError,
)


def _make_q(q_id: str, text: str, q_type: QuestionType = QuestionType.SHORT_ANSWER, marks: int = 2, **kwargs) -> Question:
    opts = kwargs.get("options")
    corr = kwargs.get("correct_option")
    if q_type == QuestionType.MULTIPLE_CHOICE and not opts:
        opts = ["A", "B", "C", "D"]
        corr = "A"
    return Question(
        question_id=q_id,
        question_text=text,
        question_type=q_type,
        marks=marks,
        difficulty=kwargs.get("difficulty", Difficulty.MEDIUM),
        topic="Algebra",
        answer=kwargs.get("answer", "4"),
        solution=kwargs.get("solution", "Solution details"),
        options=opts,
        correct_option=corr,
        math_verification=kwargs.get("math_verification"),
    )


class TestNormalizationAndSimilarity:
    """Tests for text normalization and similarity metrics."""

    def test_strip_question_prefixes(self):
        assert normalize_question_text("1. Find the value of x") == "value of x"
        assert normalize_question_text("Q2. Calculate the roots of equation") == "roots of equation"
        assert normalize_question_text("Question 3: What is the perimeter?") == "perimeter"
        assert normalize_question_text("(a) Solve the following: 2x + 1 = 5") == "2x + 1 = 5"

    def test_math_symbol_normalization(self):
        s1 = normalize_question_text("x² + 2x − 3 = 0")
        assert "^2" in s1
        assert "-" in s1
        s2 = normalize_question_text("5 × 4 ÷ 2")
        assert "*" in s2
        assert "/" in s2

    def test_identical_similarity(self):
        assert calculate_similarity("Find the roots of x^2 - 4 = 0", "Find the roots of x^2 - 4 = 0") == 1.0

    def test_near_duplicate_high_similarity(self):
        q1 = "Calculate the roots of the quadratic equation 2x^2 + 5x - 3 = 0"
        q2 = "Find the roots of quadratic equation 2x^2 + 5x - 3 = 0"
        sim = calculate_similarity(q1, q2)
        assert sim > 0.75

    def test_different_questions_low_similarity(self):
        q1 = "Find the roots of quadratic equation x^2 + 5x + 6 = 0"
        q2 = "State and prove Pythagoras theorem for a right-angled triangle"
        sim = calculate_similarity(q1, q2)
        assert sim < 0.35


class TestDuplicateDetection:
    """Tests for is_duplicate and candidate deduplication."""

    def test_detect_exact_duplicate(self):
        q1 = _make_q("q1", "What is the square root of 64?")
        q2 = _make_q("q2", "What is the square root of 64?")
        dup, reason = is_duplicate(q2, [q1])
        assert dup is True
        assert "Exact normalized text match" in (reason or "")

    def test_detect_near_duplicate(self):
        q1 = _make_q("q1", "Find the area of a circular field with radius 7 meters.")
        q2 = _make_q("q2", "Find the area of that circular field with radius 7 meters.")
        dup, reason = is_duplicate(q2, [q1])
        assert dup is True
        assert "High textual similarity" in (reason or "")

    def test_distinct_questions_not_duplicate(self):
        q1 = _make_q("q1", "Find the value of sin 30 degrees.")
        q2 = _make_q("q2", "Find the value of cos 60 degrees.")
        dup, _ = is_duplicate(q2, [q1])
        assert dup is False

    def test_mcq_same_options_near_duplicate(self):
        q1 = _make_q(
            "q1",
            "Which of the following is a prime number?",
            q_type=QuestionType.MULTIPLE_CHOICE,
            marks=1,
            options=["4", "6", "7", "9"],
            correct_option="7",
        )
        q2 = _make_q(
            "q2",
            "Identify the prime number among the options below:",
            q_type=QuestionType.MULTIPLE_CHOICE,
            marks=1,
            options=["4", "6", "7", "9"],
            correct_option="7",
        )
        dup, _ = is_duplicate(q2, [q1])
        assert dup is True

    def test_deduplicate_candidates(self):
        candidates = [
            _make_q("q1", "Find the perimeter of rectangle with length 5 and width 3."),
            _make_q("q2", "Calculate the perimeter of rectangle with length 5 and width 3."),
            _make_q("q3", "Find the area of triangle with base 4 and height 6."),
        ]
        unique, rejected = deduplicate_candidates(candidates)
        assert len(unique) == 2
        assert rejected == 1
        assert unique[0].question_id == "q1"
        assert unique[1].question_id == "q3"


class TestCandidatePoolSelection:
    """Tests for select_version_questions across multiple versions."""

    @pytest.fixture
    def sample_blueprint(self):
        sec_a = SectionBlueprint(
            section_id="A",
            title="Section A: MCQs",
            question_type=QuestionType.MULTIPLE_CHOICE,
            num_questions=2,
            section_marks=2,
            marks_per_question=1,
        )
        sec_b = SectionBlueprint(
            section_id="B",
            title="Section B: Short Answer",
            question_type=QuestionType.SHORT_ANSWER,
            num_questions=1,
            section_marks=2,
            marks_per_question=2,
        )
        return ExamBlueprint(
            title="Test Blueprint",
            sections=[sec_a, sec_b],
            total_questions=3,
            total_marks=4,
        )

    def test_select_two_unique_versions(self, sample_blueprint):
        # We need 2 versions:
        # Section A needs 2 * 2 = 4 MCQs
        # Section B needs 1 * 2 = 2 Short Answer
        candidates = [
            _make_q("mcq1", "MCQ 1 text", q_type=QuestionType.MULTIPLE_CHOICE, marks=1),
            _make_q("mcq2", "MCQ 2 text", q_type=QuestionType.MULTIPLE_CHOICE, marks=1),
            _make_q("mcq3", "MCQ 3 text", q_type=QuestionType.MULTIPLE_CHOICE, marks=1),
            _make_q("mcq4", "MCQ 4 text", q_type=QuestionType.MULTIPLE_CHOICE, marks=1),
            _make_q("sa1", "SA 1 text", q_type=QuestionType.SHORT_ANSWER, marks=2),
            _make_q("sa2", "SA 2 text", q_type=QuestionType.SHORT_ANSWER, marks=2),
        ]

        versions_q, versions_sec = select_version_questions(candidates, sample_blueprint, num_versions=2)

        assert len(versions_q) == 2
        assert len(versions_sec) == 2

        # Each version has 3 questions and 4 total marks
        for v_idx in range(2):
            qs = versions_q[v_idx]
            assert len(qs) == 3
            assert sum(q.marks for q in qs) == 4
            secs = versions_sec[v_idx]
            assert len(secs) == 2
            assert secs[0].num_questions == 2
            assert secs[0].section_marks == 2
            assert secs[1].num_questions == 1
            assert secs[1].section_marks == 2

        # Cross-version uniqueness: all selected question IDs are disjoint!
        v1_ids = {q.question_id for q in versions_q[0]}
        v2_ids = {q.question_id for q in versions_q[1]}
        assert v1_ids.isdisjoint(v2_ids)

    def test_reject_math_discrepancy_candidate(self, sample_blueprint):
        # One candidate has MathCheckStatus.DISCREPANCY
        bad_mcq = _make_q(
            "bad_mcq",
            "Bad math MCQ",
            q_type=QuestionType.MULTIPLE_CHOICE,
            marks=1,
            math_verification=MathVerificationResult(
                status=MathCheckStatus.DISCREPANCY,
                details="Roots did not satisfy equation",
            ),
        )
        good_mcqs = [
            _make_q(f"mcq_{i}", f"Valid MCQ {i}", q_type=QuestionType.MULTIPLE_CHOICE, marks=1)
            for i in range(4)
        ]
        good_sas = [
            _make_q(f"sa_{i}", f"Valid SA {i}", q_type=QuestionType.SHORT_ANSWER, marks=2)
            for i in range(2)
        ]
        pool = [bad_mcq] + good_mcqs + good_sas

        versions_q, _ = select_version_questions(pool, sample_blueprint, num_versions=2)
        all_selected_ids = {q.question_id for v in versions_q for q in v}
        assert "bad_mcq" not in all_selected_ids

    def test_candidate_shortage_raises_error(self, sample_blueprint):
        # Only 3 MCQs provided when 4 are required for 2 versions
        candidates = [
            _make_q("mcq1", "MCQ 1", q_type=QuestionType.MULTIPLE_CHOICE, marks=1),
            _make_q("mcq2", "MCQ 2", q_type=QuestionType.MULTIPLE_CHOICE, marks=1),
            _make_q("mcq3", "MCQ 3", q_type=QuestionType.MULTIPLE_CHOICE, marks=1),
            _make_q("sa1", "SA 1", q_type=QuestionType.SHORT_ANSWER, marks=2),
            _make_q("sa2", "SA 2", q_type=QuestionType.SHORT_ANSWER, marks=2),
        ]

        with pytest.raises(CandidatePoolShortageError) as exc_info:
            select_version_questions(candidates, sample_blueprint, num_versions=2)

        err = exc_info.value
        assert err.section_id == "A"
        assert err.needed == 4
        assert err.available == 3
