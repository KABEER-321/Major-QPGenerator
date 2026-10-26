# tests/test_math_verify.py
# Tests for optional SymPy mathematical expression checking.

import pytest
from app.schemas import Question, QuestionType, Difficulty, MathCheckStatus
from app.math_verify import (
    verify_question_math,
    _check_equation_roots,
    _check_discriminant,
    _check_arithmetic_evaluation,
)


def _make_question(text: str, answer: str, solution: str = "") -> Question:
    return Question(
        question_text=text,
        question_type=QuestionType.SHORT_ANSWER,
        marks=4,
        difficulty=Difficulty.MEDIUM,
        topic="Quadratic Equations",
        answer=answer,
        solution=solution,
    )


class TestEquationRoots:
    def test_verifies_valid_quadratic_roots(self):
        q = _make_question(
            text="Solve the quadratic equation 2x^2 + x - 300 = 0.",
            answer="x = 12 and x = -25/2",
        )
        res = verify_question_math(q)
        assert res.status == MathCheckStatus.VERIFIED
        assert "SymPy verified" in res.details
        assert res.expression_checked is not None

    def test_detects_incorrect_root_discrepancy(self):
        q = _make_question(
            text="Find the root of 2x^2 + x - 300 = 0.",
            answer="x = 10",
        )
        res = verify_question_math(q)
        assert res.status == MathCheckStatus.DISCREPANCY
        assert "failed" in res.details

    def test_verifies_simple_linear_equation(self):
        q = _make_question(
            text="Solve 3x - 15 = 0.",
            answer="x = 5",
        )
        res = verify_question_math(q)
        assert res.status == MathCheckStatus.VERIFIED


class TestDiscriminantCheck:
    def test_verifies_correct_discriminant(self):
        q = _make_question(
            text="Find the discriminant for 2x^2 + 1x - 300 = 0.",
            answer="Discriminant D = 2401",
        )
        res = verify_question_math(q)
        assert res.status == MathCheckStatus.VERIFIED
        assert "2401" in res.details

    def test_detects_wrong_discriminant_discrepancy(self):
        q = _make_question(
            text="Find the discriminant for 2x^2 + 1x - 300 = 0.",
            answer="Discriminant D = 500",
        )
        res = verify_question_math(q)
        assert res.status == MathCheckStatus.DISCREPANCY
        assert "does not match" in res.details


class TestArithmeticEvaluation:
    def test_verifies_arithmetic_equality(self):
        q = _make_question(
            text="Calculate the value of the discriminant expression.",
            answer="2401",
            solution="D = 1^2 - 4*2*(-300) = 2401",
        )
        res = verify_question_math(q)
        assert res.status == MathCheckStatus.VERIFIED

    def test_detects_arithmetic_discrepancy(self):
        q = _make_question(
            text="Add numbers",
            answer="5",
            solution="2 + 2 = 5",
        )
        res = verify_question_math(q)
        assert res.status == MathCheckStatus.DISCREPANCY


class TestUnsupportedAndSafety:
    def test_conceptual_question_returns_unsupported(self):
        q = _make_question(
            text="State the Fundamental Theorem of Arithmetic.",
            answer="Every composite number can be expressed as a product of primes uniquely.",
        )
        res = verify_question_math(q)
        assert res.status == MathCheckStatus.UNSUPPORTED
        assert "Conceptual or textual" in res.details

    def test_malformed_math_string_does_not_crash(self):
        q = _make_question(
            text="What is @#$% ^&* ? ===",
            answer="Invalid syntax",
        )
        res = verify_question_math(q)
        assert res.status in (MathCheckStatus.UNSUPPORTED, MathCheckStatus.SKIPPED)
