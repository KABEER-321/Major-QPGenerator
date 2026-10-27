# app/math_verify.py
# Optional, limited mathematical expression verification using SymPy.
# As required by MASTER_PROMPT Section 3:
# "SymPy only for optional, limited checking of supported mathematical expressions;
#  do not claim that it can validate every question or proof."

import logging
import re
from typing import Optional

import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor,
)

from app.schemas import Question, MathVerificationResult, MathCheckStatus

logger = logging.getLogger(__name__)

TRANSFORMATIONS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
)


def _clean_math_str(s: str) -> str:
    """Normalize unicode superscripts and symbols into parseable expressions."""
    s = s.replace("²", "^2").replace("³", "^3").replace("⁴", "^4")
    s = s.replace("×", "*").replace("÷", "/").replace("−", "-")
    s = s.replace("±", "+-")
    return s


def _safe_parse_expr(expr_str: str) -> Optional[sp.Expr]:
    """Safely parse a mathematical expression string using SymPy."""
    try:
        clean = _clean_math_str(expr_str.strip())
        # Basic sanity check: only allow mathematical characters
        if not re.match(r"^[\w\s\+\-\*\/\^\(\)\.\,\=]+$", clean):
            return None
        return parse_expr(clean, transformations=TRANSFORMATIONS)
    except Exception:
        return None


def _check_equation_roots(question_text: str, answer_text: str) -> Optional[MathVerificationResult]:
    """Check if proposed roots/solutions satisfy an equation in the question.

    Example:
      Question: "Find the roots of 2x^2 + x - 300 = 0"
      Answer: "x = 12 and x = -25/2"
    """
    clean_q = _clean_math_str(question_text)
    if "=" not in clean_q:
        return None

    # Split around '=' and isolate math tokens from natural language words
    parts = clean_q.split("=", 1)
    words = parts[0].strip().split()
    math_words = []
    for w in reversed(words):
        alpha_chars = re.sub(r"[^a-zA-Z]", "", w)
        if len(alpha_chars) > 1 and alpha_chars.lower() not in ("sin", "cos", "tan", "sqrt", "log", "ln"):
            break
        math_words.insert(0, w)

    if not math_words:
        return None
    lhs_str = " ".join(math_words).strip()

    rhs_match = re.match(r"^([-\d\/\.a-zA-Z\^\(\)]+)", parts[1].strip())
    if not rhs_match:
        return None
    rhs_str = rhs_match.group(1).rstrip(".,;")

    lhs = _safe_parse_expr(lhs_str)
    rhs = _safe_parse_expr(rhs_str)
    if lhs is None or rhs is None:
        return None

    equation = sp.Eq(lhs, rhs)
    symbols = list(equation.free_symbols)
    if len(symbols) != 1:
        return None
    var = symbols[0]

    # Search for roots in answer: e.g. "x = 12", "x = -25/2", "roots are 12 and -25/2"
    clean_ans = _clean_math_str(answer_text)
    # If the question or answer explicitly specifies discriminant, do not treat D as an equation root
    if "discriminant" in clean_q.lower() or "discriminant" in clean_ans.lower():
        return None

    root_patterns = re.findall(rf"{var}\s*=\s*([-\d\/\.]+)", clean_ans)
    if not root_patterns:
        # Try generic root numbers only if "root" or "solution" is indicated
        if any(w in clean_ans.lower() for w in ("root", "solution", "value")):
            root_patterns = re.findall(r"(?:^|\s)(?:roots?|is|are)?\s*([-\d]+(?:/[-\d]+)?|\d+\.\d+)", clean_ans)

    if not root_patterns:
        return None

    verified_roots = []
    discrepancy_roots = []

    for r_str in root_patterns:
        r_val = _safe_parse_expr(r_str)
        if r_val is None:
            continue
        try:
            # Substitute root into (lhs - rhs)
            diff = sp.simplify((lhs - rhs).subs(var, r_val))
            if diff == 0:
                verified_roots.append(f"{var} = {r_str}")
            else:
                discrepancy_roots.append(f"{var} = {r_str} (residual: {diff})")
        except Exception:
            continue

    if discrepancy_roots:
        return MathVerificationResult(
            status=MathCheckStatus.DISCREPANCY,
            details=f"Equation {lhs_str} = {rhs_str} failed for: {', '.join(discrepancy_roots)}",
            expression_checked=f"{lhs_str} = {rhs_str}",
        )

    if verified_roots:
        return MathVerificationResult(
            status=MathCheckStatus.VERIFIED,
            details=f"SymPy verified root(s) {', '.join(verified_roots)} satisfy {lhs_str} = {rhs_str}",
            expression_checked=f"{lhs_str} = {rhs_str}",
        )

    return None


def _check_discriminant(question_text: str, answer_text: str) -> Optional[MathVerificationResult]:
    """Check if discriminant calculation D = b^2 - 4ac matches stated value."""
    clean_ans = _clean_math_str(answer_text)
    # Look for D = <number> or Discriminant = <number>
    d_match = re.search(r"(?:D|discriminant)\s*=\s*([-\d]+)", clean_ans, re.IGNORECASE)
    if not d_match:
        return None

    stated_d = int(d_match.group(1))

    # Look for quadratic coefficients in question: ax^2 + bx + c = 0
    clean_q = _clean_math_str(question_text)
    eq_match = re.search(r"([-\d]*)\s*x\^2\s*([\+\-]\s*[\d]*)\s*x\s*([\+\-]\s*\d+)\s*=\s*0", clean_q)
    if eq_match:
        a_str, b_str, c_str = eq_match.groups()
        a_str = a_str.replace(" ", "")
        b_str = b_str.replace(" ", "")
        c_str = c_str.replace(" ", "")

        a = int(a_str) if a_str and a_str not in ("+", "-") else (-1 if a_str == "-" else 1)
        b = int(b_str) if b_str and b_str not in ("+", "-") else (-1 if b_str == "-" else 1)
        c = int(c_str)

        calc_d = b**2 - 4 * a * c
        if calc_d == stated_d:
            return MathVerificationResult(
                status=MathCheckStatus.VERIFIED,
                details=f"SymPy verified discriminant D = b² - 4ac = ({b})² - 4({a})({c}) = {calc_d}",
                expression_checked=f"b^2 - 4ac = ({b})^2 - 4({a})({c})",
            )
        else:
            return MathVerificationResult(
                status=MathCheckStatus.DISCREPANCY,
                details=f"Calculated discriminant D = {calc_d} does not match stated D = {stated_d}",
                expression_checked=f"b^2 - 4ac = {calc_d} != {stated_d}",
            )

    return None


def _check_arithmetic_evaluation(answer_text: str, solution_text: str) -> Optional[MathVerificationResult]:
    """Check explicit arithmetic equations in answer or solution (e.g. 9 + 10 = 19 or 1^2 - 4(2)(-300) = 2401)."""
    # Evaluate expressions in solution_text first, then answer_text (without cross-gluing numbers)
    for text in (solution_text, answer_text):
        if not text or "=" not in text:
            continue
        clean = _clean_math_str(text)
        # Search for equality pattern: <expr> = <number>
        eqs = re.findall(r"([\d\+\-\*\/\^\(\)\s]{3,})\s*=\s*([-\d]+(?:\.\d+)?)", clean)
        for lhs_raw, rhs_raw in eqs:
            lhs_raw = lhs_raw.strip()
            rhs_raw = rhs_raw.strip()
            # If adjacent numbers without operators exist (e.g. "19 9 + 10"), isolate the rightmost expression
            if re.search(r"\b\d+\s+\d+\b", lhs_raw):
                parts = re.split(r"\b\d+\s+", lhs_raw)
                lhs_raw = parts[-1].strip()

            if not any(op in lhs_raw for op in ("+", "-", "*", "/", "^")):
                continue
            try:
                lhs_val = _safe_parse_expr(lhs_raw)
                rhs_val = _safe_parse_expr(rhs_raw)
                if lhs_val is not None and rhs_val is not None:
                    if sp.simplify(lhs_val - rhs_val) == 0:
                        return MathVerificationResult(
                            status=MathCheckStatus.VERIFIED,
                            details=f"SymPy verified equality: {lhs_raw} = {rhs_raw}",
                            expression_checked=f"{lhs_raw} = {rhs_raw}",
                        )
                    else:
                        return MathVerificationResult(
                            status=MathCheckStatus.DISCREPANCY,
                            details=f"Arithmetic discrepancy: {lhs_raw} evaluates to {lhs_val}, not {rhs_raw}",
                            expression_checked=f"{lhs_raw} != {rhs_raw}",
                        )
            except Exception:
                continue
    return None


def verify_question_math(question: Question) -> MathVerificationResult:
    """Run limited, optional mathematical validation using SymPy.

    Checks:
    1. Equation roots satisfaction (if equation and roots are present)
    2. Quadratic discriminant evaluation
    3. Basic arithmetic expression equality

    If the question is conceptual, geometric, or textual with no extractable
    supported formula, it returns UNSUPPORTED. Never raises exceptions.
    """
    try:
        # 1. Prioritize discriminant check if discriminant is explicitly mentioned
        if "discriminant" in question.question_text.lower() or "discriminant" in question.answer.lower():
            disc_res = _check_discriminant(question.question_text, question.answer)
            if disc_res:
                return disc_res

        # 2. Check equation roots
        root_res = _check_equation_roots(question.question_text, question.answer)
        if root_res:
            return root_res

        # 3. Fallback discriminant check
        disc_res = _check_discriminant(question.question_text, question.answer)
        if disc_res:
            return disc_res

        # 4. Check arithmetic evaluation in answer/solution
        arith_res = _check_arithmetic_evaluation(question.answer, question.solution)
        if arith_res:
            return arith_res

        return MathVerificationResult(
            status=MathCheckStatus.UNSUPPORTED,
            details="Conceptual or textual question without extractable algebraic formula.",
            expression_checked=None,
        )
    except Exception as e:
        logger.debug("SymPy math verification exception: %s", e)
        return MathVerificationResult(
            status=MathCheckStatus.UNSUPPORTED,
            details=f"Verification bypassed due to parse limitation: {e}",
            expression_checked=None,
        )
