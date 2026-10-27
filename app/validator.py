# app/validator.py
# Comprehensive question quality, clarity, coherence, and paper integrity validation.
# Prevents:
# - Vague wording, missing figures, missing numerical values, dangling questions
# - Incoherent merging of disjoint domain scenarios in long-answer questions
# - Exact and conceptual duplicates across sections
# - Section count, question type, and total marks inconsistencies

from __future__ import annotations

import re
from typing import Optional

from app.duplicates import is_duplicate
from app.schemas import MathCheckStatus, Paper, Question, QuestionType


# Patterns indicating references to non-existent visual diagrams/figures/tables
VISUAL_REF_PATTERN = re.compile(
    r"\b(?:figure|diagram|picture|image|table)\s+(?:below|above|given|shown|adjacent|here)\b|"
    r"\b(?:as\s+shown\s+in\s+(?:the\s+)?(?:figure|diagram|picture|table))\b|"
    r"\b(?:look\s+at\s+the\s+(?:figure|diagram|picture|table))\b|"
    r"\b(?:from\s+the\s+(?:given\s+)?(?:figure|diagram|picture|table))\b",
    re.IGNORECASE,
)

# Mutually disjoint domain concepts that must NEVER be mashed together into a single question
DISJOINT_DOMAINS = [
    (
        "Banking & Savings",
        {"bank", "deposit", "deposited", "account", "interest", "withdrew", "withdrawal", "passbook"},
    ),
    (
        "Factory & Manufacturing",
        {"factory", "manufactured", "manufacturing", "produced", "soap", "packets", "cartons", "warehouse"},
    ),
    (
        "Village Demographics",
        {"village", "population", "inhabitants", "census"},
    ),
    (
        "Geometry & Angles",
        {"angle", "angles", "protractor", "degrees", "acute", "obtuse", "vertex", "ray"},
    ),
    (
        "Circles & Geometry",
        {"circle", "radius", "diameter", "chord", "circumference", "semicircle"},
    ),
    (
        "Railway & Train Travel",
        {"train", "locomotive", "railway", "platform", "express"},
    ),
    (
        "Water & Tumbler Capacity",
        {"vessel", "tumbler", "liters", "milliliters", "capacity"},
    ),
]


def check_question_clarity(
    text: str,
    qtype: QuestionType,
    options: Optional[list[str]] = None,
) -> tuple[bool, Optional[str]]:
    """
    Check if a question is complete, self-contained, and understandable.
    Rejects questions with missing visual artifacts, missing operands, or dangling prompts.
    """
    clean = text.strip()
    if len(clean) < 10:
        return False, "Question text is too short or incomplete (< 10 characters)."

    # 1. Reject references to non-existent visual diagrams
    m = VISUAL_REF_PATTERN.search(clean)
    if m:
        return (
            False,
            f"Question references missing visual artifact ('{m.group(0)}'). Question must be fully self-contained.",
        )

    # 2. Reject dangling prompts
    if clean.endswith((":", "following", "following:")) and (not options or len(options) == 0):
        return False, "Question text ends with incomplete prompt ('following:') without options or data."

    # 3. Reject comparison questions without operands or options
    if (
        re.search(r"^(?:which\s+is\s+(?:greater|smaller|larger|heavier|longer|more|less))\s*\??$", clean, re.IGNORECASE)
        and not options
    ):
        return False, "Incomplete comparison question without values or options."

    # 4. Reject computation prompts without numbers
    if re.search(r"\b(?:find\s+the\s+sum|add|subtract|multiply|divide|product\s+of)\b", clean, re.IGNORECASE):
        nums = re.findall(r"\b\d+\b", clean)
        # Check if numbers are defined in question text or options
        if len(nums) < 2 and (not options or len(options) < 2):
            return False, "Computation question lacks required numerical values."

    return True, None


def check_question_coherence(text: str) -> tuple[bool, Optional[str]]:
    """
    Check if a question maintains a single, unified scenario.
    Rejects incoherent mashups of two unrelated domain storylines.
    """
    tokens = set(re.findall(r"[a-z]+", text.lower()))
    matched_domains = []
    for dname, dwords in DISJOINT_DOMAINS:
        if dwords.intersection(tokens):
            matched_domains.append(dname)

    # If question mentions words from 2 or more distinct disjoint domains, reject
    if len(matched_domains) > 1:
        domains_str = ", ".join(matched_domains)
        return False, f"Incoherently combined unrelated domain scenarios: {domains_str}"

    return True, None


def check_grade_age_appropriateness(q: Question) -> tuple[bool, Optional[str]]:
    """
    Validate that question language, concepts, and format match the student's age/grade.
    Specifically protects Class 1 (age 6-7) from:
    - English comprehension / vocabulary / definition questions
    - Adult commercial contexts (banking, factories, payroll, demographics)
    - Excessive wordiness or abstract essays in short/long answer
    """
    g_str = str(q.grade or "")
    m = re.search(r"\b([1-5])\b", g_str)
    grade_num = int(m.group(1)) if m else None

    # If grade is not specified on question, try parsing from chapter
    if grade_num is None and q.chapter:
        m2 = re.search(r"Class\s*([1-5])", q.chapter, re.IGNORECASE)
        if m2:
            grade_num = int(m2.group(1))

    text_lower = q.question_text.lower()

    if grade_num == 1:
        # 1. Prohibit definition / essay / English grammar questions
        definitional_patterns = [
            (r"\bwhat\s+(?:does|is)\b.*\b(?:mean|meaning|represent|stand for)\b", "definition of mathematical symbol/term"),
            (r"\b(?:what\s+does\b.*\b(?:symbol|sign|term|word))\b", "definition of mathematical symbol/term"),
            (r"\b(?:define|give\s+the\s+definition\s+of)\b", "definition prompt"),
            (r"\b(?:state\s+(?:the\s+)?vocabulary|terms?\s+used\s+in)\b", "vocabulary recall"),
            (r"\b(?:explain\s+(?:in\s+words|why|what\s+term|how))\b", "written explanation"),
            (r"\b(?:in\s+your\s+own\s+words)\b", "essay prompt"),
            (r"\b(?:write\s+a\s+sentence\s+explaining)\b", "written sentence explanation"),
            (r"\b(?:describe\s+(?:the\s+meaning|the\s+rule))\b", "rule description"),
            (r"\b(?:english\s+word|vocabulary\s+term)\b", "English vocabulary test"),
        ]
        for pat, desc in definitional_patterns:
            if re.search(pat, text_lower):
                return False, f"Class 1 questions must test direct math skills, not {desc} (matched '{pat}')."

        # 2. Prohibit adult commercial / manufacturing / demographic contexts
        adult_patterns = [
            (r"\b(?:bank|deposit|deposited|withdrawal|account\s+balance|interest)\b", "banking/financial"),
            (r"\b(?:factory|manufacture|manufactured|warehouse|production)\b", "factory manufacturing"),
            (r"\b(?:salary|employee|census|population\s+of\s+(?:a\s+)?village)\b", "adult demographics/employment"),
            (r"\b(?:wholesale|retailer|merchant\s+purchased)\b", "commercial merchant trading"),
        ]
        for pat, desc in adult_patterns:
            if re.search(pat, text_lower):
                return False, f"Class 1 questions cannot use adult {desc} contexts (matched '{pat}')."

        # 3. Word count check for Class 1 (excluding sub-part sums)
        words = text_lower.split()
        if q.question_type == QuestionType.MULTIPLE_CHOICE and len(words) > 35:
            return False, f"Class 1 MCQ is too wordy for 6-year-old readers ({len(words)} words; max 35)."
        if q.question_type == QuestionType.SHORT_ANSWER and len(words) > 40:
            return False, f"Class 1 Short Answer is too wordy ({len(words)} words; max 40)."

        # 4. Long Answer must have sub-parts or clear arithmetic sums, not a giant word essay
        if q.question_type == QuestionType.LONG_ANSWER:
            has_subparts = bool(re.search(r"\([a-d]\)|\(i\)|\b[a-d]\)|\b1\)\s*.*2\)", q.question_text))
            has_math_ops = bool(re.search(r"[+\-=__]{2,}|[+−×÷=]", q.question_text))
            if not has_subparts and not has_math_ops and len(words) > 45:
                return False, "Class 1 Long Answer must consist of structured simple sub-problems or arithmetic sums, not long word essays."

    elif grade_num == 2:
        # Avoid heavy adult commercial contexts for Class 2
        for pat in [r"\b(?:bank|deposit|deposited|withdrawal|interest|factory|manufactured)\b"]:
            if re.search(pat, text_lower):
                return False, "Class 2 questions should avoid adult banking/factory contexts."

    return True, None


def validate_question_quality(q: Question) -> tuple[bool, Optional[str]]:
    """
    Perform full quality validation on an individual question.
    Checks: clarity, coherence, MCQ completeness, SymPy mathematical check, and age-appropriateness.
    """
    # 1. Clarity check
    ok, err = check_question_clarity(q.question_text, q.question_type, q.options)
    if not ok:
        return False, err

    # 2. Coherence check
    ok, err = check_question_coherence(q.question_text)
    if not ok:
        return False, err

    # 3. MCQ completeness check
    if q.question_type == QuestionType.MULTIPLE_CHOICE:
        if not q.options or len(q.options) < 4:
            return False, f"Multiple choice question must have at least 4 options (found {len(q.options) if q.options else 0})."
        if not q.correct_option or not str(q.correct_option).strip():
            return False, "Multiple choice question must specify a correct_option."
        norm_opts = [str(o).strip().lower() for o in q.options]
        norm_corr = str(q.correct_option).strip().lower()
        if norm_corr not in norm_opts:
            return False, f"correct_option '{q.correct_option}' does not match any of the provided options: {q.options}."
        if len(set(norm_opts)) < len(norm_opts):
            return False, "Multiple choice question contains duplicate options."

    # 4. Mathematical verification check (SymPy)
    if q.math_verification and q.math_verification.status == MathCheckStatus.DISCREPANCY:
        return (
            False,
            f"Mathematical discrepancy detected: {q.math_verification.details or 'Calculation check failed'}",
        )

    # 5. Grade / age appropriateness check
    ok, err = check_grade_age_appropriateness(q)
    if not ok:
        return False, err

    return True, None


def validate_paper_questions_uniqueness(questions: list[Question]) -> tuple[bool, Optional[str]]:
    """
    Perform pairwise uniqueness check across ALL questions from all sections.
    Catches exact duplicates, near-duplicates, and repeated scenario templates.
    """
    for i in range(len(questions)):
        qi = questions[i]
        for j in range(i + 1, len(questions)):
            qj = questions[j]
            is_dup, reason = is_duplicate(qj, [qi])
            if is_dup:
                return (
                    False,
                    f"Questions {i+1} and {j+1} are duplicate or repeated problems: {reason} "
                    f"(Q{i+1}: '{qi.question_text[:60]}...' vs Q{j+1}: '{qj.question_text[:60]}...')",
                )
    return True, None


def validate_full_paper(paper: Paper) -> tuple[bool, list[str]]:
    """
    Perform full audit of an assembled paper before saving or presenting to user.
    Returns (is_valid, list_of_errors).
    """
    errors: list[str] = []

    # 1. Question count and marks sum
    if len(paper.questions) != paper.num_questions:
        errors.append(
            f"Question count mismatch: expected {paper.num_questions}, got {len(paper.questions)}."
        )

    actual_marks = sum(q.marks for q in paper.questions)
    if actual_marks != paper.total_marks:
        errors.append(
            f"Total marks mismatch: expected {paper.total_marks}, got {actual_marks}."
        )

    # 2. Individual question quality
    for idx, q in enumerate(paper.questions, 1):
        ok, err = validate_question_quality(q)
        if not ok:
            errors.append(f"Question {idx} quality error: {err}")

    # 3. Cross-section uniqueness
    ok, err = validate_paper_questions_uniqueness(paper.questions)
    if not ok and err:
        errors.append(f"Cross-section duplication error: {err}")

    # 4. Section structure consistency
    if paper.sections:
        sec_q_count = sum(len(s.questions) for s in paper.sections)
        if sec_q_count != len(paper.questions):
            errors.append(
                f"Sections question count sum ({sec_q_count}) does not match paper questions ({len(paper.questions)})."
            )
        sec_m_count = sum(s.section_marks for s in paper.sections)
        if sec_m_count != paper.total_marks:
            errors.append(
                f"Sections marks sum ({sec_m_count}) does not match paper total marks ({paper.total_marks})."
            )

    return len(errors) == 0, errors
