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
from app.schemas import (
    ExamType,
    MathCheckStatus,
    Paper,
    Question,
    QuestionType,
    parse_grade_number,
)


# Patterns indicating references to non-existent visual diagrams/figures/tables
VISUAL_REF_PATTERN = re.compile(
    r"\b(?:figure|diagram|picture|image|table)\s+(?:below|above|given|shown|adjacent|here)\b|"
    r"\b(?:as\s+shown\s+in\s+(?:the\s+)?(?:figure|diagram|picture|table))\b|"
    r"\b(?:look\s+at\s+the\s+(?:figure|diagram|picture|table))\b|"
    r"\b(?:from\s+the\s+(?:given\s+)?(?:figure|diagram|picture|table))\b",
    re.IGNORECASE,
)

# Patterns indicating textbook meta-references that should never appear in student test papers
TEXTBOOK_META_PATTERN = re.compile(
    r"\b(?:as\s+discussed\s+in|according\s+to\s+lesson|as\s+introduced\s+in|discussed\s+in\s+lesson|"
    r"in\s+lesson\s+\d+|in\s+chapter\s+\d+|from\s+lesson\s+\d+|from\s+chapter\s+\d+|"
    r"as\s+studied\s+in\s+class|as\s+given\s+in\s+lesson|according\s+to\s+chapter)\b",
    re.IGNORECASE,
)

# Patterns indicating AI drafting notes or generated commentary
AI_DRAFTING_PATTERN = re.compile(
    r"\b(?:note:|drafting\s+note|alternative\s+instruction|option\s+1:|option\s+2:|here\s+is\s+a\s+question|please\s+note)\b",
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

    # 2. Reject textbook meta-references
    m_meta = TEXTBOOK_META_PATTERN.search(clean)
    if m_meta:
        return (
            False,
            f"Question contains textbook meta-reference ('{m_meta.group(0)}'). Examination questions must be phrased directly without referencing lesson numbers or textbook chapters.",
        )

    # 3. Reject AI drafting commentary
    m_ai = AI_DRAFTING_PATTERN.search(clean)
    if m_ai:
        return (
            False,
            f"Question contains AI drafting notes or generated commentary ('{m_ai.group(0)}').",
        )

    # 4. Reject dangling prompts
    if clean.endswith((":", "following", "following:")) and (not options or len(options) == 0):
        return False, "Question text ends with incomplete prompt ('following:') without options or data."

    # 5. Reject comparison questions without operands or options
    if (
        re.search(r"^(?:which\s+is\s+(?:greater|smaller|larger|heavier|longer|more|less))\s*\??$", clean, re.IGNORECASE)
        and not options
    ):
        return False, "Incomplete comparison question without values or options."

    # 6. Reject computation prompts without numbers
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
    Enforces that questions across all grades test mathematics rather than English vocabulary
    or reading comprehension, and sets age-appropriate bounds for Classes 1 to 5.
    """
    g_str = str(getattr(q, "curriculum_grade", None) or q.grade or "")
    m = re.search(r"\b([1-5])\b", g_str)
    grade_num = int(m.group(1)) if m else None

    # If grade is not specified on question, try parsing from chapter
    if grade_num is None and q.chapter:
        m2 = re.search(r"Class\s*([1-5])", q.chapter, re.IGNORECASE)
        if m2:
            grade_num = int(m2.group(1))

    text_lower = q.question_text.lower()
    words = text_lower.split()

    # 1. Universal Prohibition: NEVER test English vocabulary, definition writing, or essay explanations
    definitional_patterns = [
        (r"\bwhat\s+(?:does|is)\b.*\b(?:mean|meaning|represent|stand for)\b", "definition of mathematical symbol/term"),
        (r"\b(?:what\s+does\b.*\b(?:symbol|sign|term|word))\b", "definition of mathematical symbol/term"),
        (r"\b(?:define|give\s+the\s+definition\s+of)\b", "definition prompt"),
        (r"\b(?:state\s+(?:the\s+)?vocabulary|terms?\s+used\s+in)\b", "vocabulary recall"),
        (r"\bstate\s+(?:the\s+)?(?:spatial\s+relationship\s+)?term\s+used\b", "vocabulary recall"),
        (r"\b(?:explain\s+(?:in\s+words|why|what\s+term|how))\b", "written explanation"),
        (r"\b(?:in\s+your\s+own\s+words)\b", "essay prompt"),
        (r"\b(?:write\s+a\s+sentence\s+explaining)\b", "written sentence explanation"),
        (r"\b(?:describe\s+(?:the\s+meaning|the\s+rule))\b", "rule description"),
        (r"\b(?:english\s+word|vocabulary\s+term)\b", "English vocabulary test"),
        (r"\bwhich\s+spatial\s+vocabulary\s+term\b", "spatial vocabulary recall"),
        (r"\bwhat\s+spatial\s+vocabulary\s+term\b", "spatial vocabulary recall"),
        (r"\bstate\s+the\s+spatial\s+relationship\s+term\b", "spatial vocabulary recall"),
        (r"\bspatial\s+understanding\s+vocabulary\b", "spatial vocabulary recall"),
    ]
    for pat, desc in definitional_patterns:
        if re.search(pat, text_lower):
            if grade_num == 1:
                return False, f"Class 1 questions must test direct math skills, not {desc} (matched '{pat}')."
            return False, f"Mathematics questions must test mathematical skills, not {desc} (matched '{pat}')."

    # 2. Grade-Specific Boundaries
    if grade_num == 1:
        # Prohibit adult commercial contexts
        adult_patterns = [
            (r"\b(?:bank|deposit|deposited|withdrawal|account\s+balance|interest)\b", "banking/financial"),
            (r"\b(?:factory|manufacture|manufactured|warehouse|production)\b", "factory manufacturing"),
            (r"\b(?:salary|employee|census|population\s+of\s+(?:a\s+)?village)\b", "adult demographics/employment"),
            (r"\b(?:wholesale|retailer|merchant\s+purchased)\b", "commercial merchant trading"),
        ]
        for pat, desc in adult_patterns:
            if re.search(pat, text_lower):
                return False, f"Class 1 questions cannot use adult {desc} contexts (matched '{pat}')."

        # Word count limits for 6-year-old readers
        if q.question_type == QuestionType.MULTIPLE_CHOICE and len(words) > 28:
            return False, f"Class 1 MCQ is too wordy for 6-year-old readers ({len(words)} words; max 28)."
        if q.question_type == QuestionType.SHORT_ANSWER and len(words) > 35:
            return False, f"Class 1 Short Answer is too wordy ({len(words)} words; max 35)."

        # Long Answer must have sub-parts or clear arithmetic sums, not a giant word essay
        if q.question_type == QuestionType.LONG_ANSWER:
            has_subparts = bool(re.search(r"\([a-d]\)|\(i\)|\b[a-d]\)|\b1\)\s*.*2\)", q.question_text))
            has_math_ops = bool(re.search(r"[+\-=__]{2,}|[+−×÷=]", q.question_text))
            if not has_subparts and not has_math_ops and len(words) > 40:
                return False, "Class 1 Long Answer must consist of structured simple sub-problems or arithmetic sums, not long word essays."

        # Numbers in direct Class 1 arithmetic operations should not exceed 20
        ops_match = re.findall(r"\b(\d+)\s*[+\-−]\s*(\d+)\b", q.question_text)
        for n1, n2 in ops_match:
            if int(n1) > 20 or int(n2) > 20:
                return False, f"Class 1 arithmetic operations are limited to numbers up to 20 (found {n1}, {n2})."

    elif grade_num == 2:
        for pat in [r"\b(?:bank|deposit|deposited|withdrawal|interest|factory|manufactured|salary|warehouse)\b"]:
            if re.search(pat, text_lower):
                return False, "Class 2 questions should avoid adult banking/factory contexts."

        if q.question_type == QuestionType.MULTIPLE_CHOICE and len(words) > 35:
            return False, f"Class 2 MCQ is too wordy ({len(words)} words; max 35)."
        if q.question_type == QuestionType.SHORT_ANSWER and len(words) > 40:
            return False, f"Class 2 Short Answer is too wordy ({len(words)} words; max 40)."
        if q.question_type == QuestionType.LONG_ANSWER:
            has_subparts = bool(re.search(r"\([a-d]\)|\(i\)|\b[a-d]\)|\b1\)\s*.*2\)", q.question_text))
            has_math_ops = bool(re.search(r"[+\-=__]{2,}|[+−×÷=]", q.question_text))
            if not has_subparts and not has_math_ops and len(words) > 50:
                return False, "Class 2 Long Answer must have structured subparts or clear calculations."

    elif grade_num == 3:
        if q.question_type == QuestionType.MULTIPLE_CHOICE and len(words) > 40:
            return False, f"Class 3 MCQ is too wordy ({len(words)} words; max 40)."
        if q.question_type == QuestionType.SHORT_ANSWER and len(words) > 50:
            return False, f"Class 3 Short Answer is too wordy ({len(words)} words; max 50)."

    elif grade_num == 4:
        if q.question_type == QuestionType.MULTIPLE_CHOICE and len(words) > 50:
            return False, f"Class 4 MCQ is too wordy ({len(words)} words; max 50)."
        if q.question_type == QuestionType.SHORT_ANSWER and len(words) > 60:
            return False, f"Class 4 Short Answer is too wordy ({len(words)} words; max 60)."

    elif grade_num == 5:
        if q.question_type == QuestionType.MULTIPLE_CHOICE and len(words) > 55:
            return False, f"Class 5 MCQ is too wordy ({len(words)} words; max 55)."
        if q.question_type == QuestionType.SHORT_ANSWER and len(words) > 65:
            return False, f"Class 5 Short Answer is too wordy ({len(words)} words; max 65)."

    # Maximum prompt length across all classes
    if len(words) > 85:
        return False, f"Question is excessively verbose ({len(words)} words; max 85). Avoid unnecessary story details."

    return True, None


def validate_question_quality(q: Question) -> tuple[bool, Optional[str]]:
    """
    Perform full quality validation on an individual question.
    Checks: clarity, coherence, MCQ completeness, SymPy mathematical check, mark allocation, and age-appropriateness.
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

    # 4. Long Answer mark & structure check: high mark questions (>= 3 marks) must have structured work
    if q.marks >= 3 and q.question_type == QuestionType.LONG_ANSWER:
        # Check that high mark questions are not trivial 1-step arithmetic facts
        nums = re.findall(r"\b\d+\b", q.question_text)
        has_subparts = bool(re.search(r"\([a-d]\)|\(i\)|\b[a-d]\)|\b1\)\s*.*2\)", q.question_text))
        if len(nums) <= 2 and not has_subparts and len(q.question_text.split()) < 15:
            return False, f"Long answer question with {q.marks} marks lacks sufficient problem depth or subparts."

    # 5. Mathematical verification check (SymPy)
    if q.math_verification and q.math_verification.status == MathCheckStatus.DISCREPANCY:
        return (
            False,
            f"Mathematical discrepancy detected: {q.math_verification.details or 'Calculation check failed'}",
        )

    # 6. Grade / age appropriateness check
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


def validate_paper_chapter_marks_distribution(
    paper: Paper,
    blueprint: Optional[ExamBlueprint] = None,
    permitted_chapters: Optional[list[str]] = None,
) -> tuple[bool, Optional[str], dict[str, int], dict[str, int]]:
    """
    Validate the syllabus-wise marks and question distribution across eligible chapters.
    Catches:
    1. Chapter neglect: an eligible chapter receives 0 questions or 0 marks when N >= M.
    2. Single-chapter domination: one chapter monopolizes >= 70% of total marks when M >= 3
       while other eligible chapters receive <= 2 marks (such as the 1/1/23 marks distribution failure).
    3. Mathematical inconsistency: section marks, question marks, chapter totals, and overall total.
    Returns:
        (is_valid, error_message, actual_chapter_marks, actual_chapter_questions)
    """
    actual_marks: dict[str, int] = {}
    actual_qs: dict[str, int] = {}
    permitted = permitted_chapters or paper.permitted_units or []

    for q in paper.questions:
        ch = q.chapter or q.topic or "Unknown"
        if permitted:
            for pu in permitted:
                if pu.lower() in ch.lower() or ch.lower() in pu.lower():
                    ch = pu
                    break
        actual_marks[ch] = actual_marks.get(ch, 0) + q.marks
        actual_qs[ch] = actual_qs.get(ch, 0) + 1

    # 1. Mathematical consistency check
    total_q_marks = sum(q.marks for q in paper.questions)
    if total_q_marks != paper.total_marks:
        return False, f"Sum of question marks ({total_q_marks}) does not equal paper total marks ({paper.total_marks}).", actual_marks, actual_qs

    total_ch_marks = sum(actual_marks.values())
    if total_ch_marks != paper.total_marks:
        return False, f"Sum of chapter marks ({total_ch_marks}) does not equal paper total marks ({paper.total_marks}).", actual_marks, actual_qs

    if paper.sections:
        for sec in paper.sections:
            sec_sum = sum(q.marks for q in sec.questions)
            if sec_sum != sec.section_marks:
                return False, f"Section '{sec.section_id}' questions mark sum ({sec_sum}) does not equal section marks ({sec.section_marks}).", actual_marks, actual_qs

    # Determine chapters to check
    chapters_to_check = permitted
    if not chapters_to_check and blueprint and blueprint.allocated_chapters:
        chapters_to_check = blueprint.allocated_chapters
    if not chapters_to_check and paper.covered_chapters:
        chapters_to_check = paper.covered_chapters

    m = len(chapters_to_check)
    n = paper.num_questions

    # 2. Neglect Check: when N >= M, every eligible chapter must have at least 1 question & non-zero marks
    if n >= m and m > 1:
        zero_chapters = [c for c in chapters_to_check if actual_marks.get(c, 0) == 0 or actual_qs.get(c, 0) == 0]
        if zero_chapters:
            return (
                False,
                f"Syllabus marks distribution failure: eligible chapter(s) completely neglected with 0 marks: {', '.join(zero_chapters)}.",
                actual_marks,
                actual_qs,
            )

    # 3. Domination Check:
    # Only applies when N >= M and M >= 3 and total_marks >= 10
    if n >= m and m >= 3 and paper.total_marks >= 10:
        for ch, ch_m in actual_marks.items():
            share = ch_m / paper.total_marks
            if share >= 0.70:
                neglected = [c for c in chapters_to_check if c != ch and actual_marks.get(c, 0) <= 2]
                if neglected:
                    return (
                        False,
                        f"Syllabus marks distribution failure: Chapter '{ch}' dominates with {ch_m}/{paper.total_marks} marks ({share * 100:.1f}%), "
                        f"while other eligible chapters ({', '.join(neglected)}) receive <= 2 marks. "
                        f"Expected balanced syllabus distribution across eligible chapters.",
                        actual_marks,
                        actual_qs,
                    )

    # 4. Blueprint Target Deviation Check (if blueprint targets are available)
    if blueprint and getattr(blueprint, "chapter_marks_allocation", None):
        target_alloc = blueprint.chapter_marks_allocation
        for ch, target_m in target_alloc.items():
            act_m = actual_marks.get(ch, 0)
            diff = abs(act_m - target_m)
            max_allowed_diff = max(5, int(0.30 * paper.total_marks))
            if diff > max_allowed_diff:
                return (
                    False,
                    f"Syllabus marks distribution mismatch: Chapter '{ch}' has {act_m} marks (target: {target_m} marks, diff: {diff}M > allowed {max_allowed_diff}M).",
                    actual_marks,
                    actual_qs,
                )

    return True, None, actual_marks, actual_qs


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

    # 5. Examination Scope & Syllabus Integrity
    from app.generator import is_question_in_forbidden_units

    # Verify allowed chapter association
    if paper.permitted_units:
        for idx, q in enumerate(paper.questions, 1):
            if not q.chapter or q.chapter in ("Mathematics Curriculum", "General Math", "Mathematics"):
                errors.append(f"Question {idx} lacks specific chapter association.")
            else:
                matches_permitted = any(
                    pu.lower() in q.chapter.lower() or q.chapter.lower() in pu.lower()
                    for pu in paper.permitted_units
                )
                if not matches_permitted:
                    errors.append(
                        f"Question {idx} chapter '{q.chapter}' does not belong to permitted chapters: {paper.permitted_units}"
                    )

    # Verify no questions use forbidden units
    if paper.forbidden_units:
        for idx, q in enumerate(paper.questions, 1):
            is_forb, forb_reason = is_question_in_forbidden_units(q, paper.forbidden_units, paper.permitted_units)
            if is_forb:
                errors.append(f"Question {idx} uses concepts from forbidden chapter: {forb_reason}")

    # Verify examination-type structural rules
    if paper.exam_type == ExamType.PREREQUISITE:
        acad_num = parse_grade_number(paper.academic_grade or paper.grade)
        curr_num = parse_grade_number(paper.curriculum_grade or "")
        if acad_num is not None and curr_num is not None:
            if curr_num != acad_num - 1:
                errors.append(
                    f"Prerequisite examination curriculum grade ({curr_num}) must be exactly 1 grade below academic grade ({acad_num})."
                )
    elif paper.exam_type == ExamType.MIDTERM:
        if paper.forbidden_units is not None and len(paper.forbidden_units) == 0 and paper.permitted_units and len(paper.permitted_units) > 1:
            errors.append("Mid-Semester examination must have excluded second-half chapters.")
    elif paper.exam_type == ExamType.FINAL:
        if paper.forbidden_units:
            errors.append("Final examination permits content from the full textbook and cannot forbid chapters.")

    # 6. Syllabus-Wise Chapter Marks & Question Distribution
    dist_ok, dist_err, _, _ = validate_paper_chapter_marks_distribution(
        paper=paper,
        permitted_chapters=paper.permitted_units,
    )
    if not dist_ok and dist_err:
        errors.append(dist_err)

    return len(errors) == 0, errors
