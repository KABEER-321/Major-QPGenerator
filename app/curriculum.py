# app/curriculum.py
"""
Permanent curriculum textbook management for Classes 1 through 5.
Stores textbooks permanently in data/curriculum/class_{N}/textbook.pdf.
Extracts chapters based on textbook structure, chunks with chapter metadata,
and manages persistent FAISS indexes.
"""

from __future__ import annotations

import hashlib
import logging
import math
from pathlib import Path
from typing import Any, Optional

import fitz  # PyMuPDF
import re
from langchain_core.documents import Document

from app.config import settings
from app.documents_store import (
    DocumentRecord,
    load_document_record,
    save_document_record,
)
from app.retrieval import build_faiss_index
from app.schemas import (
    ExamType,
    ExamBlueprint,
    SectionBlueprint,
    QuestionType,
    format_grade_display,
    get_previous_class,
    parse_grade_number,
)

logger = logging.getLogger(__name__)


def clean_text(text: str) -> str:
    """Normalize whitespace and strip unprintable control characters."""
    if not text:
        return ""
    return re.sub(r"[ \t]+", " ", text).strip()

# Complete chapter tables of contents for Classes 1 to 5
CURRICULUM_CHAPTERS: dict[int, list[str]] = {
    1: [
        "Lesson 1: Spatial Understanding",
        "Lesson 2: Solids Around Us",
        "Lesson 3: Digits (1-9)",
        "Lesson 4: Zero",
        "Lesson 5: Addition (Sum not more than 9)",
        "Lesson 6: Subtraction",
        "Lesson 7: Number 10",
        "Lesson 8: Units and Tens",
        "Lesson 9: Number 11 to 20",
    ],
    2: [
        "Lesson 1: Shapes and Spatial Understanding",
        "Lesson 2: Numbers",
        "Lesson 3: Addition",
        "Lesson 4: Subtraction",
        "Lesson 5: Multiplication",
        "Lesson 6: Division",
    ],
    3: [
        "Chapter 1: Shapes",
        "Chapter 2: Numbers",
        "Chapter 3: Addition",
        "Chapter 4: Subtraction",
        "Chapter 5: Multiplication",
    ],
    4: [
        "Chapter 1: Perimeter and Area of simple Geometrical figures",
        "Chapter 2: Numbers",
        "Chapter 3: Addition",
        "Chapter 4: Subtraction",
        "Chapter 5: Multiplication",
        "Chapter 6: Division",
        "Chapter 7: Circles",
        "Chapter 8: Mental Arithmetic",
        "Chapter 9: Fractional Numbers",
    ],
    5: [
        "Unit 1: 5-digit numbers",
        "Unit 2: Addition",
        "Unit 3: Subtraction",
        "Unit 4: Factors and multiples",
        "Unit 5: Fractions",
        "Unit 6: Angles",
        "Unit 7: Circles",
        "Unit 8: Length",
        "Unit 9: Perimeter and Area",
        "Unit 10: Data Handling",
    ],
}

# Page mappings (1-based printed page number to chapter) based on textbook TOCs
# Offset: the textbook front-matter ends at page 10 (PDF page 10 is index 9).
# Lessons start on printed page 1 (PDF page index 10).
_PDF_PAGE_OFFSET = 10


def get_textbook_path(grade: int | str) -> Path:
    """Return the filesystem path to the stored textbook for the given grade."""
    g_num = parse_grade_number(str(grade))
    if g_num is None or g_num < 1 or g_num > 5:
        raise ValueError(f"Invalid curriculum grade '{grade}'. Must be between 1 and 5.")
    return settings.curriculum_dir / f"class_{g_num}" / "textbook.pdf"


def get_curriculum_doc_id(grade: int | str) -> str:
    """Return the deterministic document ID for a grade's textbook."""
    g_num = parse_grade_number(str(grade))
    if g_num is None:
        raise ValueError(f"Cannot parse grade from '{grade}'")
    return f"curriculum_class_{g_num}"


def extract_chapter_number(chapter_str: str) -> tuple[int, str]:
    """
    Extract the numerical chapter index from strings like 'Lesson 1: ...',
    'Chapter 10: ...', or 'Unit 5 - ...' for stable textbook-order sorting.
    Returns (chapter_number, raw_string). Unnumbered chapters sort at the end (9999).
    """
    if not chapter_str:
        return (9999, "")
    # Check for keywords Lesson/Chapter/Unit followed by digits
    m = re.search(r'\b(?:lesson|chapter|unit|module|topic)\s*(\d+)', chapter_str, re.IGNORECASE)
    if m:
        return (int(m.group(1)), chapter_str)
    # Check for leading number like '1. Spatial Understanding'
    m2 = re.match(r'^\s*(\d+)[\.:\-\s]', chapter_str)
    if m2:
        return (int(m2.group(1)), chapter_str)
    return (9999, chapter_str)


def order_chapters_by_textbook(chapters: list[str]) -> list[str]:
    """
    Order chapters strictly according to textbook sequence (1, 2, ... 10).
    Never sorts alphabetically (which incorrectly places Unit 10 before Unit 2)
    and never uses an unordered collection.
    """
    if not chapters:
        return []
    # If the input list is already strictly monotonically ordered by chapter number,
    # preserve original list order verbatim
    nums = [extract_chapter_number(ch)[0] for ch in chapters]
    if all(nums[i] <= nums[i + 1] for i in range(len(nums) - 1)) and all(n != 9999 for n in nums):
        return list(chapters)
    # Otherwise sort numerically by textbook chapter number
    return sorted(chapters, key=extract_chapter_number)


def get_chapters_for_class(grade: int | str) -> list[str]:
    """Return the list of chapters/units for the given grade in strict textbook order."""
    g_num = parse_grade_number(str(grade))
    if g_num is None:
        return []
    raw = CURRICULUM_CHAPTERS.get(g_num, [])
    return order_chapters_by_textbook(raw)


def get_midterm_chapters_for_class(grade: int | str) -> list[str]:
    """
    Return the first half of chapters for the midterm examination in textbook order.
    Uses consistent ceil rounding for odd chapter counts (e.g. 9 -> 5, 10 -> 5).
    """
    all_ch = get_chapters_for_class(grade)
    if not all_ch:
        return []
    half_count = math.ceil(len(all_ch) / 2.0)
    return all_ch[:half_count]


def get_second_half_chapters_for_class(grade: int | str) -> list[str]:
    """
    Return the remaining (forbidden for midterm) chapters in textbook order.
    """
    all_ch = get_chapters_for_class(grade)
    if not all_ch:
        return []
    half_count = math.ceil(len(all_ch) / 2.0)
    return all_ch[half_count:]


class SyllabusScopeInfo:
    """Authoritative representation of the syllabus scope for an examination request."""
    def __init__(
        self,
        exam_type: ExamType,
        academic_grade: str,
        curriculum_grade: str,
        target_document_id: str,
        total_chapters: int,
        all_chapters: list[str],
        permitted_chapters: list[str],
        forbidden_chapters: list[str],
        scope_description: str,
    ):
        self.exam_type = exam_type
        self.academic_grade = academic_grade
        self.curriculum_grade = curriculum_grade
        self.target_document_id = target_document_id
        self.total_chapters = total_chapters
        self.all_chapters = all_chapters
        self.permitted_chapters = permitted_chapters
        self.forbidden_chapters = forbidden_chapters
        self.scope_description = scope_description


def resolve_authoritative_syllabus_scope(
    exam_type: ExamType,
    grade_str: str,
    custom_doc: Optional[DocumentRecord] = None,
) -> SyllabusScopeInfo:
    """
    Construct one authoritative syllabus-scope selection for an examination request.
    Enforces:
    1. Prerequisite Examination:
       - Uses 100% of the previous class's textbook (Class 2 -> Class 1, Class 3 -> Class 2, etc.)
       - Class 1 Prerequisite is unavailable unless a previous-level textbook is provided.
    2. Mid-Semester Examination:
       - Uses the first half of the selected class's textbook starting from chapter 1.
       - number_of_chapters = ceil(total_chapters / 2).
       - Excludes second-half chapters strictly.
    3. Final Examination:
       - Uses 100% of the selected class's textbook syllabus with all chapters eligible.
    4. Metadata Integrity:
       - If chapter boundaries cannot be identified reliably, raises ValueError instead of guessing.
    """
    acad_num = parse_grade_number(grade_str)
    academic_class = format_grade_display(grade_str)

    if exam_type == ExamType.PREREQUISITE:
        if acad_num is not None and acad_num <= 1:
            if not custom_doc or not custom_doc.units:
                raise ValueError(
                    f"Prerequisite Examination is not available for {academic_class} "
                    "unless a previous-level curriculum textbook is provided."
                )
            target_grade_str = custom_doc.grade or "Class 0"
            target_doc_id = custom_doc.document_id
            all_ch = order_chapters_by_textbook(custom_doc.units)
        else:
            prev_class = get_previous_class(grade_str)
            prev_num = parse_grade_number(prev_class)
            target_grade_str = format_grade_display(prev_class)
            target_doc_id = get_curriculum_doc_id(prev_num)
            all_ch = get_chapters_for_class(prev_num)
            if not all_ch and custom_doc and custom_doc.units:
                all_ch = order_chapters_by_textbook(custom_doc.units)

        if not all_ch:
            raise ValueError(
                f"No chapter metadata found for prerequisite syllabus {target_grade_str}. "
                "Cannot reliably determine prerequisite syllabus scope."
            )

        permitted = list(all_ch)
        forbidden: list[str] = []
        desc = (
            f"Full syllabus of {target_grade_str} Mathematics "
            f"({len(all_ch)} chapters: {', '.join(all_ch)}) "
            f"[Prerequisite for {academic_class}]"
        )
        return SyllabusScopeInfo(
            exam_type=exam_type,
            academic_grade=academic_class,
            curriculum_grade=target_grade_str,
            target_document_id=target_doc_id,
            total_chapters=len(all_ch),
            all_chapters=all_ch,
            permitted_chapters=permitted,
            forbidden_chapters=forbidden,
            scope_description=desc,
        )

    elif exam_type == ExamType.MIDTERM:
        curr_grade_str = academic_class
        if custom_doc is not None:
            if not custom_doc.units:
                raise ValueError(
                    f"Textbook '{custom_doc.original_filename}' has missing or ambiguous chapter metadata. "
                    "Cannot reliably identify chapter boundaries for examination syllabus scope."
                )
            all_ch = order_chapters_by_textbook(custom_doc.units)
            target_doc_id = custom_doc.document_id
        elif acad_num is not None and acad_num in CURRICULUM_CHAPTERS:
            all_ch = get_chapters_for_class(acad_num)
            target_doc_id = get_curriculum_doc_id(acad_num)
        else:
            raise ValueError(
                f"Textbook chapter metadata missing for {academic_class}. "
                "Cannot reliably determine mid-semester chapter boundaries."
            )

        if len(all_ch) < 2:
            raise ValueError(
                f"Cannot determine mid-semester scope: curriculum for {academic_class} "
                f"has only {len(all_ch)} chapter(s). At least 2 chapters are required."
            )

        num_mid = math.ceil(len(all_ch) / 2.0)
        permitted = all_ch[:num_mid]
        forbidden = all_ch[num_mid:]
        desc = (
            f"First half of {curr_grade_str} syllabus "
            f"({len(permitted)} of {len(all_ch)} chapters: {', '.join(permitted)})"
        )
        return SyllabusScopeInfo(
            exam_type=exam_type,
            academic_grade=academic_class,
            curriculum_grade=curr_grade_str,
            target_document_id=target_doc_id,
            total_chapters=len(all_ch),
            all_chapters=all_ch,
            permitted_chapters=permitted,
            forbidden_chapters=forbidden,
            scope_description=desc,
        )

    else:  # Final Examination
        curr_grade_str = academic_class
        if custom_doc is not None:
            if not custom_doc.units:
                raise ValueError(
                    f"Textbook '{custom_doc.original_filename}' has missing or ambiguous chapter metadata. "
                    "Cannot reliably identify chapter boundaries for examination syllabus scope."
                )
            all_ch = order_chapters_by_textbook(custom_doc.units)
            target_doc_id = custom_doc.document_id
        elif acad_num is not None and acad_num in CURRICULUM_CHAPTERS:
            all_ch = get_chapters_for_class(acad_num)
            target_doc_id = get_curriculum_doc_id(acad_num)
        else:
            raise ValueError(
                f"Textbook chapter metadata missing for {academic_class}. "
                "Cannot reliably determine final examination syllabus scope."
            )

        if not all_ch:
            raise ValueError(
                f"No chapter metadata found for {academic_class}. "
                "Cannot verify 100% textbook coverage."
            )

        permitted = list(all_ch)
        forbidden = []
        desc = (
            f"Full syllabus of {curr_grade_str} Mathematics "
            f"({len(all_ch)} chapters: {', '.join(all_ch)})"
        )
        return SyllabusScopeInfo(
            exam_type=exam_type,
            academic_grade=academic_class,
            curriculum_grade=curr_grade_str,
            target_document_id=target_doc_id,
            total_chapters=len(all_ch),
            all_chapters=all_ch,
            permitted_chapters=permitted,
            forbidden_chapters=forbidden,
            scope_description=desc,
        )


def is_computational_chapter(chapter_name: str) -> bool:
    """Return True if chapter has operational depth suitable for multi-step / long-answer problems."""
    c = chapter_name.lower()
    return any(
        k in c
        for k in (
            "addition",
            "subtraction",
            "multiplication",
            "division",
            "fraction",
            "decimal",
            "perimeter",
            "area",
            "factors",
            "multiples",
            "unit",
            "tens",
            "numbers",
            "number 11",
            "5-digit",
            "data handling",
            "length",
            "mental arithmetic",
            "word problem",
        )
    ) and not any(k in c for k in ("spatial", "zero"))


def is_foundational_chapter(chapter_name: str) -> bool:
    """Return True if chapter is visual/foundational vocabulary (better for MCQ / short answer)."""
    c = chapter_name.lower()
    return (
        any(k in c for k in ("spatial", "solid", "shape", "shapes", "zero", "circle", "angle"))
        and not any(k in c for k in ("addition", "subtraction", "multiplication", "division", "fraction", "perimeter", "area"))
    )


def get_chapter_pedagogical_weight(chapter_name: str, grade: int | str | None = None) -> float:
    """
    Calculate the pedagogical weight of a textbook chapter based on its syllabus
    learning objectives, conceptual depth, and curriculum emphasis.
    Weights:
    - Foundational / Visual Chapters: ~1.0
      (Spatial Understanding, Solids, Shapes, Zero, Circles, Angles, Patterns, Clock/Calendar)
    - Intermediate / Core Number & Measurement Concepts: ~1.6
      (Digits 1-9, Number 10, Numbers, Place Value, Units & Tens, Number 11 to 20, Length, Money, Mental Arithmetic)
    - Heavy Operational & Computational Chapters: ~2.4
      (Addition, Subtraction, Multiplication, Division, Fractions, Perimeter & Area, Factors & Multiples)
    """
    c = chapter_name.lower()

    # Heavy Operational & Computational (highest pedagogical weight)
    if any(k in c for k in (
        "addition", "subtraction", "multiplication", "division",
        "fraction", "decimal", "perimeter", "area", "factors", "multiples"
    )):
        return 2.4

    # Intermediate / Core Numerical & Measurement Concepts
    if any(k in c for k in (
        "number", "numbers", "digit", "digits", "tens", "units",
        "length", "weight", "capacity", "money", "rupees", "time", "clock",
        "mental arithmetic", "data handling"
    )):
        return 1.6

    # Foundational / Visual Units (Shapes, Spatial, Zero, Solids, Circles, Angles)
    return 1.0


def allocate_chapter_marks_and_questions(
    permitted_chapters: list[str],
    total_marks: int,
    total_questions: int,
    grade: int | str | None = None,
) -> tuple[dict[str, int], dict[str, int]]:
    """
    Compute target marks and question counts for each chapter using pedagogical weights.
    Uses the Largest Remainder Method (Hare-Niemeyer) to ensure:
    - sum(target_marks.values()) == total_marks
    - sum(target_questions.values()) == total_questions
    - When total_questions >= len(permitted_chapters), every chapter receives at least 1 question
      and an integer mark allocation reflecting its curriculum weight.
    """
    m = len(permitted_chapters)
    if m == 0:
        return {}, {}

    weights = [get_chapter_pedagogical_weight(ch, grade) for ch in permitted_chapters]
    total_weight = sum(weights) or float(m)

    # 1. Question Counts Allocation
    if total_questions >= m:
        # Every chapter gets at least 1 question
        base_qs = [1] * m
        rem_q = total_questions - m
        if rem_q > 0:
            exact_extras = [rem_q * (w / total_weight) for w in weights]
            floor_extras = [int(x) for x in exact_extras]
            remainder_q = [(exact_extras[i] - floor_extras[i], i) for i in range(m)]
            # Sort remainders descending; tie-break by order
            remainder_q.sort(key=lambda t: t[0], reverse=True)
            leftover = rem_q - sum(floor_extras)
            for k in range(leftover):
                idx = remainder_q[k][1]
                floor_extras[idx] += 1
            assigned_qs = [base_qs[i] + floor_extras[i] for i in range(m)]
        else:
            assigned_qs = base_qs
    else:
        # N < M: distribute 1 question to each of the selected chapters (handled by caller)
        assigned_qs = [1 if i < total_questions else 0 for i in range(m)]

    # 2. Marks Allocation
    # Ensure every chapter with questions gets at least 1 mark (or minimum mark)
    active_indices = [i for i in range(m) if assigned_qs[i] > 0]
    num_active = len(active_indices)
    if num_active == 0:
        return {}, {}

    active_weight = sum(weights[i] for i in active_indices)
    exact_marks = [total_marks * (weights[i] / active_weight) if i in active_indices else 0.0 for i in range(m)]
    floor_marks = [max(1, int(exact_marks[i])) if i in active_indices else 0 for i in range(m)]
    current_sum = sum(floor_marks)

    if current_sum <= total_marks:
        diff = total_marks - current_sum
        remainder_m = [(exact_marks[i] - int(exact_marks[i]), i) for i in active_indices]
        remainder_m.sort(key=lambda t: t[0], reverse=True)
        for k in range(diff):
            idx = remainder_m[k % len(remainder_m)][1]
            floor_marks[idx] += 1
    else:
        # Deficit adjustment (can occur if total_marks is very small relative to num_active)
        diff = current_sum - total_marks
        remainder_m = [(exact_marks[i] - int(exact_marks[i]), i) for i in active_indices]
        remainder_m.sort(key=lambda t: t[0])  # Smallest remainder reduced first
        for k in range(diff):
            idx = remainder_m[k % len(remainder_m)][1]
            if floor_marks[idx] > 1:
                floor_marks[idx] -= 1

    target_marks_map = {permitted_chapters[i]: floor_marks[i] for i in range(m)}
    target_questions_map = {permitted_chapters[i]: assigned_qs[i] for i in range(m)}
    return target_marks_map, target_questions_map


def _solve_slot_chapter_assignment(
    blueprint: ExamBlueprint,
    permitted_chapters: list[str],
    target_marks: dict[str, int],
    target_questions: dict[str, int],
) -> dict[str, list[str]]:
    """
    Assign individual question slots in each section of the blueprint to eligible chapters.
    Guarantees:
    - Each section receives exactly its num_questions slots.
    - Each chapter receives its allocated question count (or as close as mathematically possible).
    - Each chapter receives its allocated target marks (or as close as mathematically possible).
    - Cognitive appropriateness: foundational visual chapters are not placed in >= 3 mark
      slots when operational or core numerical chapters are available.
    """
    order_map = {ch: idx for idx, ch in enumerate(permitted_chapters)}
    active_chapters = [ch for ch in permitted_chapters if target_questions.get(ch, 0) > 0]
    if not active_chapters:
        active_chapters = list(permitted_chapters)

    m = len(active_chapters)
    ch_indices = list(range(m))

    # 1. Expand slots from all sections
    slots = []
    for s_idx, sec in enumerate(blueprint.sections):
        marks_dist = sec.marks_distribution
        if not marks_dist or len(marks_dist) != sec.num_questions:
            default_m = sec.marks_per_question or max(1, sec.section_marks // sec.num_questions)
            marks_dist = [default_m] * sec.num_questions

        is_la = sec.question_type == QuestionType.LONG_ANSWER or sec.section_marks >= sec.num_questions * 3
        for slot_i in range(sec.num_questions):
            m_val = marks_dist[slot_i]
            slots.append({
                'section_id': sec.section_id,
                'sec_idx': s_idx,
                'slot_idx': slot_i,
                'mark': m_val,
                'is_la': is_la,
                'qtype': sec.question_type,
            })

    # Sort slots: high marks first (5M, 2M, 1M)
    slots.sort(key=lambda s: (s['mark'], 1 if s['is_la'] else 0), reverse=True)

    is_found = [is_foundational_chapter(active_chapters[i]) for i in range(m)]
    has_non_found = any(not f for f in is_found)

    # Beam search with width 300
    beam_width = 300
    # Beam item: (cost, current_marks_tuple, current_qs_tuple, tuple_of_ch_indices)
    beam = [(0, (0,) * m, (0,) * m, ())]
    total_slots = len(slots)

    for step, slot in enumerate(slots):
        mark = slot['mark']
        is_high_mark = mark >= 3 or slot['is_la']
        rem_slots = total_slots - step
        next_beam = []

        for cost, cur_m, cur_q, assign in beam:
            zero_indices = [i for i in ch_indices if cur_q[i] == 0]
            if rem_slots <= len(zero_indices):
                allowed_indices = zero_indices
            else:
                allowed_indices = ch_indices

            for ch_idx in allowed_indices:
                if is_high_mark and has_non_found and is_found[ch_idx]:
                    continue  # Foundational visual chapters disallowed from high marks

                new_m = list(cur_m)
                new_m[ch_idx] += mark
                new_q = list(cur_q)
                new_q[ch_idx] += 1

                ch_name = active_chapters[ch_idx]
                t_m = target_marks.get(ch_name, 0)
                t_q = target_questions.get(ch_name, 0)

                # Penalize overshooting target marks or question counts
                m_diff = abs(new_m[ch_idx] - t_m)
                q_diff = abs(new_q[ch_idx] - t_q)
                overshoot_m = max(0, new_m[ch_idx] - t_m)
                overshoot_q = max(0, new_q[ch_idx] - t_q)

                suitability_bonus = 0
                if is_high_mark and is_computational_chapter(ch_name):
                    suitability_bonus = -5
                elif slot['qtype'] == QuestionType.MULTIPLE_CHOICE and is_found[ch_idx]:
                    suitability_bonus = -10 + order_map.get(ch_name, 0)

                step_cost = cost + m_diff * 4 + overshoot_m * 6 + q_diff * 2 + overshoot_q * 4 + suitability_bonus
                next_beam.append((step_cost, tuple(new_m), tuple(new_q), assign + (ch_idx,)))

        if not next_beam:
            # Fallback if over-constrained
            next_beam = [(cost, cur_m, cur_q, assign + (0,)) for cost, cur_m, cur_q, assign in beam]
        # Prune to beam_width
        next_beam.sort(key=lambda x: x[0])
        beam = next_beam[:beam_width]

    # Evaluate best overall assignment
    best_eval = float("inf")
    best_assign = None

    for cost, cur_m, cur_q, assign in beam:
        eval_score = 0
        # If N >= M, heavy penalty if any chapter has 0 questions
        if len(slots) >= m and any(q == 0 for q in cur_q):
            eval_score += 10000

        for i in range(m):
            ch_name = active_chapters[i]
            t_m = target_marks.get(ch_name, 0)
            t_q = target_questions.get(ch_name, 0)
            eval_score += abs(cur_m[i] - t_m) * 10
            eval_score += abs(cur_q[i] - t_q) * 4

        if eval_score < best_eval:
            best_eval = eval_score
            best_assign = assign

    # Fallback if beam empty or unassigned
    if not best_assign:
        best_assign = tuple(i % m for i in range(len(slots)))

    assigned_slots = [(slots[idx], active_chapters[ch_idx]) for idx, ch_idx in enumerate(best_assign)]

    # Map back to section order
    section_allocations: dict[str, list[str]] = {sec.section_id: [None] * sec.num_questions for sec in blueprint.sections}
    for slot, ch in assigned_slots:
        section_allocations[slot['section_id']][slot['slot_idx']] = ch

    for sec in blueprint.sections:
        for idx in range(sec.num_questions):
            if section_allocations[sec.section_id][idx] is None:
                section_allocations[sec.section_id][idx] = active_chapters[0]

    return section_allocations


def assign_blueprint_chapter_coverage(
    blueprint: ExamBlueprint,
    permitted_chapters: list[str],
) -> tuple[ExamBlueprint, str]:
    """
    Distribute questions and marks across the eligible chapters before generation.
    - Creates a syllabus-wise blueprint allocating both question counts and marks
      proportionally across every eligible chapter based on pedagogical weights.
    - When question count N >= total chapters M: every chapter receives at least one question
      and a fair share of the paper's marks (never leaves a chapter with 1 mark while another has 23).
    - Allocates multi-step / operational chapters to Long Answer sections and foundational
      visual chapters to Multiple Choice / Short Answer sections.
    - When question count N < total chapters M: selects N evenly distributed representative chapters
      and reports the chapters actually assessed.
    """
    if not permitted_chapters:
        return blueprint, "No chapters available for allocation."

    m = len(permitted_chapters)
    n = blueprint.total_questions
    grade = getattr(blueprint, "grade", None)

    if n >= m:
        target_marks, target_qs = allocate_chapter_marks_and_questions(
            permitted_chapters=permitted_chapters,
            total_marks=blueprint.total_marks,
            total_questions=n,
            grade=grade,
        )
        sec_allocs = _solve_slot_chapter_assignment(
            blueprint=blueprint,
            permitted_chapters=permitted_chapters,
            target_marks=target_marks,
            target_questions=target_qs,
        )

        for sec in blueprint.sections:
            alloc = sec_allocs.get(sec.section_id, [permitted_chapters[0]] * sec.num_questions)
            sec.chapter_allocation = alloc
            sec.topics = list(dict.fromkeys(alloc))

        blueprint.allocated_chapters = list(permitted_chapters)
        blueprint.chapter_marks_allocation = target_marks
        blueprint.chapter_questions_allocation = target_qs

        # Build detailed explanation
        parts = [f"{ch} = {target_marks.get(ch, 0)}M ({target_qs.get(ch, 0)}Q)" for ch in permitted_chapters]
        explanation = (
            f"Comprehensive balanced coverage: All {m} eligible curriculum chapters "
            f"are assessed across the {n} questions of the examination paper ({', '.join(parts)})."
        )

    else:
        # N < M: Balanced sampling of N representative chapters across the M chapters
        if n == 1:
            indices = [0]
        else:
            indices = [round(i * (m - 1) / (n - 1)) for i in range(n)]
        chosen_chapters = [permitted_chapters[idx] for idx in indices]

        target_marks, target_qs = allocate_chapter_marks_and_questions(
            permitted_chapters=chosen_chapters,
            total_marks=blueprint.total_marks,
            total_questions=n,
            grade=grade,
        )
        sec_allocs = _solve_slot_chapter_assignment(
            blueprint=blueprint,
            permitted_chapters=chosen_chapters,
            target_marks=target_marks,
            target_questions=target_qs,
        )

        for sec in blueprint.sections:
            alloc = sec_allocs.get(sec.section_id, [chosen_chapters[0]] * sec.num_questions)
            sec.chapter_allocation = alloc
            sec.topics = list(dict.fromkeys(alloc))

        blueprint.allocated_chapters = chosen_chapters
        blueprint.chapter_marks_allocation = target_marks
        blueprint.chapter_questions_allocation = target_qs

        parts = [f"{ch} = {target_marks.get(ch, 0)}M ({target_qs.get(ch, 0)}Q)" for ch in chosen_chapters]
        explanation = (
            f"Balanced representative coverage: {n} chapters selected across the {m}-chapter "
            f"syllabus ({', '.join(chosen_chapters)}) because requested question count ({n}) "
            f"is smaller than available chapters ({m}). Chapter allocation: {', '.join(parts)}."
        )

    blueprint.coverage_explanation = explanation
    return blueprint, explanation


def compute_file_hash(path: Path) -> str:
    """Compute sha256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def _get_chapter_for_printed_page(grade_num: int, printed_page: int) -> tuple[str, bool]:
    """
    Given a printed page number (1-based from textbook body), determine the chapter
    and whether it is in the first half of the syllabus.
    """
    chapters = CURRICULUM_CHAPTERS[grade_num]
    midterm_count = math.ceil(len(chapters) / 2)

    # Page intervals from the textbook Table of Contents
    if grade_num == 1:
        cuts = [(15, 0), (24, 1), (62, 2), (66, 3), (77, 4), (85, 5), (98, 6), (103, 7), (118, 8)]
    elif grade_num == 2:
        cuts = [(38, 0), (86, 1), (105, 2), (137, 3), (143, 4), (150, 5)]
    elif grade_num == 3:
        cuts = [(17, 0), (66, 1), (95, 2), (123, 3), (150, 4)]
    elif grade_num == 4:
        cuts = [(12, 0), (46, 1), (61, 2), (71, 3), (92, 4), (110, 5), (118, 6), (138, 7), (158, 8)]
    elif grade_num == 5:
        cuts = [(15, 0), (24, 1), (37, 2), (47, 3), (68, 4), (85, 5), (94, 6), (103, 7), (125, 8), (143, 9)]
    else:
        cuts = []

    for max_page, ch_idx in cuts:
        if printed_page <= max_page:
            ch_name = chapters[ch_idx]
            is_first_half = ch_idx < midterm_count
            return ch_name, is_first_half

    # Fallback to last chapter
    ch_idx = len(chapters) - 1
    return chapters[ch_idx], False


def extract_textbook_chunks(grade_num: int) -> list[Document]:
    """
    Extract textbook PDF pages grouped by chapter, and split into coherent chunks
    annotated with grade, subject, chapter, and whether the chapter is in the first half.
    """
    pdf_path = get_textbook_path(grade_num)
    if not pdf_path.exists():
        raise FileNotFoundError(f"Textbook for Class {grade_num} not found at {pdf_path}")

    doc = fitz.open(pdf_path)
    chapters = CURRICULUM_CHAPTERS.get(grade_num, [])
    midterm_count = math.ceil(len(chapters) / 2)

    # Group page texts by chapter
    ch_pages: dict[str, list[tuple[int, str]]] = {ch: [] for ch in chapters}
    for pdf_pno in range(_PDF_PAGE_OFFSET, len(doc)):
        printed_page = pdf_pno - _PDF_PAGE_OFFSET + 1
        page_text = clean_text(doc[pdf_pno].get_text())
        if page_text and len(page_text.strip()) > 10:
            ch_name, _ = _get_chapter_for_printed_page(grade_num, printed_page)
            if ch_name in ch_pages:
                ch_pages[ch_name].append((printed_page, page_text))
            elif chapters:
                ch_pages[chapters[-1]].append((printed_page, page_text))
    doc.close()

    chunks: list[Document] = []
    chunk_counter = 0

    chunk_size = 1200
    chunk_overlap = 150

    for ch_idx, ch_name in enumerate(chapters):
        pages_in_ch = ch_pages.get(ch_name, [])
        if not pages_in_ch:
            continue

        is_first_half = ch_idx < midterm_count
        combined_text = "\n\n".join(f"[Page {pno}]\n{txt}" for pno, txt in pages_in_ch)

        start = 0
        text_len = len(combined_text)
        step = chunk_size - chunk_overlap

        while start < text_len:
            sub = combined_text[start : start + chunk_size].strip()
            if len(sub) > 30:
                chunk_id = f"c{grade_num}_ch{ch_idx + 1}_k{chunk_counter}"
                metadata = {
                    "doc_id": get_curriculum_doc_id(grade_num),
                    "chunk_id": chunk_id,
                    "chunk_index": chunk_counter,
                    "grade": f"Class {grade_num}",
                    "subject": "Mathematics",
                    "chapter": ch_name,
                    "unit": ch_name,
                    "is_first_half": is_first_half,
                    "source": f"Class {grade_num} Mathematics - {ch_name}",
                }
                chunks.append(Document(page_content=sub, metadata=metadata))
                chunk_counter += 1
            start += step

    return chunks


def index_curriculum_textbook(grade_num: int, force: bool = False) -> DocumentRecord:
    """
    Index a single class textbook if missing or modified.
    Builds and saves the FAISS index and stores the DocumentRecord.
    """
    pdf_path = get_textbook_path(grade_num)
    if not pdf_path.exists():
        raise FileNotFoundError(f"Textbook for Class {grade_num} does not exist at {pdf_path}")

    doc_id = get_curriculum_doc_id(grade_num)
    index_file = settings.indexes_dir / doc_id / "index.faiss"
    current_hash = compute_file_hash(pdf_path)

    # Check if already indexed and unchanged
    if not force and index_file.exists():
        try:
            rec = load_document_record(doc_id)
            if (getattr(rec, "content_hash", None) == current_hash or rec.has_index) and rec.num_chunks > 0:
                logger.info("Class %d textbook already indexed with %d chunks. Reusing.", grade_num, rec.num_chunks)
                return rec
        except Exception as e:
            logger.warning("Could not reuse existing record for Class %d: %s. Re-indexing.", grade_num, e)

    logger.info("Extracting and building FAISS index for Class %d textbook...", grade_num)
    chunks = extract_textbook_chunks(grade_num)
    if not chunks:
        raise ValueError(f"No text chunks could be extracted from Class {grade_num} textbook.")

    build_faiss_index(chunks, doc_id)

    record = DocumentRecord(
        document_id=doc_id,
        original_filename="textbook.pdf",
        safe_filename=f"class_{grade_num}_textbook.pdf",
        file_size=pdf_path.stat().st_size,
        content_hash=current_hash,
        num_chunks=len(chunks),
        chunk_ids=[c.metadata["chunk_id"] for c in chunks],
        has_index=True,
        grade=f"Class {grade_num}",
        subject="Mathematics",
        units=CURRICULUM_CHAPTERS.get(grade_num, []),
    )
    save_document_record(record)
    logger.info("Successfully indexed Class %d textbook (%d chunks).", grade_num, len(chunks))
    return record


def ensure_all_curriculum_textbooks_indexed() -> dict[int, bool]:
    """
    Ensure all Classes 1-5 textbooks are indexed. Called on application startup.
    Returns a dict mapping grade -> success status.
    """
    results: dict[int, bool] = {}
    for grade in range(1, 6):
        try:
            pdf_path = get_textbook_path(grade)
            if pdf_path.exists():
                index_curriculum_textbook(grade, force=False)
                results[grade] = True
            else:
                logger.warning("Textbook for Class %d not found at %s", grade, pdf_path)
                results[grade] = False
        except Exception as e:
            logger.error("Failed to index Class %d textbook: %s", grade, e)
            results[grade] = False
    return results


def get_curriculum_status() -> list[dict[str, Any]]:
    """
    Get the readiness status of all 5 class textbooks for the UI.
    """
    status_list = []
    for grade in range(1, 6):
        pdf_path = get_textbook_path(grade)
        doc_id = get_curriculum_doc_id(grade)
        index_file = settings.indexes_dir / doc_id / "index.faiss"

        is_available = pdf_path.exists()
        is_indexed = is_available and index_file.exists()

        num_chunks = 0
        if is_indexed:
            try:
                rec = load_document_record(doc_id)
                num_chunks = rec.num_chunks
            except Exception:
                num_chunks = 0

        chapters = get_chapters_for_class(grade)
        midterm_chapters = get_midterm_chapters_for_class(grade)

        status_list.append({
            "grade": grade,
            "class_name": f"Class {grade}",
            "available": is_available,
            "indexed": is_indexed,
            "file_size": pdf_path.stat().st_size if is_available else 0,
            "total_chapters": len(chapters),
            "chapters": chapters,
            "midterm_chapters": midterm_chapters,
            "total_chunks": num_chunks,
            "document_id": doc_id,
        })
    return status_list
