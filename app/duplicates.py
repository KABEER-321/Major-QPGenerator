# app/duplicates.py
# Question text normalization, duplicate and near-duplicate detection,
# and candidate pool selection for QPGen.

import re
from typing import Optional

from app.schemas import (
    Question,
    QuestionType,
    ExamBlueprint,
    PaperSection,
    MathCheckStatus,
)


class CandidatePoolShortageError(Exception):
    """Raised when the candidate pool cannot fulfill the requested blueprint versions without duplication."""

    def __init__(
        self,
        message: str,
        section_id: Optional[str] = None,
        needed: int = 0,
        available: int = 0,
    ):
        super().__init__(message)
        self.section_id = section_id
        self.needed = needed
        self.available = available


def normalize_question_text(text: str) -> str:
    """
    Normalize question text for duplicate detection.
    Strips question numbers, punctuation, common filler words,
    and normalizes math symbols and whitespace.
    """
    s = text.strip()

    # 1. Strip leading question numbers like "1.", "Q1.", "Q 1:", "Question 1:", "(a)", "(i)"
    s = re.sub(r"^(?:(?:Q(?:uestion)?\s*\d+[:\.\)]?)|(?:\d+[\.\)]))\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"^\([a-z0-9ivxlcdm]+\)\s*", "", s, flags=re.IGNORECASE)

    # 2. Normalize math symbols
    s = s.replace("²", "^2").replace("³", "^3").replace("⁴", "^4")
    s = s.replace("×", "*").replace("÷", "/").replace("−", "-").replace("±", "+-")

    # 3. Lowercase
    s = s.lower()

    # 4. Remove punctuation except math operators (+ - * / ^ =)
    s = re.sub(r"[^\w\s\+\-\*\/\^\=]", " ", s)

    # 5. Remove common leading filler phrases
    fillers = [
        r"^find\s+the\s+",
        r"^calculate\s+the\s+",
        r"^evaluate\s+the\s+",
        r"^determine\s+the\s+",
        r"^solve\s+the\s+following\s*",
        r"^what\s+is\s+the\s+",
        r"^which\s+of\s+the\s+following\s+is\s+",
        r"^state\s+the\s+",
    ]
    for filler in fillers:
        s = re.sub(filler, "", s)

    # 6. Normalize whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _token_jaccard(tokens1: set[str], tokens2: set[str]) -> float:
    """Calculate Jaccard similarity between two token sets."""
    if not tokens1 and not tokens2:
        return 1.0
    if not tokens1 or not tokens2:
        return 0.0
    intersection = len(tokens1.intersection(tokens2))
    union = len(tokens1.union(tokens2))
    return intersection / union if union > 0 else 0.0


def _char_ngrams(s: str, n: int = 3) -> set[str]:
    """Generate character n-grams."""
    clean = re.sub(r"\s+", "", s)
    if len(clean) < n:
        return {clean} if clean else set()
    return {clean[i : i + n] for i in range(len(clean) - n + 1)}


def calculate_similarity(text1: str, text2: str) -> float:
    """
    Calculate similarity between two question texts (0.0 to 1.0).
    Uses normalized tokens and character 3-grams.
    """
    n1 = normalize_question_text(text1)
    n2 = normalize_question_text(text2)

    if n1 == n2:
        return 1.0

    tokens1 = set(n1.split())
    tokens2 = set(n2.split())
    tok_sim = _token_jaccard(tokens1, tokens2)

    ngrams1 = _char_ngrams(n1, 3)
    ngrams2 = _char_ngrams(n2, 3)
    ngram_sim = _token_jaccard(ngrams1, ngrams2)

    # Return the maximum of token Jaccard and n-gram Jaccard
    return max(tok_sim, ngram_sim)


def is_duplicate(
    question: Question,
    existing_questions: list[Question],
    threshold: float = 0.75,
) -> tuple[bool, Optional[str]]:
    """
    Check if a question is an exact or near duplicate of any question in existing_questions.
    Returns (True, reason) if a duplicate is detected, else (False, None).
    """
    norm_q = normalize_question_text(question.question_text)

    for ex in existing_questions:
        if ex.question_id == question.question_id:
            return True, f"Identical question_id '{question.question_id}'."

        norm_ex = normalize_question_text(ex.question_text)
        if norm_q == norm_ex:
            return True, f"Exact normalized text match with existing question '{ex.question_text}'."

        sim = calculate_similarity(question.question_text, ex.question_text)
        if sim >= threshold:
            return True, f"High textual similarity ({sim:.2f} >= {threshold}) with existing question '{ex.question_text}'."

        # If both are MCQs, check if options and answers match even if question phrasing slightly differed
        if (
            question.question_type == QuestionType.MULTIPLE_CHOICE
            and ex.question_type == QuestionType.MULTIPLE_CHOICE
            and question.options
            and ex.options
        ):
            sorted_opts_q = sorted(opt.strip().lower() for opt in question.options)
            sorted_opts_ex = sorted(opt.strip().lower() for opt in ex.options)
            if sorted_opts_q == sorted_opts_ex:
                if question.correct_option == ex.correct_option or sim >= 0.25:
                    return True, f"Identical options and answer with existing question '{ex.question_text}'."

    return False, None


def deduplicate_candidates(
    candidates: list[Question],
    similarity_threshold: float = 0.75,
) -> tuple[list[Question], int]:
    """
    Filter candidate questions, discarding duplicates and near-duplicates.
    Returns (unique_candidates, rejected_count).
    """
    unique: list[Question] = []
    rejected_count = 0

    for q in candidates:
        is_dup, _ = is_duplicate(q, unique, threshold=similarity_threshold)
        if is_dup:
            rejected_count += 1
        else:
            unique.append(q)

    return unique, rejected_count


def select_version_questions(
    candidate_pool: list[Question],
    blueprint: ExamBlueprint,
    num_versions: int,
) -> tuple[list[list[Question]], list[list[PaperSection]]]:
    """
    Select questions from candidate_pool without replacement across versions.
    Each version must strictly fulfill the blueprint sections, question counts, and marks.

    Returns:
        versions_questions: list of list of Question (one list per version)
        versions_sections: list of list of PaperSection (one list per version)

    Raises:
        CandidatePoolShortageError if candidates cannot satisfy all versions.
    """
    # 1. Filter out any candidate marked with DISCREPANCY in mathematical verification
    valid_pool = [
        q for q in candidate_pool
        if not (q.math_verification and q.math_verification.status == MathCheckStatus.DISCREPANCY)
    ]

    # 2. Group candidates by question_type
    pool_by_type: dict[QuestionType, list[Question]] = {}
    for q in valid_pool:
        pool_by_type.setdefault(q.question_type, []).append(q)

    # 3. For each section in blueprint, verify total available count across num_versions
    # and select disjoint subsets for each version
    version_sections_map: list[list[PaperSection]] = [[] for _ in range(num_versions)]
    used_q_ids: set[str] = set()

    for sec in blueprint.sections:
        needed_per_version = sec.num_questions
        total_needed = needed_per_version * num_versions
        target_marks_dist = sec.get_marks_distribution()

        # Candidates matching question type
        avail = [q for q in pool_by_type.get(sec.question_type, []) if q.question_id not in used_q_ids]

        if len(avail) < total_needed:
            raise CandidatePoolShortageError(
                f"Candidate pool shortage for section '{sec.section_id}' ({sec.title}). "
                f"Requires {total_needed} questions ({needed_per_version} per version * {num_versions} versions), "
                f"but only {len(avail)} valid unique candidates of type '{sec.question_type.value}' are available.",
                section_id=sec.section_id,
                needed=total_needed,
                available=len(avail),
            )

        # Partition candidates into versions
        for v_idx in range(num_versions):
            selected_for_version: list[Question] = []

            for i in range(needed_per_version):
                target_mark = target_marks_dist[i]
                # Look for candidate with matching mark first
                candidate_idx = None
                for idx, cand in enumerate(avail):
                    if cand.question_id not in used_q_ids and cand.marks == target_mark:
                        candidate_idx = idx
                        break

                # If no exact mark match, take any available candidate of this section type
                if candidate_idx is None:
                    for idx, cand in enumerate(avail):
                        if cand.question_id not in used_q_ids:
                            candidate_idx = idx
                            break

                if candidate_idx is None:
                    raise CandidatePoolShortageError(
                        f"Candidate pool shortage for section '{sec.section_id}'. "
                        f"Cannot select {needed_per_version} questions for version {v_idx + 1}.",
                        section_id=sec.section_id,
                        needed=total_needed,
                        available=len(selected_for_version),
                    )

                chosen = avail.pop(candidate_idx)
                # Ensure mark matches blueprint section mark
                if chosen.marks != target_mark:
                    chosen = chosen.model_copy(update={"marks": target_mark})

                used_q_ids.add(chosen.question_id)
                selected_for_version.append(chosen)

            paper_sec = PaperSection(
                section_id=sec.section_id,
                title=sec.title,
                question_type=sec.question_type,
                num_questions=sec.num_questions,
                section_marks=sec.section_marks,
                questions=selected_for_version,
            )
            version_sections_map[v_idx].append(paper_sec)

    # 4. Flatten questions per version
    versions_questions: list[list[Question]] = []
    for v_idx in range(num_versions):
        v_qs = []
        for sec in version_sections_map[v_idx]:
            v_qs.extend(sec.questions)
        versions_questions.append(v_qs)

    return versions_questions, version_sections_map
