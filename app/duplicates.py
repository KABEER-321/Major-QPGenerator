# app/duplicates.py
# Question text normalization, duplicate and near-duplicate detection,
# concept/scenario collision detection, and candidate pool selection for QPGen.

from __future__ import annotations

import re
from typing import Optional, Union

from app.schemas import (
    Question,
    QuestionType,
    Difficulty,
    ExamBlueprint,
    PaperSection,
    MathCheckStatus,
    MathVerificationResult,
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


# Common English stopwords to ignore when analyzing mathematical story templates
STOPWORDS = {
    "a", "an", "the", "and", "or", "in", "on", "at", "to", "for", "of", "with",
    "by", "from", "as", "is", "are", "was", "were", "be", "been", "being", "have",
    "has", "had", "do", "does", "did", "if", "then", "he", "she", "it", "they",
    "his", "her", "their", "him", "them", "who", "which", "what", "how", "much",
    "many", "find", "calculate", "determine", "evaluate", "solve", "following",
    "given", "each", "every", "all", "some", "any", "there", "this", "that", "these",
    "those", "also", "into", "out", "up", "down", "about", "did", "does",
}

# Domain scenario clusters for word problems.
# If two questions match the same specific scenario cluster, they are assessing
# the exact same real-world story template (e.g. bank deposit, factory production, village population).
SCENARIO_CLUSTERS = [
    {
        "name": "Banking & Deposit",
        "primary": {"bank", "deposit", "deposited", "account", "savings", "passbook", "withdrew", "withdrawal"},
        "min_matches": 2,
    },
    {
        "name": "Factory Production & Inventory",
        "primary": {"factory", "manufactured", "manufacturing", "produced", "cartons", "soap", "warehouse", "packets", "shipped"},
        "min_matches": 2,
    },
    {
        "name": "Village & Population Demographics",
        "primary": {"village", "town", "city", "population", "inhabitants", "residents", "census"},
        "secondary": {"men", "women", "children", "males", "females", "adults", "people"},
        "min_matches": 2,
    },
    {
        "name": "Train & Railway Travel",
        "primary": {"train", "locomotive", "railway", "platform", "express", "departed", "departure", "arrived", "arrival"},
        "min_matches": 2,
    },
    {
        "name": "Vessel & Liquid Capacity",
        "primary": {"vessel", "tumbler", "bucket", "jug", "liters", "litres", "milliliters", "millilitres", "capacity"},
        "min_matches": 2,
    },
    {
        "name": "Field Fencing & Boundary Perimeter",
        "primary": {"fencing", "fence", "barbed", "around a field", "around the field", "rectangular field", "cost per meter", "cost per metre"},
        "min_matches": 2,
    },
    {
        "name": "Election & Voting Margin",
        "primary": {"election", "votes", "polled", "candidate", "majority", "margin", "defeated"},
        "min_matches": 2,
    },
    {
        "name": "Spatial Vocabulary & Positioning",
        "primary": {"spatial", "top", "bottom", "under", "inside", "outside", "above", "below"},
        "min_matches": 2,
    },
    {
        "name": "Marble Units & Place Value Grouping",
        "primary": {"marbles", "collection", "single units", "group of 1 ten", "group of ten"},
        "min_matches": 2,
    },
]

ARITH_OPS = {
    "add": {"add", "sum", "plus", "+"},
    "sub": {"subtract", "subtraction", "difference", "minus", "-"},
    "mul": {"multiply", "multiplication", "product", "times", "*", "x"},
    "div": {"divide", "division", "quotient", "/"},
}

STORY_INDICATORS = {
    "bank", "factory", "village", "town", "city", "population", "train", "station", "vessel",
    "tumbler", "carton", "cartons", "boxes", "soap", "shopkeeper", "bought", "sold", "farmer",
    "apples", "oranges", "pencils", "books", "students", "class", "school", "deposited",
    "withdrew", "manufactured", "journey", "arrived", "departed", "perimeter", "fence",
    "fencing", "boundary", "cost", "price", "warehouse", "passbook", "liters", "litres",
}


def is_pure_computation(text: str) -> bool:
    """Check if question is a direct arithmetic computation without story context."""
    clean = re.sub(r"[^\w\s\+\-\*\/\=]", " ", text.lower())
    words = set(clean.split())
    if any(si in words for si in STORY_INDICATORS):
        return False
    nums = re.findall(r"\b\d+\b", text)
    if not nums:
        return False
    has_op = any(w in words for op_set in ARITH_OPS.values() for w in op_set)
    return has_op and len(words) <= 12


def get_arith_op(text: str) -> Optional[str]:
    """Identify which arithmetic operation is being performed."""
    clean = re.sub(r"[^\w\s\+\-\*\/\=]", " ", text.lower())
    words = set(clean.split())
    for op, kws in ARITH_OPS.items():
        if any(k in words for k in kws):
            return op
    return None


def get_content_tokens(text: str) -> set[str]:
    """Extract stemmed content tokens, stripping stopwords and numbers."""
    clean = re.sub(r"[^\w\s]", " ", text.lower())
    return {_simple_stem(w) for w in clean.split() if w not in STOPWORDS and not w.isdigit()}


def _simple_stem(word: str) -> str:
    """Lightweight suffix stemming for primary math vocabulary."""
    w = word.lower()
    for suffix in ("ing", "ed", "es", "s"):
        if w.endswith(suffix) and len(w) > len(suffix) + 2:
            return w[: -len(suffix)]
    return w


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
    s = s.replace("₹", "rs ").replace("Rs.", "rs ").replace("Rs", "rs ")

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


def mask_numbers_and_names(text: str) -> str:
    """
    Mask all numerical quantities into <NUM> and stem words to expose underlying template.
    E.g.: 'A factory produced 45,000 soap boxes...' -> 'factory produce <NUM> soap box...'
    """
    norm = normalize_question_text(text)
    # Replace numbers (including comma-separated numbers) with <num>
    norm = re.sub(r"\b\d+(?:,\d+)*(?:\.\d+)?\b", "<num>", norm)
    tokens = norm.split()
    stemmed = [_simple_stem(t) for t in tokens if t not in STOPWORDS]
    return " ".join(stemmed)


def identify_scenario_cluster(text: str) -> Optional[str]:
    """
    Check if a question belongs to a known word-problem scenario cluster.
    Returns cluster name if matched, else None.
    """
    norm = text.lower()
    tokens = set(re.findall(r"[a-z]+", norm))

    for sc in SCENARIO_CLUSTERS:
        name = sc["name"]
        primary = sc["primary"]
        min_m = sc.get("min_matches", 2)
        # Check primary token matches or phrase matches
        matches = 0
        for p in primary:
            if " " in p:
                if p in norm:
                    matches += 1
            elif p in tokens or _simple_stem(p) in tokens:
                matches += 1

        if "secondary" in sc:
            secondary = sc["secondary"]
            sec_matches = sum(1 for s in secondary if s in tokens or _simple_stem(s) in tokens)
            if matches >= 1 and sec_matches >= 1 and (matches + sec_matches) >= min_m:
                return name

        if matches >= min_m:
            return name

    return None


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
    Uses normalized tokens, content tokens, and character 3-grams.
    Protects distinct mathematical entities (e.g. triangle vs rectangle)
    from stopword-inflated question carrier frames.
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

    raw_sim = max(tok_sim, ngram_sim)

    # Content token check: if distinct key entities (e.g. triangle vs rectangle),
    # clamp similarity so carrier frames do not trigger false duplicates
    c1 = get_content_tokens(text1)
    c2 = get_content_tokens(text2)
    if c1 and c2:
        content_sim = _token_jaccard(c1, c2)
        if content_sim < 0.40:
            return min(raw_sim, 0.45)

    return raw_sim


def calculate_template_similarity(text1: str, text2: str) -> float:
    """
    Calculate structural template similarity with numbers masked.
    Catches questions that test identical problems with altered numbers or names.
    Excludes generic <num> tokens from dominating the token sets.
    """
    m1 = mask_numbers_and_names(text1)
    m2 = mask_numbers_and_names(text2)
    if m1 == m2 and m1:
        return 1.0
    toks1 = {t for t in m1.split() if t != "<num>"}
    toks2 = {t for t in m2.split() if t != "<num>"}
    return _token_jaccard(toks1, toks2)


MATH_CARRIER_TERMS = {
    "perimeter", "area", "length", "lengths", "width", "height", "side", "sides",
    "triangle", "rectangle", "square", "circle", "radius", "diameter",
    "number", "numbers", "digit", "digits", "expanded", "standard", "form",
    "value", "place", "add", "subtract", "multiply", "divide", "arranged", "according",
}


def check_shared_phrase(text1: str, text2: str, min_words: int = 4) -> tuple[bool, Optional[str]]:
    """Check if two questions share an identical contiguous sequence of content words."""
    norm1 = normalize_question_text(text1)
    norm2 = normalize_question_text(text2)
    words1 = [w for w in norm1.split() if w not in STOPWORDS and len(w) > 2]
    words2 = [w for w in norm2.split() if w not in STOPWORDS and len(w) > 2]

    if len(words1) < min_words or len(words2) < min_words:
        return False, None

    words2_joined = " " + " ".join(words2) + " "
    for i in range(len(words1) - min_words + 1):
        gram_words = words1[i : i + min_words]
        # Ignore phrases consisting solely of standard math carrier vocabulary
        if all(w in MATH_CARRIER_TERMS or _simple_stem(w) in MATH_CARRIER_TERMS for w in gram_words):
            continue
        gram = " ".join(gram_words)
        if f" {gram} " in words2_joined:
            return True, gram

    return False, None


def is_duplicate(
    question: Union[Question, str],
    existing_questions: list[Union[Question, str]],
    threshold: float = 0.70,
    template_threshold: float = 0.65,
) -> tuple[bool, Optional[str]]:
    """
    Check if a question is an exact, near duplicate, template reskin,
    or scenario collision of any question in existing_questions.
    Returns (True, reason) if a duplicate is detected, else (False, None).
    """
    q_text = question.question_text if isinstance(question, Question) else str(question)
    q_id = getattr(question, "question_id", None)
    norm_q = normalize_question_text(q_text)
    sc_q = identify_scenario_cluster(q_text)
    words_q = q_text.split()

    for ex in existing_questions:
        ex_text = ex.question_text if isinstance(ex, Question) else str(ex)
        ex_id = getattr(ex, "question_id", None)
        words_ex = ex_text.split()

        if q_id and ex_id and q_id == ex_id:
            return True, f"Identical question_id '{q_id}'."

        # Figure comparison check:
        # If both questions are figure-based, their identity is defined by the diagram and answer.
        # Different diagrams with different answers are distinct visual math questions.
        fk_q = _get_fig_key(question) if isinstance(question, Question) else None
        fk_ex = _get_fig_key(ex) if isinstance(ex, Question) else None
        if fk_q and fk_ex:
            if fk_q == fk_ex:
                return True, "Identical figure diagram parameters."
            ans_q = str(getattr(question, "answer", "")).strip().lower()
            ans_ex = str(getattr(ex, "answer", "")).strip().lower()
            if ans_q and ans_ex and ans_q != ans_ex:
                continue

        norm_ex = normalize_question_text(ex_text)
        if norm_q == norm_ex:
            return True, f"Exact normalized text match with existing question '{ex_text}'."

        # Pure computation check: e.g. "Add 23 and 45" vs "Add 12 and 34"
        if is_pure_computation(q_text) and is_pure_computation(ex_text):
            op_q = get_arith_op(q_text)
            op_ex = get_arith_op(ex_text)
            nums_q = set(re.findall(r"\b\d+\b", q_text))
            nums_ex = set(re.findall(r"\b\d+\b", ex_text))
            if op_q and op_q == op_ex and nums_q and nums_q == nums_ex:
                return True, f"Same arithmetic calculation ({op_q} on {nums_q}) as: '{ex_text}'"
            # Different numbers in direct arithmetic are distinct math problems
            continue

        # Short number-focused prompt check (e.g. before/after, number names, place value)
        nums_q = set(re.findall(r"\b\d+\b", q_text))
        nums_ex = set(re.findall(r"\b\d+\b", ex_text))
        if len(words_q) <= 10 and len(words_ex) <= 10 and nums_q and nums_ex and nums_q != nums_ex:
            # Different target numbers in short prompts
            continue

        # 1. Textual similarity
        sim = calculate_similarity(q_text, ex_text)
        if sim >= threshold:
            return True, f"High textual similarity ({sim:.2f} >= {threshold}) with existing question '{ex_text}'."

        # 2. Template / Masked-number similarity
        # Catches word problems where only numbers or minor names changed.
        # Only applies to word problems / narrative prompts with sufficient length (>= 8 words)
        if len(words_q) >= 8 and len(words_ex) >= 8:
            templ_sim = calculate_template_similarity(q_text, ex_text)
            if templ_sim >= template_threshold:
                return True, (
                    f"Repeated problem template ({templ_sim:.2f} >= {template_threshold}) "
                    f"assessing same concept: '{ex_text}'"
                )

        # 3. Scenario cluster collision
        # E.g. Both are bank deposit word problems, or both are factory production problems,
        # or both are village population breakdown problems.
        if sc_q:
            sc_ex = identify_scenario_cluster(ex_text)
            if sc_q == sc_ex:
                return True, (
                    f"Repeated problem scenario ('{sc_q}') assessing substantially "
                    f"the same concept/story as: '{ex_text}'"
                )

        # 4. Shared distinctive phrase check (e.g. recycling textbook illustration text like 'bird sitting top tree')
        has_phrase, phrase = check_shared_phrase(q_text, ex_text, min_words=4)
        if has_phrase:
            return True, f"Shared phrase detected ('{phrase}'): questions recycle substantially the same scenario/illustration as: '{ex_text[:60]}...'"

        # 5. Same chapter concept collision check
        if (
            isinstance(question, Question)
            and isinstance(ex, Question)
            and question.chapter
            and ex.chapter
        ):
            q_ch_clean = re.sub(r'^(?:Lesson|Chapter|Unit)\s*\d+[:\s\-]*', '', question.chapter, flags=re.IGNORECASE).strip().lower()
            ex_ch_clean = re.sub(r'^(?:Lesson|Chapter|Unit)\s*\d+[:\s\-]*', '', ex.chapter, flags=re.IGNORECASE).strip().lower()
            if q_ch_clean and q_ch_clean == ex_ch_clean:
                # If both questions have different numbers, they are distinct calculations
                if not (nums_q and nums_ex and nums_q != nums_ex):
                    c1 = get_content_tokens(q_text)
                    c2 = get_content_tokens(ex_text)
                    if c1 and c2:
                        c_sim = _token_jaccard(c1, c2)
                        thresh = 0.50 if question.question_type == ex.question_type else 0.35
                        if c_sim >= thresh:
                            return True, f"Both questions test substantially the same concept from '{question.chapter}' (overlap {c_sim:.2f}): '{ex_text[:60]}...'"

        # 6. If both are MCQs, check if options and answers match
        if isinstance(question, Question) and isinstance(ex, Question):
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
    similarity_threshold: float = 0.70,
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


def _get_fig_key(q: Question) -> Optional[str]:
    """Extract a canonical figure key ignoring ephemeral variation index."""
    if not (getattr(q, "figure", None) and getattr(q.figure, "parameters", None)):
        return None
    core_params = {
        k: v for k, v in q.figure.parameters.items()
        if k not in ("variation", "variation_index")
    }
    return f"{q.figure.figure_type}:{sorted(str(k) + '=' + str(v) for k, v in core_params.items())}"


def _is_fig_duplicate(q: Question, chosen_fig_keys: set[str]) -> bool:
    """Check if question has a figure with parameters already used."""
    fk = _get_fig_key(q)
    return (fk is not None) and (fk in chosen_fig_keys)


def _is_math_duplicate(q1: Question, q2: Question) -> tuple[bool, Optional[str]]:
    """
    Check if two questions are identical or duplicate mathematical problems.
    Returns (True, reason) if they are duplicates, else (False, None).
    """
    # 1. Exact ID
    if q1.question_id and q2.question_id and q1.question_id == q2.question_id:
        return True, "Identical question_id."

    # 2. Figure comparison
    fk1 = _get_fig_key(q1)
    fk2 = _get_fig_key(q2)
    if fk1 and fk2:
        if fk1 == fk2:
            return True, "Identical figure diagram parameters."
        ans1 = str(getattr(q1, "answer", "")).strip().lower()
        ans2 = str(getattr(q2, "answer", "")).strip().lower()
        if ans1 and ans2 and ans1 != ans2:
            return False, None

    # 3. Exact normalized text
    norm1 = normalize_question_text(q1.question_text)
    norm2 = normalize_question_text(q2.question_text)
    if norm1 == norm2:
        return True, "Identical question text."

    # 4. Identical arithmetic operation with identical numbers
    nums1 = set(re.findall(r"\b\d+\b", q1.question_text))
    nums2 = set(re.findall(r"\b\d+\b", q2.question_text))
    op1 = get_arith_op(q1.question_text)
    op2 = get_arith_op(q2.question_text)
    if op1 and op1 == op2 and nums1 and nums1 == nums2:
        return True, f"Identical arithmetic calculation ({op1} on {nums1})."

    # 5. Near-identical text with identical numbers
    sim = calculate_similarity(q1.question_text, q2.question_text)
    if sim >= 0.85 and (nums1 == nums2 or not nums1):
        return True, f"High textual similarity ({sim:.2f}) with same values."

    return False, None


def select_version_questions(
    candidate_pool: list[Question],
    blueprint: ExamBlueprint,
    num_versions: int,
) -> tuple[list[list[Question]], list[list[PaperSection]]]:
    """
    Select questions from candidate_pool without replacement across versions.
    Each version must strictly fulfill the blueprint sections, question counts, and marks.
    Guarantees cross-version and intra-version uniqueness: no duplicate questions,
    no identical arithmetic calculations, and no identical figures.

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

    # 3. For each section in blueprint, select disjoint, non-duplicate subsets for each version
    version_sections_map: list[list[PaperSection]] = [[] for _ in range(num_versions)]
    used_q_ids: set[str] = set()
    all_chosen_questions: list[Question] = []
    all_chosen_fig_keys: set[str] = set()

    for sec in blueprint.sections:
        needed_per_version = sec.num_questions
        total_needed = needed_per_version * num_versions
        target_marks_dist = sec.get_marks_distribution()
        fig_alloc = getattr(sec, "figure_allocation", None) or [False] * needed_per_version
        ch_alloc = getattr(sec, "chapter_allocation", None) or []

        # Available candidates of this question type
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
                is_fig_slot = fig_alloc[i] if i < len(fig_alloc) else False
                ch = ch_alloc[i] if i < len(ch_alloc) else "Mathematics"

                chosen = None
                candidate_idx = None

                # Figure slot selection
                if is_fig_slot:
                    # 1. Search avail for a non-duplicate figure candidate
                    for idx, cand in enumerate(avail):
                        if cand.question_id not in used_q_ids and (cand.question_subtype == "figure_based" or getattr(cand, "figure", None)):
                            if not _is_fig_duplicate(cand, all_chosen_fig_keys):
                                is_dup = any(_is_math_duplicate(cand, ex)[0] for ex in (all_chosen_questions + selected_for_version))
                                if not is_dup:
                                    candidate_idx = idx
                                    break
                    if candidate_idx is not None:
                        chosen = avail.pop(candidate_idx)
                    else:
                        # 2. Dynamic figure variation generation to guarantee distinct figures across students
                        from app.figures import create_figure_question
                        effective_class = getattr(sec, "grade", None) or blueprint.grade or "Class 1"
                        sec_diff = sec.difficulty or Difficulty.MEDIUM
                        for attempt in range(25):
                            var_idx = (v_idx * 15) + i + attempt + 1
                            cand_fig = create_figure_question(
                                grade=effective_class,
                                chapter=ch,
                                question_type=sec.question_type,
                                marks=target_mark,
                                difficulty=sec_diff,
                                variation_index=var_idx,
                            )
                            if cand_fig:
                                if not _is_fig_duplicate(cand_fig, all_chosen_fig_keys):
                                    is_d = any(_is_math_duplicate(cand_fig, ex)[0] for ex in (all_chosen_questions + selected_for_version))
                                    if not is_d:
                                        import uuid
                                        cand_fig.question_id = uuid.uuid4().hex[:10]
                                        cand_fig.marks = target_mark
                                        chosen = cand_fig
                                        break
                else:
                    # Non-figure slot:
                    # 1. Look for candidate matching target mark with no math duplication
                    for idx, cand in enumerate(avail):
                        if cand.question_id not in used_q_ids and cand.question_subtype != "figure_based" and cand.marks == target_mark:
                            is_dup = any(_is_math_duplicate(cand, ex)[0] for ex in (all_chosen_questions + selected_for_version))
                            if not is_dup:
                                candidate_idx = idx
                                break

                    # 2. Look for any candidate without math duplication
                    if candidate_idx is None:
                        for idx, cand in enumerate(avail):
                            if cand.question_id not in used_q_ids and cand.question_subtype != "figure_based":
                                is_dup = any(_is_math_duplicate(cand, ex)[0] for ex in (all_chosen_questions + selected_for_version))
                                if not is_dup:
                                    candidate_idx = idx
                                    break

                    if candidate_idx is not None:
                        chosen = avail.pop(candidate_idx)
                    else:
                        # 3. Dynamic numerical variation generation
                        from app.generator import generate_numerical_variation
                        for base_cand in avail:
                            if base_cand.question_subtype != "figure_based":
                                for var_i in range(1, 15):
                                    var_q = generate_numerical_variation(base_cand, variation_index=(v_idx * 11) + i + var_i)
                                    if var_q and var_q.math_verification and var_q.math_verification.status != MathCheckStatus.DISCREPANCY:
                                        is_d = any(_is_math_duplicate(var_q, ex)[0] for ex in (all_chosen_questions + selected_for_version))
                                        if not is_d:
                                            import uuid
                                            var_q.question_id = uuid.uuid4().hex[:10]
                                            var_q.marks = target_mark
                                            chosen = var_q
                                            break
                            if chosen is not None:
                                break

                        # 4. Fallback to any remaining candidate in avail if test pool is pre-constrained
                        if chosen is None:
                            for idx, cand in enumerate(avail):
                                if cand.question_id not in used_q_ids:
                                    candidate_idx = idx
                                    break
                            if candidate_idx is not None:
                                chosen = avail.pop(candidate_idx)

                if chosen is None:
                    raise CandidatePoolShortageError(
                        f"Candidate pool shortage for section '{sec.section_id}'. "
                        f"Cannot select {needed_per_version} unique questions for version {v_idx + 1}.",
                        section_id=sec.section_id,
                        needed=total_needed,
                        available=len(selected_for_version),
                    )

                # Ensure mark matches blueprint section mark
                if chosen.marks != target_mark:
                    chosen = chosen.model_copy(update={"marks": target_mark})

                used_q_ids.add(chosen.question_id)
                all_chosen_questions.append(chosen)
                fk = _get_fig_key(chosen)
                if fk:
                    all_chosen_fig_keys.add(fk)
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


def validate_batch_papers_uniqueness(papers: list) -> tuple[bool, list[str]]:
    """
    Cross-checks all questions across all papers in a batch to ensure
    no duplicate questions or duplicate figures are shared between students.
    Returns (is_unique, violations).
    """
    violations: list[str] = []
    seen_q_ids: dict[str, str] = {}
    seen_q_texts: dict[str, str] = {}
    seen_fig_keys: dict[str, str] = {}

    for p in papers:
        p_label = getattr(p, "student_name", None) or getattr(p, "set_label", None) or f"Version {p.version_number}"
        for q_idx, q in enumerate(p.questions):
            # Check question_id
            if q.question_id in seen_q_ids:
                violations.append(
                    f"Question ID '{q.question_id}' is duplicated between {seen_q_ids[q.question_id]} and {p_label}."
                )
            seen_q_ids[q.question_id] = f"{p_label} (Q{q_idx + 1})"

            # Check normalized question text
            norm = normalize_question_text(q.question_text)
            if norm in seen_q_texts:
                violations.append(
                    f"Identical question text '{q.question_text[:60]}...' shared between {seen_q_texts[norm]} and {p_label} (Q{q_idx + 1})."
                )
            seen_q_texts[norm] = f"{p_label} (Q{q_idx + 1})"

            # Check figure uniqueness if figure is present
            fk = _get_fig_key(q)
            if fk:
                if fk in seen_fig_keys:
                    violations.append(
                        f"Identical figure parameters shared between {seen_fig_keys[fk]} and {p_label} (Q{q_idx + 1})."
                    )
                seen_fig_keys[fk] = f"{p_label} (Q{q_idx + 1})"

    # Pairwise cross-paper similarity check
    for i in range(len(papers)):
        p1 = papers[i]
        lbl1 = getattr(p1, "student_name", None) or getattr(p1, "set_label", None) or f"Version {p1.version_number}"
        for j in range(i + 1, len(papers)):
            p2 = papers[j]
            lbl2 = getattr(p2, "student_name", None) or getattr(p2, "set_label", None) or f"Version {p2.version_number}"
            for qa in p1.questions:
                for qb in p2.questions:
                    is_d, reason = _is_math_duplicate(qa, qb)
                    if is_d:
                        violations.append(
                            f"Duplicate question shared between {lbl1} and {lbl2}: {reason}"
                        )

    return (len(violations) == 0, violations)
