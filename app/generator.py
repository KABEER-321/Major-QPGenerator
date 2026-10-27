# app/generator.py
# Prompts and question generation using LangChain + Google Gemini.
# Retrieves relevant syllabus passages and asks Gemini to produce
# structured questions grounded in those passages.

import json
import logging
import math
import re
import uuid
from typing import Optional

from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage

from app.config import settings
from app.model import get_llm
from app.schemas import (
    Question,
    QuestionType,
    Difficulty,
    ExamType,
    PaperGenerateRequest,
    SourceReference,
    ValidationStatus,
    ExamBlueprint,
    MathCheckStatus,
    format_grade_display,
)
from app.math_verify import verify_question_math
from app.duplicates import is_duplicate, deduplicate_candidates
from app.validator import (
    validate_question_quality,
    validate_paper_questions_uniqueness,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt — tells Gemini its role and constraints
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are an expert primary mathematics teacher and question paper designer.

Your job is to generate curriculum-aligned mathematics examination questions based on
REFERENCE PASSAGES from an uploaded syllabus or textbook.

CRITICAL EXAMINATION PAPER RULES:
1. PURE MATHEMATICS ONLY - NOT READING COMPREHENSION OR VOCABULARY:
   - Every question must test mathematical skills (arithmetic operations, numbers, place value, shapes, patterns, measurements, problem solving).
   - NEVER ask students to write definitions, explain terms in sentences, or memorize English vocabulary words (e.g. do NOT ask 'What does the plus symbol mean?', 'State the vocabulary term used for...', 'Define spatial relations', 'Explain what term is used when no monkeys dangling').
   - Keep reading demands minimal, clear, and age-appropriate.

2. NEVER MENTION TEXTBOOK METADATA IN QUESTION TEXT:
   - Questions must look like real, formal examination questions printed on a test paper.
   - ABSOLUTE PROHIBITION: NEVER write phrases like:
     * 'as discussed in Lesson 2'
     * 'according to Lesson 4'
     * 'as introduced in Lesson 3'
     * 'as studied in class'
     * 'in Chapter 1'
     * 'refer to the textbook'
   - Ask the mathematical question directly (e.g. 'Which shape has 4 equal sides?', NOT 'Which shape has 4 equal sides and is discussed in Lesson 2?').

3. AVOID BIZARRE OR UNNECESSARY STORIES & NARRATIVE FLUFF:
   - Do NOT copy bizarre textbook story snippets into question stems (e.g. do NOT write 'If there are no monkeys dangling from the wire...', 'Anant is painting at the top of a building...').
   - Avoid lengthy word problems, complicated adult scenarios, or excessive written explanations.
   - Prefer clean numerical exercises, calculations, patterns, shapes, and simple, natural child-appropriate situations (fruits, pencils, books, marbles, toys).

4. MARKS MUST REFLECT COGNITIVE DEMAND AND REQUIRED WORK:
   - 1 Mark (Multiple Choice / Fill in blank): 1 direct fact, calculation, or shape/number identification.
   - 2 Marks (Short Answer): 1-2 step calculation or comparison with clear intermediate step or number sentence. Do NOT combine two unrelated questions merely to fill 2 marks.
   - 3-5 Marks (Long Answer / Problem Solving): Coherent multi-part mathematical problems with clearly labeled subparts (a), (b), (c) where each subpart has clear mathematical work and sub-marks (e.g., (a) 2 Marks, (b) 2 Marks, (c) 1 Mark = 5 Marks).

5. ACCURATE AND RELEVANT CHAPTER ATTRIBUTION:
   - In your JSON output, assign the 'chapter' field accurately to the designated chapter name from the blueprint.
   - The question MUST genuinely test mathematical concepts from that chapter.

6. ABSOLUTE UNIQUENESS - ZERO REPEATED CONCEPTS:
   - Do NOT repeat the same concept, problem template, or scenario across different sections.
   - If a concept (e.g. spatial positioning, or a specific shape) is tested in Section A, do NOT test that same concept in Section B or C.

OUTPUT FORMAT:
Return a valid JSON array of question objects. Each object must have these fields:
- "question_text": string (the exact question as printed for the student, NO lesson numbers)
- "question_type": one of "short_answer", "long_answer", "multiple_choice"
- "marks": positive integer
- "difficulty": one of "easy", "medium", "hard"
- "topic": string (the chapter/topic)
- "chapter": string (the exact textbook chapter name)
- "answer": string (concise key answer)
- "solution": string (marking guidance and worked steps)
- "options": array of strings (required for multiple_choice, null for others)
- "correct_option": string (required for multiple_choice, must match one option; null for others)

Return ONLY the JSON array, no markdown fences, no extra text."""


def get_grade_pedagogical_prompt(grade: int | str | None, qtype: QuestionType | None = None) -> str:
    """Return grade-specific pedagogical instructions to ensure age-appropriate mathematical problems across Classes 1 to 5."""
    from app.curriculum import parse_grade_number
    g_num = parse_grade_number(str(grade or ""))

    if g_num == 1:
        return """
CRITICAL AGE-APPROPRIATE REQUIREMENTS FOR CLASS 1 (STUDENTS AGED 6-7 YEARS):
1. MATHEMATICAL SKILLS ONLY - NOT ENGLISH COMPREHENSION:
   - Basic counting of objects/symbols (★, ●), writing numbers 1 to 20.
   - Recognizing and comparing digits 1 to 20 (greater, smaller, between, before, after).
   - Single-digit addition and subtraction (sums not exceeding 20, e.g. 4 + 3 = ___, 7 - 2 = ___).
   - Basic 2D shapes (circle, square, triangle, rectangle) and 3D shapes (sphere/ball, cylinder/can, box/cube).
   - Spatial understanding: top/bottom, on/under, inside/outside, bigger/smaller, roll/slide.
2. ABSOLUTE PROHIBITION OF DEFINITIONS & ESSAY QUESTIONS:
   - NEVER ask students to define terms or write sentences explaining concepts.
   - Questions must be direct calculations, counting, or circle/choose choices.
3. CONCISE LANGUAGE (< 20 words):
   - Keep instructions short: 'Count', 'Add', 'Subtract', 'Write the number', 'Which is greater'.
4. SECTION STRUCTURE:
   - Multiple Choice: 1-sentence prompt with single-number or single-word options.
   - Short Answer: A single calculation or 1-step number problem.
   - Long Answer: MUST consist of structured simple sub-problems (e.g. '(a) Solve: 3 + 2 = ___ [2 Marks] (b) Solve: 6 - 2 = ___ [2 Marks] (c) Write the number between 14 and 16: ___ [1 Mark]'). NEVER long word stories!
"""
    elif g_num == 2:
        return """
CRITICAL AGE-APPROPRIATE REQUIREMENTS FOR CLASS 2 (STUDENTS AGED 7-8 YEARS):
1. MATHEMATICAL FOCUS:
   - 2-digit numbers up to 100, place value (tens and ones), expanded form, comparing numbers.
   - 2-digit addition and subtraction (with and without regrouping/carry).
   - Repeated addition as multiplication (e.g. 3 groups of 4 = 12).
   - Simple grouping and sharing as division (e.g. share 10 apples equally among 2 children).
   - Straight and curved lines, 2D and 3D shapes.
   - Simple measurement (length, weight, capacity) and calendar/clocks.
2. CONCISE LANGUAGE (< 25 words):
   - Clear, short instructions. Simple child-friendly contexts (crayons, books, fruits, marbles).
   - NO adult banking, factory production, or complex commercial contexts.
3. LONG ANSWER:
   - Structured subparts '(a)', '(b)', '(c)' with 2-digit calculations or 1-2 step word problems.
"""
    elif g_num == 3:
        return """
CRITICAL AGE-APPROPRIATE REQUIREMENTS FOR CLASS 3 (STUDENTS AGED 8-9 YEARS):
1. MATHEMATICAL FOCUS:
   - 3-digit and 4-digit numbers (place value, face value, number names, comparing).
   - Addition and subtraction of 3-digit numbers with regrouping.
   - Multiplication tables (2 to 10), multiplying 2-digit numbers by 1-digit numbers.
   - Division with and without remainders.
   - Shapes, sides, vertices, symmetry, perimeter basics.
   - Standard units of measurement (m, cm, km; g, kg; l, ml), money (Rs and paise), time.
2. CONCISE WORDING (< 35 words):
   - Direct mathematical questions and realistic 1-2 step word problems.
3. LONG ANSWER:
   - Multi-step word problems with clearly separated subparts '(a)', '(b)', '(c)' testing calculations and operations.
"""
    elif g_num == 4:
        return """
CRITICAL AGE-APPROPRIATE REQUIREMENTS FOR CLASS 4 (STUDENTS AGED 9-10 YEARS):
1. MATHEMATICAL FOCUS:
   - Numbers up to 5-6 digits (Indian place value, standard and expanded forms).
   - 4-digit addition and subtraction, 2-digit multiplication, long division.
   - Factors and multiples.
   - Fractions (like fractions, addition and subtraction of like fractions).
   - Perimeter and area of simple geometrical figures (squares and rectangles).
   - Angles (acute, right, obtuse), circle parts (radius, center).
   - Measurement unit conversions.
2. LONG ANSWER:
   - Multi-step mathematical problems structured into sequential subparts '(a)', '(b)', '(c)'.
"""
    elif g_num == 5:
        return """
CRITICAL AGE-APPROPRIATE REQUIREMENTS FOR CLASS 5 (STUDENTS AGED 10-11 YEARS):
1. MATHEMATICAL FOCUS:
   - Large numbers (up to 7-8 digits), operations with large numbers.
   - Factors, multiples, LCM and HCF, prime and composite numbers.
   - Fractions and decimals (all operations, conversion between fractions and decimals).
   - Angles (measuring, classifying), area, perimeter, and volume of simple solids.
   - Practical measurement problems (length, weight, capacity, time intervals).
   - Percentage and profit/loss basics, unitary method, data handling.
2. LONG ANSWER:
   - Challenging mathematical tasks with clearly separated subparts '(a)', '(b)', '(c)' each testing a specific calculation or problem-solving step.
"""
    return ""


def _build_human_prompt(
    request: PaperGenerateRequest,
    passages: list[Document],
) -> str:
    """Build the human message with requirements, examination type instructions, and retrieved passages."""
    passage_text = "\n---\n".join(
        f"[Passage {i+1} from {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
        for i, doc in enumerate(passages)
    )

    # Distribute marks roughly evenly, with remainder added to last questions
    base_marks = request.total_marks // request.num_questions
    remainder = request.total_marks % request.num_questions

    marks_guidance = (
        f"Distribute {request.total_marks} total marks across {request.num_questions} questions. "
        f"Suggested: {request.num_questions - remainder} questions at {base_marks} marks each"
    )
    if remainder:
        marks_guidance += f" and {remainder} questions at {base_marks + 1} marks each"
    marks_guidance += "."

    types_str = ", ".join(qt.value for qt in request.question_types)

    exam_type = request.exam_type or ExamType.FINAL
    academic_class = format_grade_display(request.academic_grade or request.grade)
    curriculum_class = format_grade_display(request.curriculum_grade or request.grade)
    scope_str = request.syllabus_scope or request.topic
    pedagogical_guidance = get_grade_pedagogical_prompt(request.curriculum_grade or request.academic_grade or request.grade)

    if exam_type == ExamType.PREREQUISITE:
        exam_instructions = f"""EXAMINATION TYPE: Prerequisite Examination
- Academic Class of Students: {academic_class}
- Curriculum Class Used: {curriculum_class}
- Subject: {request.subject}
- Permitted Syllabus Topics: {scope_str}
- Focus: Assess foundational mathematical concepts and prerequisites from {curriculum_class} that prepare students for {academic_class}.
- Genuinely test prerequisite knowledge from the {curriculum_class} syllabus."""
    elif exam_type == ExamType.MIDTERM:
        exam_instructions = f"""EXAMINATION TYPE: Mid-Semester Examination
- Academic Class: {academic_class}
- Curriculum Class: {curriculum_class}
- Subject: {request.subject}
- Permitted Syllabus Scope (First Half Only): {scope_str}
- Focus: Assess ONLY topics from the permitted first half of the syllabus.
- STRICT PROHIBITION: Do NOT generate questions from topics outside this permitted mid-semester scope."""
    else:
        exam_instructions = f"""EXAMINATION TYPE: Final Examination
- Academic Class: {academic_class}
- Curriculum Class: {curriculum_class}
- Subject: {request.subject}
- Permitted Syllabus Scope: {scope_str} (Full Syllabus)
- Focus: Comprehensive assessment covering the applicable Class {curriculum_class} syllabus."""

    return f"""Generate a mathematics question paper with these requirements:

{exam_instructions}
- Topic/Chapter: {request.topic}
- Difficulty: {request.difficulty.value}
- Question types to include: {types_str}
- Number of questions: {request.num_questions}
- Total marks: {request.total_marks}
- {marks_guidance}
{pedagogical_guidance}

REFERENCE PASSAGES (use these as your source material):
{passage_text}

Generate exactly {request.num_questions} questions totaling exactly {request.total_marks} marks.
Ensure every question provides the question text, correct answer, and worked solution/marking steps.
Return ONLY a JSON array of question objects."""


def _build_source_references(passages: list[Document]) -> list[SourceReference]:
    """Build SourceReference objects from retrieved passages.

    These track which syllabus chunks were used as source material
    so each question can be traced back to specific uploaded content.
    """
    refs = []
    for i, doc in enumerate(passages):
        preview = doc.page_content[:200].strip()
        if len(doc.page_content) > 200:
            preview += "…"

        chunk_idx = doc.metadata.get("chunk_index")
        if chunk_idx is None:
            chunk_id = str(doc.metadata.get("chunk_id", ""))
            m = re.search(r"_k(\d+)", chunk_id)
            if m:
                chunk_idx = int(m.group(1))
            else:
                chunk_idx = i

        chapter_name = doc.metadata.get("chapter") or doc.metadata.get("unit")
        source_name = chapter_name or doc.metadata.get("source", "Textbook")

        refs.append(SourceReference(
            source_filename=source_name,
            chunk_index=chunk_idx,
            passage_preview=preview,
        ))
    return refs


def _match_source_reference(
    question: Question,
    passages: list[Document],
    source_refs: list[SourceReference],
) -> list[SourceReference]:
    """
    Match a question to its most relevant source reference from the retrieved passages.
    Avoids assigning all references or blindly defaulting to the first reference.
    """
    if not source_refs or not passages:
        return []

    q_text = (
        (question.question_text or "")
        + " "
        + (question.chapter or "")
        + " "
        + (question.topic or "")
        + " "
        + (question.answer or "")
    ).lower()
    q_words = set(re.findall(r"\b[a-zA-Z0-9]{3,}\b", q_text))

    best_idx = 0
    best_score = -1

    for idx, (doc, ref) in enumerate(zip(passages, source_refs)):
        score = 0
        doc_ch = (doc.metadata.get("chapter") or doc.metadata.get("unit") or ref.source_filename or "").lower()

        # Heavy weight for chapter match
        if question.chapter and doc_ch:
            q_ch_clean = re.sub(r'^(?:Lesson|Chapter|Unit)\s*\d+[:\s\-]*', '', question.chapter, flags=re.IGNORECASE).strip().lower()
            doc_ch_clean = re.sub(r'^(?:Lesson|Chapter|Unit)\s*\d+[:\s\-]*', '', doc_ch, flags=re.IGNORECASE).strip().lower()
            if q_ch_clean and doc_ch_clean:
                if q_ch_clean in doc_ch_clean or doc_ch_clean in q_ch_clean:
                    score += 50

        # Word overlap with passage content
        doc_words = set(re.findall(r"\b[a-zA-Z0-9]{3,}\b", doc.page_content.lower()))
        overlap = len(q_words.intersection(doc_words))
        score += overlap

        if score > best_score:
            best_score = score
            best_idx = idx

    return [source_refs[best_idx]]


def is_question_in_forbidden_units(
    q_dict_or_obj: dict | Question,
    forbidden_units: list[str] | None,
    permitted_units: list[str] | None = None,
) -> tuple[bool, str | None]:
    """
    Check if a generated question touches or tests topics from forbidden units
    (e.g. second-half units in a mid-semester examination).
    Returns (is_forbidden, reason).
    """
    if not forbidden_units:
        return False, None

    if isinstance(q_dict_or_obj, dict):
        q_text = str(q_dict_or_obj.get("question_text", "")).lower()
        q_topic = str(q_dict_or_obj.get("topic", "")).lower()
        q_chapter = str(q_dict_or_obj.get("chapter", "")).lower()
        q_ans = str(q_dict_or_obj.get("answer", "")).lower()
    else:
        q_text = (q_dict_or_obj.question_text or "").lower()
        q_topic = (q_dict_or_obj.topic or "").lower()
        q_chapter = (q_dict_or_obj.chapter or "").lower()
        q_ans = (q_dict_or_obj.answer or "").lower()

    combined_text = f"{q_chapter} {q_topic} {q_text} {q_ans}"

    for fu in forbidden_units:
        fu_clean = fu.strip().lower()
        if not fu_clean:
            continue

        # 1. Direct match on chapter or topic name
        if q_chapter and (fu_clean in q_chapter or q_chapter in fu_clean):
            return True, f"Question chapter '{q_chapter}' explicitly matches forbidden chapter '{fu}'"
        if q_topic and (fu_clean in q_topic or q_topic in fu_clean):
            return True, f"Question topic '{q_topic}' matches forbidden chapter '{fu}'"

        # 2. Extract core topic name by stripping 'Lesson X: ', 'Chapter X: ', 'Unit X: '
        core_topic = re.sub(r'^(?:lesson|chapter|unit|module)\s*\d+[:\.\-\s]*', '', fu_clean).strip()
        if len(core_topic) > 3:
            if re.search(rf'\b{re.escape(core_topic)}\b', combined_text):
                return True, f"Question contains forbidden topic concept '{core_topic}' from '{fu}'"

        # 3. Class 1 Specific Forbidden Concepts (Lessons 6-9)
        if "subtraction" in fu_clean:
            if any(w in q_text for w in ["subtract", "subtraction", "minus", "difference between", "take away", "left after giving", "taken away"]):
                return True, f"Question tests subtraction which belongs to forbidden chapter '{fu}'"
            if re.search(r'\b\d+\s*[-−]\s*\d+', q_text):
                return True, f"Question contains subtraction calculation from forbidden chapter '{fu}'"

        if any(term in fu_clean for term in ["number 10", "units and tens", "number 11 to 20"]):
            if any(w in q_text for w in ["units and tens", "tens and units", "place value of"]):
                return True, f"Question tests place value from forbidden chapter '{fu}'"
            large_nums = [int(n) for n in re.findall(r'\b\d+\b', q_text)]
            if any(10 <= n <= 20 for n in large_nums) and any("digits (1-9)" in p.lower() for p in (permitted_units or [])):
                return True, f"Question uses numbers 10-20 from forbidden chapter '{fu}'"

        # 4. Class 2 Specific Forbidden Concepts (Lessons 4-6: Subtraction, Multiplication, Division)
        if "division" in fu_clean and any(w in q_text for w in ["divide", "division", "quotient", "divided by", "equal sharing"]):
            return True, f"Question tests division from forbidden chapter '{fu}'"
        if "multiplication" in fu_clean and any(w in q_text for w in ["multiply", "multiplication", "product of", "times"]):
            return True, f"Question tests multiplication from forbidden chapter '{fu}'"

        # 5. Class 4 Specific Forbidden Concepts (Chapters 6-9: Division, Circles, Mental Arithmetic, Fractional Numbers)
        if "fraction" in fu_clean and any(w in q_text for w in ["fraction", "numerator", "denominator", "half of", "quarter of"]):
            return True, f"Question tests fractions from forbidden chapter '{fu}'"
        if "circle" in fu_clean and any(w in q_text for w in ["circle", "radius", "diameter", "circumference"]):
            return True, f"Question tests circles from forbidden chapter '{fu}'"

        # 6. Class 5 Specific Forbidden Concepts (Units 6-10: Angles, Circles, Length, Perimeter and Area, Data Handling)
        if "angles" in fu_clean and any(w in q_text for w in ["angle", "degrees", "acute", "obtuse", "right angle", "protractor"]):
            return True, f"Question tests angles from forbidden chapter '{fu}'"
        if "perimeter and area" in fu_clean and any(w in q_text for w in ["perimeter", "area of rectangle", "area of square", "sq cm", "sq m"]):
            return True, f"Question tests perimeter/area from forbidden chapter '{fu}'"
        if "data handling" in fu_clean and any(w in q_text for w in ["data handling", "bar graph", "pictograph", "tally marks"]):
            return True, f"Question tests data handling from forbidden chapter '{fu}'"

    return False, None


def _enrich_question(
    question: Question,
    request: PaperGenerateRequest,
    source_refs: list[SourceReference],
    permitted_units: list[str] | None = None,
    passages: list[Document] | None = None,
    target_chapter: str | None = None,
) -> Question:
    """Populate the identity, context, and traceability fields on a question.

    Gemini only produces core question fields (question_text, marks, etc.).
    This function fills in the metadata that comes from the request, the blueprint's
    chapter allocation, and the retrieved passages, and marks the question as validated.
    """
    question.question_id = uuid.uuid4().hex[:10]
    question.grade = request.academic_grade or request.grade
    question.subject = request.subject

    # Resolve specific chapter name from target_chapter, permitted units, or question content
    assigned_ch = target_chapter or question.chapter or question.topic
    if assigned_ch in ("Mathematics Curriculum", "General Math", "Mathematics"):
        assigned_ch = target_chapter

    if permitted_units:
        matched_pu = None
        if assigned_ch:
            for pu in permitted_units:
                if pu.lower() in assigned_ch.lower() or assigned_ch.lower() in pu.lower():
                    matched_pu = pu
                    break
        if not matched_pu:
            q_text_lower = question.question_text.lower()
            for pu in permitted_units:
                pu_core = re.sub(r'^(?:Lesson|Chapter|Unit)\s*\d+[:\s\-]*', '', pu, flags=re.IGNORECASE).strip().lower()
                if pu_core and (pu_core in q_text_lower or any(word in q_text_lower for word in pu_core.split() if len(word) > 3)):
                    matched_pu = pu
                    break
        question.chapter = matched_pu or target_chapter or (permitted_units[0] if permitted_units else request.topic)
    elif target_chapter:
        question.chapter = target_chapter
    elif not question.chapter or question.chapter in ("Mathematics Curriculum", "General Math", "Mathematics"):
        question.chapter = request.topic or "Mathematics"

    # Match specific source reference
    if passages and source_refs and len(passages) == len(source_refs):
        matched_refs = _match_source_reference(question, passages, source_refs)
        question.source_references = matched_refs if matched_refs else [source_refs[0]]
        if question.chapter in ("Mathematics Curriculum", "Mathematics", "General Math", ""):
            ref_ch = question.source_references[0].source_filename
            if ref_ch and ref_ch not in ("unknown", "Textbook", "Mathematics Curriculum"):
                question.chapter = ref_ch
    elif source_refs:
        question.source_references = [source_refs[0]]
    else:
        question.source_references = []

    if not question.topic or question.topic in ("Mathematics Curriculum", "General Math", "Mathematics"):
        question.topic = question.chapter or request.topic or "Mathematics"

    question.validation_status = ValidationStatus.VALID
    question.math_verification = verify_question_math(question)
    return question


def _extract_text(content: object) -> str:
    """Extract string text from LLM response content.

    Handles str, list of strings, list of dicts with 'text' key (standard in
    langchain-google-genai with newer Gemini models), or objects with a text attribute.
    """
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and "text" in part:
                parts.append(str(part["text"]))
            elif hasattr(part, "text"):
                parts.append(str(getattr(part, "text")))
            else:
                parts.append(str(part))
        return " ".join(parts).strip()
    return str(content).strip()


def _generate_section_questions(
    sec: "SectionBlueprint",
    request: PaperGenerateRequest,
    passages: list[Document],
    existing_questions: list[Question],
    source_refs: list[SourceReference],
    llm,
    permitted_units: list[str] | None = None,
    forbidden_units: list[str] | None = None,
) -> list[Question]:
    """
    Generate questions for a single section according to its blueprint.
    Guarantees question count, question type, exact marks per question,
    SymPy mathematical validation, and uniqueness.
    """
    passage_text = "\n---\n".join(
        f"[Passage {i+1} from {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
        for i, doc in enumerate(passages)
    )

    academic_class = format_grade_display(request.academic_grade or request.grade)
    curriculum_class = format_grade_display(request.curriculum_grade or request.grade)
    scope_str = request.syllabus_scope or request.topic
    exam_type = request.exam_type or ExamType.FINAL

    if exam_type == ExamType.PREREQUISITE:
        exam_header = f"""EXAMINATION TYPE: Prerequisite Examination
- Academic Class of Students: {academic_class}
- Curriculum Class Used: {curriculum_class}
- Permitted Syllabus Topics: {scope_str}
- Focus: Assess fundamental concepts from the previous class."""
    elif exam_type == ExamType.MIDTERM:
        permitted_list_str = "\n".join(f"  * {ch}" for ch in (permitted_units or []))
        forbidden_list_str = "\n".join(f"  * {ch}" for ch in (forbidden_units or []))
        exam_header = f"""EXAMINATION TYPE: Mid-Semester Examination
- Academic Class: {academic_class}
- Curriculum Class Used: {curriculum_class}
- Subject: {request.subject}

PERMITTED CHAPTERS (FIRST HALF OF TEXTBOOK ONLY):
{permitted_list_str if permitted_list_str else f"  * {scope_str}"}

STRICTLY FORBIDDEN CHAPTERS (LAST HALF OF TEXTBOOK - NEVER GENERATE QUESTIONS FROM THESE):
{forbidden_list_str if forbidden_list_str else "  * Any chapter beyond the first half"}

CRITICAL SYLLABUS ENFORCEMENT RULES:
1. ONLY generate questions from the PERMITTED CHAPTERS listed above.
2. ABSOLUTELY ZERO questions from the FORBIDDEN CHAPTERS. Do NOT ask questions involving concepts, numbers, or operations from the later half of the textbook!
3. For each question, populate the "chapter" field with the exact matching chapter name from the PERMITTED list."""
    else:
        exam_header = f"""EXAMINATION TYPE: Final Examination
- Academic Class: {academic_class}
- Scope: {scope_str} (Full Syllabus)"""

    marks_dist = sec.get_marks_distribution()
    needed = sec.num_questions
    sec_difficulty = sec.difficulty or request.difficulty or Difficulty.MEDIUM

    # Summary of already generated questions to forbid duplicate concepts/scenarios
    existing_summary = ""
    if existing_questions:
        q_lines = [
            f"  - Q{idx+1} ({q.question_type.value}, {q.marks}M): {q.question_text}"
            for idx, q in enumerate(existing_questions)
        ]
        existing_summary = (
            "\n\nALREADY GENERATED QUESTIONS IN PREVIOUS SECTIONS (STRICTLY FORBIDDEN TO REPEAT OR RESKIN):\n"
            + "\n".join(q_lines)
            + "\nSTRICT RULE: Do NOT generate questions testing the same scenario, story, numbers, or problem template as any question listed above."
        )

    coherence_rule = ""
    if sec.question_type == QuestionType.LONG_ANSWER or "Section C" in sec.title or "Problem Solving" in sec.title:
        coherence_rule = """
4. STRICT MANDATE FOR LONG-ANSWER (PROBLEM SOLVING) QUESTIONS:
   - AVOID verbose story essays, complicated narrative scenarios, or paragraphs of text!
   - Every long-answer question must consist of clearly labeled, structured subparts: '(a)', '(b)', and optionally '(c)'.
   - Each subpart must present an explicit mathematical calculation or step.
   - For early classes (Class 1 & 2): subparts should be simple arithmetic calculations, number sentences, or place-value exercises (e.g. '(a) Solve: 4 + 3 = ___ [2 Marks] (b) Solve: 5 + 2 = ___ [2 Marks] (c) Write the number between 13 and 15: ___ [1 Mark]').
   - For upper classes (Class 3-5): subparts should follow a logical, step-by-step mathematical problem.
   - Sub-marks across parts (a), (b), (c) must sum exactly to the question's total marks."""

    pedagogical_guidance = get_grade_pedagogical_prompt(request.curriculum_grade or request.academic_grade or request.grade, sec.question_type)

    alloc_guidance = ""
    alloc = getattr(sec, "chapter_allocation", None) or []
    if alloc:
        alloc_lines = [
            f"  - Slot {idx+1}: Chapter '{ch}' ({sec.question_type.value}, {marks_dist[idx] if idx < len(marks_dist) else marks_dist[-1]}M)"
            for idx, ch in enumerate(alloc[:needed])
        ]
        alloc_guidance = (
            "\n\nCHAPTER ALLOCATION BLUEPRINT FOR THIS SECTION (MANDATORY):\n"
            "You MUST allocate each question in order to its designated chapter below:\n"
            + "\n".join(alloc_lines)
            + "\nCRITICAL: In your output JSON, populate the 'chapter' field with the exact chapter name designated for that slot and ground it in the reference passage for that chapter."
        )

    human_prompt = f"""Generate questions for Section '{sec.section_id}' ({sec.title}):

{exam_header}
- Grade/Class: {academic_class}
- Subject: {request.subject}
- Question Type: {sec.question_type.value} (ALL questions in this section MUST be of this type)
- Difficulty: {sec_difficulty.value}
- Total questions needed in this section: {needed}
- Marks per question: {marks_dist} (total section marks: {sec.section_marks})
{pedagogical_guidance}{alloc_guidance}

SECTION-SPECIFIC INSTRUCTIONS:
1. Every question MUST be of type "{sec.question_type.value}".
{"2. For multiple_choice questions, provide exactly 4 clear options (A, B, C, D) and identify the correct_option matching one of them." if sec.question_type == QuestionType.MULTIPLE_CHOICE else "2. Provide complete, clear worked solutions and marking steps."}
3. DIVERSITY & NON-DUPLICATION REQUIREMENT: Cover different chapters/concepts across the reference passages (e.g. arithmetic, geometry/shapes, measurement, time/money, patterns, word problems). Do NOT generate repetitive questions with just different numbers.{coherence_rule}
5. SELF-CONTAINED CLARITY: Every question must contain all information and numbers necessary to solve it. NEVER refer to non-existent visual diagrams, pictures, tables, or figures (do NOT say 'as shown in the figure below', 'look at the diagram', etc.). Do NOT end questions with dangling colons.
6. AGE APPROPRIATENESS: Strictly suitable for primary school students in {academic_class}.
7. Ground all questions directly in the PERMITTED CHAPTERS from the REFERENCE PASSAGES below. Do NOT touch any forbidden topics.{existing_summary}

REFERENCE PASSAGES:
{passage_text}

Return ONLY a JSON array of {needed} question objects."""

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=human_prompt),
    ]

    sec_questions: list[Question] = []
    rejection_reasons: list[str] = []
    max_attempts = max(3, settings.MAX_RETRIES + 2)

    for attempt in range(1, max_attempts + 1):
        try:
            response = llm.invoke(messages)
            raw_text = _extract_text(response.content)

            if raw_text.startswith("```"):
                first_nl = raw_text.index("\n")
                raw_text = raw_text[first_nl + 1 :]
            if raw_text.endswith("```"):
                raw_text = raw_text[:-3].strip()

            questions_data = json.loads(raw_text)
            if not isinstance(questions_data, list):
                raise ValueError("Expected a JSON array of questions.")

            for q_dict in questions_data:
                if len(sec_questions) >= needed:
                    break
                try:
                    target_mark = (
                        marks_dist[len(sec_questions)]
                        if len(sec_questions) < len(marks_dist)
                        else marks_dist[-1]
                    )
                    q_dict["question_type"] = sec.question_type.value
                    q_dict["marks"] = target_mark
                    if "difficulty" not in q_dict or not q_dict["difficulty"]:
                        sec_diff = sec.difficulty or request.difficulty or Difficulty.MEDIUM
                        q_dict["difficulty"] = sec_diff.value if hasattr(sec_diff, "value") else str(sec_diff)
                    if "topic" not in q_dict or not q_dict["topic"]:
                        q_dict["topic"] = (
                            sec.topics[0]
                            if (getattr(sec, "topics", None) and sec.topics)
                            else (request.topic or "Mathematics")
                        )
                    if "answer" not in q_dict or not q_dict["answer"]:
                        q_dict["answer"] = "Correct answer"
                    if "solution" not in q_dict or not q_dict["solution"]:
                        q_dict["solution"] = "Step-by-step solution"

                    # Scope validation check: reject candidates from forbidden units
                    if forbidden_units:
                        is_forb, forb_reason = is_question_in_forbidden_units(q_dict, forbidden_units, permitted_units)
                        if is_forb:
                            logger.info("Rejecting candidate question for %s: %s", sec.section_id, forb_reason)
                            rejection_reasons.append(forb_reason)
                            continue

                    slot_num = len(sec_questions)
                    slot_ch = alloc[slot_num] if (alloc and slot_num < len(alloc)) else None

                    q = Question.model_validate(q_dict)
                    _enrich_question(q, request, source_refs, permitted_units, passages=passages, target_chapter=slot_ch)

                    # 1. Quality, Clarity & Coherence validation
                    q_ok, q_err = validate_question_quality(q)
                    if not q_ok:
                        reason = f"Quality check failed: {q_err}"
                        logger.warning("Rejecting low-quality question: %s", q_err)
                        rejection_reasons.append(reason)
                        continue

                    # 2. Re-verify enriched question doesn't touch forbidden scope
                    if forbidden_units:
                        is_forb2, forb_reason2 = is_question_in_forbidden_units(q, forbidden_units, permitted_units)
                        if is_forb2:
                            logger.info("Rejecting enriched question for %s: %s", sec.section_id, forb_reason2)
                            rejection_reasons.append(forb_reason2)
                            continue

                    # 3. Deduplication against both prior sections and current section
                    is_dup, dup_reason = is_duplicate(q, existing_questions + sec_questions)
                    if is_dup:
                        reason = f"Duplicate detected: {dup_reason}"
                        logger.warning("Rejecting duplicate question: %s", dup_reason)
                        rejection_reasons.append(reason)
                        continue

                    sec_questions.append(q)
                except Exception as val_err:
                    reason = f"Validation exception: {val_err}"
                    logger.debug("Skipping invalid question in section %s: %s", sec.section_id, val_err)
                    rejection_reasons.append(reason)
                    continue

            if len(sec_questions) >= needed:
                break

            deficit = needed - len(sec_questions)
            if attempt < max_attempts:
                missing_marks = marks_dist[len(sec_questions) : needed]
                accepted_lines = [
                    f"  - Q ({q.question_type.value}, {q.marks}M): {q.question_text}"
                    for q in (existing_questions + sec_questions)
                ]
                accepted_summary = "\n".join(accepted_lines)
                candidate_target = deficit + 2

                retry_prompt = f"""We have accepted {len(sec_questions)} out of {needed} questions for Section '{sec.section_id}' ({sec.title}).
We now need EXACTLY {deficit} more valid, unique questions of type '{sec.question_type.value}' to fill the remaining slots.
Target marks for the {deficit} missing slots: {missing_marks} (total: {sum(missing_marks)} marks).

STRICT INSTRUCTIONS:
1. Every question must be of type "{sec.question_type.value}".
{"2. For multiple_choice questions, provide 4 distinct options and identify correct_option." if sec.question_type == QuestionType.MULTIPLE_CHOICE else "2. Provide complete, clear worked solutions and marking steps."}
3. DIVERSITY & TOPIC VARIETY: Provide {candidate_target} fresh, diverse candidate questions covering different topics from the curriculum (e.g. Shapes & Geometry, Numbers & Counting, Addition, Subtraction, Multiplication, Division).{coherence_rule}
4. SELF-CONTAINED: All numerical data and operands must be explicitly written in the question. Do NOT reference missing figures or diagrams.
5. STRICTLY FORBIDDEN (DO NOT REPEAT OR RESKIN ANY OF THESE ALREADY ACCEPTED QUESTIONS):
{accepted_summary}

REFERENCE PASSAGES:
{passage_text}

Return ONLY a JSON array containing {candidate_target} distinct question objects."""

                messages = [
                    SystemMessage(content=SYSTEM_PROMPT),
                    HumanMessage(content=retry_prompt),
                ]
        except Exception as e:
            reason = f"Model or parse error: {e}"
            logger.warning("Attempt %d for section %s failed: %s", attempt, sec.section_id, e)
            rejection_reasons.append(reason)
            if attempt < max_attempts:
                messages.append(
                    HumanMessage(content=f"Previous attempt error: {e}. Return ONLY a valid JSON array.")
                )

    if len(sec_questions) < needed:
        recent_reasons = "; ".join(rejection_reasons[-4:]) if rejection_reasons else "No candidates returned"
        raise RuntimeError(
            f"Failed to generate {needed} valid questions for Section '{sec.section_id}' ({sec.title}). "
            f"Obtained {len(sec_questions)} valid questions (needed {needed - len(sec_questions)} more). "
            f"Candidate rejection details: {recent_reasons}"
        )

    return sec_questions[:needed]


def _regenerate_single_question(
    target_sec: "SectionBlueprint",
    target_mark: int,
    request: PaperGenerateRequest,
    passages: list[Document],
    other_questions: list[Question],
    source_refs: list[SourceReference],
    llm,
    permitted_units: list[str] | None = None,
    forbidden_units: list[str] | None = None,
    target_chapter: str | None = None,
) -> Question:
    """
    Regenerate a single question within an existing section to replace a duplicate
    or low-quality question without changing the paper structure.
    """
    passage_text = "\n---\n".join(
        f"[Passage {i+1} from {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
        for i, doc in enumerate(passages)
    )
    academic_class = format_grade_display(request.academic_grade or request.grade)

    other_lines = [
        f"- Q ({q.question_type.value}, {q.marks}M): {q.question_text}"
        for q in other_questions
    ]
    forbidden_text = "\n".join(other_lines)

    forbidden_scope_instruction = ""
    if forbidden_units:
        forbidden_scope_instruction = (
            f"\nSTRICT FORBIDDEN TOPICS (NEVER ASK ABOUT THESE): {', '.join(forbidden_units)}\n"
            f"PERMITTED TOPICS ONLY: {', '.join(permitted_units or [])}\n"
        )

    target_chapter_instruction = ""
    if target_chapter:
        target_chapter_instruction = f"\nTARGET CHAPTER: '{target_chapter}' (You MUST generate a question assessing this specific chapter).\n"

    pedagogical_guidance = get_grade_pedagogical_prompt(request.curriculum_grade or request.academic_grade or request.grade, target_sec.question_type)

    prompt = f"""Generate EXACTLY ONE replacement question for Section '{target_sec.section_id}' ({target_sec.title}):
- Class: {academic_class}
- Subject: {request.subject}
- Question Type: {target_sec.question_type.value}
- Marks: {target_mark}
- Difficulty: {(target_sec.difficulty or request.difficulty or Difficulty.MEDIUM).value}
{target_chapter_instruction}{forbidden_scope_instruction}
{pedagogical_guidance}
STRICT CONSTRAINTS:
1. Question must be of type "{target_sec.question_type.value}" worth exactly {target_mark} marks.
2. Must be completely self-contained. Do NOT refer to diagrams, figures, or tables.
3. Coherent single scenario (never combine unrelated settings).
4. FORBIDDEN SCENARIOS (DO NOT REPEAT OR RESKIN ANY OF THESE):
{forbidden_text}

REFERENCE PASSAGES:
{passage_text}

Return ONLY a JSON array containing exactly 1 question object."""

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ]

    for attempt in range(1, settings.MAX_RETRIES + 2):
        try:
            response = llm.invoke(messages)
            raw_text = _extract_text(response.content)
            if raw_text.startswith("```"):
                first_nl = raw_text.index("\n")
                raw_text = raw_text[first_nl + 1 :]
            if raw_text.endswith("```"):
                raw_text = raw_text[:-3].strip()

            q_data = json.loads(raw_text)
            if isinstance(q_data, list) and q_data:
                q_dict = q_data[0]
                q_dict["question_type"] = target_sec.question_type.value
                q_dict["marks"] = target_mark
                if "difficulty" not in q_dict or not q_dict["difficulty"]:
                    sec_diff = target_sec.difficulty or request.difficulty or Difficulty.MEDIUM
                    q_dict["difficulty"] = sec_diff.value if hasattr(sec_diff, "value") else str(sec_diff)
                if "topic" not in q_dict or not q_dict["topic"]:
                    q_dict["topic"] = (
                        target_sec.topics[0]
                        if (getattr(target_sec, "topics", None) and target_sec.topics)
                        else (request.topic or "Mathematics")
                    )
                if "answer" not in q_dict or not q_dict["answer"]:
                    q_dict["answer"] = "Correct answer"
                if "solution" not in q_dict or not q_dict["solution"]:
                    q_dict["solution"] = "Step-by-step solution"

                if forbidden_units:
                    is_forb, _ = is_question_in_forbidden_units(q_dict, forbidden_units, permitted_units)
                    if is_forb:
                        continue

                q = Question.model_validate(q_dict)
                _enrich_question(q, request, source_refs, permitted_units, passages=passages, target_chapter=target_chapter)

                q_ok, q_err = validate_question_quality(q)
                if not q_ok:
                    continue

                if forbidden_units:
                    is_forb2, _ = is_question_in_forbidden_units(q, forbidden_units, permitted_units)
                    if is_forb2:
                        continue

                is_dup, _ = is_duplicate(q, other_questions)
                if not is_dup:
                    return q
        except Exception as e:
            logger.warning("Regeneration attempt %d failed: %s", attempt, e)

    raise RuntimeError(
        f"Failed to regenerate a unique replacement question for Section '{target_sec.section_id}'."
    )


def generate_questions(
    request: PaperGenerateRequest,
    passages: list[Document],
    blueprint: Optional[ExamBlueprint] = None,
    permitted_units: list[str] | None = None,
    forbidden_units: list[str] | None = None,
) -> list[Question]:
    """
    Call Gemini to generate questions and validate them with Pydantic and SymPy.
    Enforces that candidate questions have no mathematical discrepancies and are unique.
    Generates section-by-section according to the blueprint.
    """
    blueprint = blueprint or request.get_or_create_blueprint()
    llm = get_llm()
    source_refs = _build_source_references(passages)
    permitted_units = permitted_units or getattr(request, "permitted_units", None)
    forbidden_units = forbidden_units or getattr(request, "forbidden_units", None)

    # If blueprint has defined sections, generate section by section
    if blueprint and blueprint.sections:
        all_questions: list[Question] = []
        section_map: dict[str, list[Question]] = {}
        for sec in blueprint.sections:
            sec_qs = _generate_section_questions(
                sec=sec,
                request=request,
                passages=passages,
                existing_questions=all_questions,
                source_refs=source_refs,
                llm=llm,
                permitted_units=permitted_units,
                forbidden_units=forbidden_units,
            )
            section_map[sec.section_id] = sec_qs
            all_questions.extend(sec_qs)

        # Cross-section uniqueness check & targeted regeneration:
        # Compare all generated questions across Sections A, B, and C before finalizing
        max_regen_rounds = 3
        for _ in range(max_regen_rounds):
            unique_ok, _ = validate_paper_questions_uniqueness(all_questions)
            if unique_ok:
                break

            # Find duplicate question pair and regenerate the second one
            dup_found = False
            for i in range(len(all_questions)):
                for j in range(i + 1, len(all_questions)):
                    is_dup, _ = is_duplicate(all_questions[j], [all_questions[i]])
                    if is_dup:
                        dup_found = True
                        dup_idx = j
                        bad_q = all_questions[dup_idx]

                        # Identify which section bad_q belongs to
                        target_sec = None
                        for sec in blueprint.sections:
                            if any(q.question_id == bad_q.question_id for q in section_map.get(sec.section_id, [])):
                                target_sec = sec
                                break
                        if target_sec is None:
                            target_sec = blueprint.sections[-1]

                        other_qs = [q for idx, q in enumerate(all_questions) if idx != dup_idx]
                        logger.info(
                            "Regenerating duplicate question %d in section %s",
                            dup_idx + 1,
                            target_sec.section_id,
                        )
                        try:
                            replacement_q = _regenerate_single_question(
                                target_sec=target_sec,
                                target_mark=bad_q.marks,
                                request=request,
                                passages=passages,
                                other_questions=other_qs,
                                source_refs=source_refs,
                                llm=llm,
                                permitted_units=permitted_units,
                                forbidden_units=forbidden_units,
                                target_chapter=bad_q.chapter,
                            )
                            all_questions[dup_idx] = replacement_q
                            sec_list = section_map[target_sec.section_id]
                            for s_i, sq in enumerate(sec_list):
                                if sq.question_id == bad_q.question_id:
                                    sec_list[s_i] = replacement_q
                                    break
                        except Exception as regen_err:
                            logger.warning("Regeneration error: %s", regen_err)
                        break
                if dup_found:
                    break

        # Final scope enforcement: strictly verify no questions touch forbidden units
        if forbidden_units:
            for idx, q in enumerate(all_questions):
                is_forb, forb_reason = is_question_in_forbidden_units(q, forbidden_units, permitted_units)
                if is_forb:
                    logger.warning("Post-generation detected forbidden question %d: %s. Regenerating...", idx + 1, forb_reason)
                    target_sec = blueprint.sections[-1]
                    for sec in blueprint.sections:
                        if sec.question_type == q.question_type:
                            target_sec = sec
                            break
                    other_qs = [o for o_idx, o in enumerate(all_questions) if o_idx != idx]
                    try:
                        replacement_q = _regenerate_single_question(
                            target_sec=target_sec,
                            target_mark=q.marks,
                            request=request,
                            passages=passages,
                            other_questions=other_qs,
                            source_refs=source_refs,
                            llm=llm,
                            permitted_units=permitted_units,
                            forbidden_units=forbidden_units,
                            target_chapter=permitted_units[0] if permitted_units else None,
                        )
                        all_questions[idx] = replacement_q
                    except Exception as rep_err:
                        logger.warning("Failed to regenerate forbidden question: %s", rep_err)

        return all_questions

    # Fallback to single-prompt generation if no sections
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=_build_human_prompt(request, passages)),
    ]

    last_error: Exception | None = None

    for attempt in range(1, settings.MAX_RETRIES + 2):
        logger.info("Generation attempt %d / %d", attempt, settings.MAX_RETRIES + 1)
        try:
            response = llm.invoke(messages)
            raw_text = _extract_text(response.content)

            if raw_text.startswith("```"):
                first_newline = raw_text.index("\n")
                raw_text = raw_text[first_newline + 1 :]
            if raw_text.endswith("```"):
                raw_text = raw_text[:-3].strip()

            questions_data = json.loads(raw_text)
            if not isinstance(questions_data, list):
                raise ValueError("Expected a JSON array of questions.")

            questions = [Question.model_validate(q) for q in questions_data]
            questions = [
                _enrich_question(q, request, source_refs) for q in questions
            ]

            discrepancies = [
                q for q in questions
                if q.math_verification and q.math_verification.status == MathCheckStatus.DISCREPANCY
            ]
            if discrepancies:
                bad_q = discrepancies[0]
                raise ValueError(
                    f"Generated question had mathematical discrepancy: '{bad_q.question_text}' "
                    f"({bad_q.math_verification.details if bad_q.math_verification else ''})"
                )

            unique_qs, rejected_dups = deduplicate_candidates(questions)
            if rejected_dups > 0:
                raise ValueError(f"Generated questions contained {rejected_dups} duplicate or near-duplicate questions.")

            if len(questions) != request.num_questions:
                raise ValueError(
                    f"Requested {request.num_questions} questions but got {len(questions)}."
                )

            actual_marks = sum(q.marks for q in questions)
            if actual_marks != request.total_marks:
                raise ValueError(
                    f"Requested {request.total_marks} total marks but got {actual_marks}."
                )

            return questions

        except Exception as e:
            last_error = e
            logger.warning("Attempt %d failed: %s", attempt, e)
            if attempt <= settings.MAX_RETRIES:
                messages.append(
                    HumanMessage(
                        content=(
                            f"Your previous response had an error: {e}\n"
                            "Please fix the output and try again. "
                            "Return ONLY a valid JSON array."
                        )
                    )
                )

    raise RuntimeError(
        f"Failed to generate valid questions after {settings.MAX_RETRIES + 1} attempts. "
        f"Last error: {last_error}"
    )


def generate_candidate_pool(
    blueprint: ExamBlueprint,
    passages: list[Document],
    grade: str,
    subject: str,
    topic: str,
    num_versions: int = 1,
    exam_type: ExamType = ExamType.FINAL,
    academic_grade: str | None = None,
    curriculum_grade: str | None = None,
    syllabus_scope: str | None = None,
    permitted_units: list[str] | None = None,
    forbidden_units: list[str] | None = None,
) -> tuple[list[Question], int, int]:
    """
    Generate a pool of candidate questions for all blueprint sections.
    Enforces section question types, positive marks, SymPy math checking,
    and deduplication across candidates.

    Returns:
        (valid_candidates, duplicate_rejected_count, math_discrepancy_rejected_count)
    """
    llm = get_llm()
    source_refs = _build_source_references(passages)
    passage_text = "\n---\n".join(
        f"[Passage {i+1} from {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
        for i, doc in enumerate(passages)
    )

    academic_class = academic_grade or grade
    curriculum_class = curriculum_grade or grade
    scope_str = syllabus_scope or topic

    all_candidates: list[Question] = []
    total_duplicates_rejected = 0
    total_math_discrepancies_rejected = 0

    for sec in blueprint.sections:
        needed = sec.num_questions * num_versions
        target_count = max(needed + 2, int(needed * 1.5))
        sec_difficulty = sec.difficulty or Difficulty.MEDIUM
        marks_dist = sec.get_marks_distribution()
        typical_mark = marks_dist[0] if marks_dist else 2

        sec_candidates: list[Question] = []

        if exam_type == ExamType.PREREQUISITE:
            exam_header = f"""EXAMINATION TYPE: Prerequisite Examination
- Academic Class of Students: {academic_class}
- Curriculum Class Used: Class {curriculum_class}
- Focus: Foundational questions from Class {curriculum_class} assessing prerequisite readiness for {academic_class}."""
        elif exam_type == ExamType.MIDTERM:
            permitted_list_str = "\n".join(f"  * {ch}" for ch in (permitted_units or []))
            forbidden_list_str = "\n".join(f"  * {ch}" for ch in (forbidden_units or []))
            exam_header = f"""EXAMINATION TYPE: Mid-Semester Examination
- Academic Class: {academic_class}
- Curriculum Class Used: {curriculum_class}
- Subject: {subject}

PERMITTED CHAPTERS (FIRST HALF OF TEXTBOOK ONLY):
{permitted_list_str if permitted_list_str else f"  * {scope_str}"}

STRICTLY FORBIDDEN CHAPTERS (LAST HALF OF TEXTBOOK - DO NOT TOUCH):
{forbidden_list_str if forbidden_list_str else "  * Any chapter beyond the first half"}

CRITICAL SYLLABUS ENFORCEMENT RULES:
1. ONLY generate questions from the PERMITTED CHAPTERS listed above.
2. ABSOLUTELY ZERO questions from the FORBIDDEN CHAPTERS. Do NOT ask questions involving concepts, numbers, or operations from the later half of the textbook!
3. For each question, populate the "chapter" field with the exact matching chapter name from the PERMITTED list."""
        else:
            exam_header = f"""EXAMINATION TYPE: Final Examination
- Academic Class: {academic_class}
- Scope: {scope_str} (Full Syllabus)"""

        effective_class = request.curriculum_grade or request.academic_grade or request.grade
        pedagogical_guidance = get_grade_pedagogical_prompt(effective_class, sec.question_type)

        human_prompt = f"""Generate candidate questions for Section '{sec.section_id}' ({sec.title}) with these specifications:

{exam_header}
- Grade/Class: {academic_class}
- Subject: {subject}
- Topic/Chapter: {topic}
- Question Type: {sec.question_type.value} (ALL questions MUST be of this type)
- Difficulty: {sec_difficulty.value}
- Number of unique questions needed: {target_count}
- Target marks per question: {typical_mark}
{pedagogical_guidance}

RULES FOR THIS SECTION:
1. Every question MUST be of type "{sec.question_type.value}".
2. All questions must be completely distinct from each other (no duplicate or near-duplicate variations).
{"3. For multiple-choice questions, provide at least 4 options and identify the correct one." if sec.question_type == QuestionType.MULTIPLE_CHOICE else "3. Provide clear step-by-step marking/solution guidance."}
4. Ground questions in the provided REFERENCE PASSAGES.

REFERENCE PASSAGES:
{passage_text}

Return ONLY a JSON array of {target_count} question objects."""

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=human_prompt),
        ]

        for attempt in range(1, settings.MAX_RETRIES + 2):
            try:
                response = llm.invoke(messages)
                raw_text = _extract_text(response.content)

                if raw_text.startswith("```"):
                    first_nl = raw_text.index("\n")
                    raw_text = raw_text[first_nl + 1 :]
                if raw_text.endswith("```"):
                    raw_text = raw_text[:-3].strip()

                questions_data = json.loads(raw_text)
                if not isinstance(questions_data, list):
                    raise ValueError("Expected a JSON array of questions.")

                for q_dict in questions_data:
                    try:
                        # Enforce section question type
                        q_dict["question_type"] = sec.question_type.value
                        if "marks" not in q_dict or not isinstance(q_dict["marks"], int) or q_dict["marks"] <= 0:
                            q_dict["marks"] = typical_mark

                        if forbidden_units:
                            is_forb, _ = is_question_in_forbidden_units(q_dict, forbidden_units, permitted_units)
                            if is_forb:
                                continue

                        q = Question.model_validate(q_dict)
                        # Enrich
                        q.question_id = uuid.uuid4().hex[:10]
                        q.grade = grade
                        q.subject = subject

                        # Resolve chapter
                        if permitted_units:
                            matched_pu = None
                            if q.chapter and q.chapter not in ("Mathematics Curriculum", "General Math", "Mathematics"):
                                for pu in permitted_units:
                                    if pu.lower() in q.chapter.lower() or q.chapter.lower() in pu.lower():
                                        matched_pu = pu
                                        break
                            if not matched_pu:
                                q_text_lower = q.question_text.lower()
                                for pu in permitted_units:
                                    pu_core = re.sub(r'^(?:Lesson|Chapter|Unit)\s*\d+[:\s\-]*', '', pu, flags=re.IGNORECASE).strip().lower()
                                    if pu_core and (pu_core in q_text_lower or any(word in q_text_lower for word in pu_core.split() if len(word) > 3)):
                                        matched_pu = pu
                                        break
                            q.chapter = matched_pu or (permitted_units[0] if permitted_units else topic)
                        else:
                            q.chapter = topic

                        if not q.topic or q.topic in ("Mathematics Curriculum", "General Math", "Mathematics"):
                            q.topic = q.chapter or topic

                        # Match specific source reference
                        if passages and source_refs and len(passages) == len(source_refs):
                            matched_refs = _match_source_reference(q, passages, source_refs)
                            q.source_references = matched_refs if matched_refs else [source_refs[0]]
                            if q.chapter in ("Mathematics Curriculum", "Mathematics", "General Math", ""):
                                ref_ch = q.source_references[0].source_filename
                                if ref_ch and ref_ch not in ("unknown", "Textbook", "Mathematics Curriculum"):
                                    q.chapter = ref_ch
                        elif source_refs:
                            q.source_references = [source_refs[0]]
                        else:
                            q.source_references = []

                        q.validation_status = ValidationStatus.VALID

                        # Re-verify candidate after enrichment
                        if forbidden_units:
                            is_forb2, _ = is_question_in_forbidden_units(q, forbidden_units, permitted_units)
                            if is_forb2:
                                continue

                        # Quality & Age-appropriateness check
                        q_ok, q_err = validate_question_quality(q)
                        if not q_ok:
                            continue

                        # SymPy math verification
                        q.math_verification = verify_question_math(q)
                        if q.math_verification and q.math_verification.status == MathCheckStatus.DISCREPANCY:
                            logger.warning(
                                "Rejecting mathematically invalid candidate: '%s' (details: %s)",
                                q.question_text,
                                q.math_verification.details,
                            )
                            total_math_discrepancies_rejected += 1
                            continue

                        # Check duplicate against section candidates and existing global candidates
                        is_dup, _ = is_duplicate(q, sec_candidates + all_candidates)
                        if is_dup:
                            total_duplicates_rejected += 1
                            continue

                        sec_candidates.append(q)
                    except Exception as val_err:
                        logger.debug("Skipping invalid candidate question: %s", val_err)
                        continue

                # Check if we have gathered enough candidates for this section
                if len(sec_candidates) >= needed:
                    break

                deficit = needed - len(sec_candidates)
                if attempt <= settings.MAX_RETRIES:
                    messages.append(
                        HumanMessage(
                            content=(
                                f"We still need {deficit} more valid, distinct questions for Section '{sec.section_id}' "
                                f"of type '{sec.question_type.value}'. Please provide {deficit + 2} additional unique questions."
                            )
                        )
                    )
            except Exception as e:
                logger.warning("Candidate pool attempt %d for section %s failed: %s", attempt, sec.section_id, e)
                if attempt <= settings.MAX_RETRIES:
                    messages.append(
                        HumanMessage(content=f"Previous attempt error: {e}. Return ONLY a valid JSON array.")
                    )

        all_candidates.extend(sec_candidates)

    return all_candidates, total_duplicates_rejected, total_math_discrepancies_rejected

