# tests/test_question_quality.py
# Comprehensive automated test suite verifying question quality, pedagogical standards,
# non-duplication, mark allocation, syllabus grounding, and PDF separation across all
# examination types (Prerequisite, Mid-Semester, Final) and Classes 1 to 5.

import pytest
from app.schemas import (
    Question,
    QuestionType,
    Difficulty,
    ExamType,
    Paper,
    PaperSection,
    ExamBlueprint,
    create_default_blueprint,
    parse_grade_number,
)
from app.curriculum import (
    CURRICULUM_CHAPTERS,
    get_chapters_for_class,
    get_midterm_chapters_for_class,
    assign_blueprint_chapter_coverage,
    is_computational_chapter,
    is_foundational_chapter,
)
from app.validator import (
    check_question_clarity,
    check_question_coherence,
    check_grade_age_appropriateness,
    validate_question_quality,
    validate_paper_questions_uniqueness,
    validate_full_paper,
)
from app.duplicates import is_duplicate, check_shared_phrase
from app.pdf_export import generate_pdf


# ---------------------------------------------------------------------------
# 1. Rejection of Textbook Meta-References
# ---------------------------------------------------------------------------
def test_textbook_meta_references_strictly_rejected():
    """Verify questions with textbook lesson numbers or citations fail clarity check."""
    bad_questions = [
        "Which of the flat two-dimensional shapes has four equal sides and is discussed in Lesson 2?",
        "What is the number name for the digit 5 as introduced in Lesson 3?",
        "If there are no monkeys dangling from the wire, what number represents this quantity according to Lesson 4?",
        "Consider the sequence of numbers from 11 to 20 as studied in class: find the number after 15.",
        "From Chapter 1, identify which object rolls and which object slides.",
        "As given in Lesson 5, calculate the sum of 4 and 3.",
    ]
    for q_text in bad_questions:
        ok, err = check_question_clarity(q_text, QuestionType.MULTIPLE_CHOICE, options=["A", "B", "C", "D"])
        assert not ok, f"Expected '{q_text}' to be rejected for textbook meta-reference."
        assert "textbook meta-reference" in err.lower()


def test_clean_mathematical_questions_pass_clarity():
    """Verify direct mathematical questions pass clarity check."""
    clean_questions = [
        ("Which of the following shapes has four equal sides?", ["Circle", "Triangle", "Square", "Rectangle"]),
        ("What is the number name for 5?", ["Three", "Four", "Five", "Six"]),
        ("If all 3 apples are eaten and none remain, what number represents the remaining apples?", ["1", "0", "2", "3"]),
        ("Find the sum of 4 and 3.", None),
    ]
    for q_text, opts in clean_questions:
        ok, err = check_question_clarity(q_text, QuestionType.MULTIPLE_CHOICE if opts else QuestionType.SHORT_ANSWER, options=opts)
        assert ok, f"Clean question was unexpectedly rejected: {err}"


# ---------------------------------------------------------------------------
# 2. Rejection of English Vocabulary and Definition Testing in Mathematics
# ---------------------------------------------------------------------------
def test_english_vocabulary_and_definitions_rejected():
    """Verify that English vocabulary recall and definition questions are rejected."""
    vocab_questions = [
        "Which spatial vocabulary term describes a bird sitting on the top of a tree?",
        "State the spatial relationship term used when an object is positioned higher than another.",
        "State the term used when an object is positioned directly beneath a table.",
        "What spatial vocabulary term describes an apple placed inside a basket?",
        "Define the term addition in your own words.",
        "Write a sentence explaining what the plus sign means.",
        "Give the definition of a rectangle.",
        "Based on the spatial understanding vocabulary, write an example.",
    ]
    for q_text in vocab_questions:
        q = Question(
            question_text=q_text,
            question_type=QuestionType.MULTIPLE_CHOICE,
            marks=1,
            difficulty=Difficulty.EASY,
            grade="1",
            topic="Spatial Understanding",
            options=["Top", "Under", "Inside", "Far"],
            correct_option="Top",
            answer="Top",
            solution="Top describes the position.",
        )
        ok, err = check_grade_age_appropriateness(q)
        assert not ok, f"Expected '{q_text}' to be rejected as vocabulary/definition test."
        assert "not" in err.lower()


# ---------------------------------------------------------------------------
# 3. Excessive Language Complexity and Word Count Limits Across Classes
# ---------------------------------------------------------------------------
def test_excessive_wordiness_rejected_for_primary_grades():
    """Verify that lengthy word problems and reading comprehension walls are rejected."""
    # Giant story for Class 1 (similar to marble counting problem from user PDF)
    giant_class_1_story = (
        "Ravi has a collection of marbles. He counts them and finds that every single object counted is one unit, "
        "and 10 units make a group of 1 ten. If Ravi has 1 group of ten and 4 individual units (single units), "
        "answer the following step-by-step: (a) How many total units are there in 1 group of ten? (b) Write the number "
        "of tens and the number of units in a standard place value table (TENS and UNITS). (c) What is the total number "
        "represented by 1 ten and 4 units combined?"
    )
    q1 = Question(
        question_text=giant_class_1_story,
        question_type=QuestionType.LONG_ANSWER,
        marks=5,
        difficulty=Difficulty.MEDIUM,
        grade="1",
        topic="Units and Tens",
        answer="(a) 10 (b) 1 ten 4 units (c) 14",
        solution="Step-by-step solution",
    )
    ok, err = check_grade_age_appropriateness(q1)
    assert not ok, "Giant word story for Class 1 should be rejected."
    assert "wordy" in err.lower() or "essay" in err.lower() or "verbose" in err.lower()


def test_appropriate_class_1_long_answer_accepted():
    """Structured, concise multi-part arithmetic problem for Class 1 passes."""
    concise_la = "Solve the following addition sums: (a) 3 + 2 = ___ (b) 4 + 3 = ___ (c) 5 + 1 = ___"
    q = Question(
        question_text=concise_la,
        question_type=QuestionType.LONG_ANSWER,
        marks=5,
        difficulty=Difficulty.EASY,
        grade="1",
        topic="Addition",
        answer="(a) 5 (b) 7 (c) 6",
        solution="(a) 3+2=5 [2 Marks] (b) 4+3=7 [2 Marks] (c) 5+1=6 [1 Mark]",
    )
    ok, err = check_grade_age_appropriateness(q)
    assert ok, f"Clean Class 1 Long Answer was unexpectedly rejected: {err}"


# ---------------------------------------------------------------------------
# 4. Long Answer Mark Allocation and Structural Depth
# ---------------------------------------------------------------------------
def test_trivial_one_step_fact_rejected_for_high_marks():
    """A trivial 1-step fact like '4 + 3 = ?' cannot be a 5-mark Long Answer question."""
    trivial_q = Question(
        question_text="What is 4 + 3?",
        question_type=QuestionType.LONG_ANSWER,
        marks=5,
        difficulty=Difficulty.EASY,
        grade="1",
        topic="Addition",
        answer="7",
        solution="4 + 3 = 7",
    )
    ok, err = validate_question_quality(trivial_q)
    assert not ok
    assert "depth" in err.lower() or "subparts" in err.lower()


def test_structured_high_mark_question_passes():
    """Structured problem with clearly separated subparts passes quality validation."""
    structured_q = Question(
        question_text=(
            "Answer the following number problems: "
            "(a) Write the number name for 15. "
            "(b) What number comes immediately after 18? "
            "(c) Which number is greater: 12 or 17?"
        ),
        question_type=QuestionType.LONG_ANSWER,
        marks=5,
        difficulty=Difficulty.MEDIUM,
        grade="1",
        topic="Number 11 to 20",
        answer="(a) Fifteen (b) 19 (c) 17",
        solution="(a) Fifteen [2 Marks] (b) 19 [2 Marks] (c) 17 [1 Mark]",
    )
    ok, err = validate_question_quality(structured_q)
    assert ok, f"Structured question was unexpectedly rejected: {err}"


# ---------------------------------------------------------------------------
# 5. Shared Phrase and Cross-Section Duplication Detection
# ---------------------------------------------------------------------------
def test_shared_phrase_detected_between_mcq_and_long_answer():
    """Verify that recycling a distinctive illustration phrase across sections is blocked."""
    q_mcq = Question(
        question_text="Which spatial term describes a bird sitting on the top of a tree?",
        question_type=QuestionType.MULTIPLE_CHOICE,
        marks=1,
        difficulty=Difficulty.EASY,
        grade="1",
        topic="Spatial Understanding",
        chapter="Lesson 1: Spatial Understanding",
        options=["Top", "Under", "Inside", "Far"],
        correct_option="Top",
        answer="Top",
        solution="Top",
    )

    q_la = Question(
        question_text=(
            "A bird is sitting on the top of a tree while a dog is resting under the table: "
            "(a) Identify the bird's position. (b) Identify the dog's position."
        ),
        question_type=QuestionType.LONG_ANSWER,
        marks=5,
        difficulty=Difficulty.MEDIUM,
        grade="1",
        topic="Spatial Understanding",
        chapter="Lesson 1: Spatial Understanding",
        answer="(a) Top (b) Under",
        solution="Solutions",
    )

    # 1. check_shared_phrase detects 'bird sitting top tree'
    has_phrase, phrase = check_shared_phrase(q_mcq.question_text, q_la.question_text)
    assert has_phrase is True
    assert "bird" in phrase and "top" in phrase

    # 2. is_duplicate flags this pair as duplicate
    is_dup, reason = is_duplicate(q_la, [q_mcq])
    assert is_dup is True
    assert "shared phrase" in reason.lower() or "same concept" in reason.lower()


def test_same_chapter_concept_collision_detected():
    """Two questions in the same chapter testing identical spatial concepts fail uniqueness."""
    q1 = Question(
        question_text="Choose the correct position of an object resting on top of a box.",
        question_type=QuestionType.MULTIPLE_CHOICE,
        marks=1,
        difficulty=Difficulty.EASY,
        grade="1",
        chapter="Lesson 1: Spatial Understanding",
        topic="Spatial Understanding",
        options=["Top", "Under", "Inside", "Bottom"],
        correct_option="Top",
        answer="Top",
        solution="Top",
    )
    q2 = Question(
        question_text="Identify whether the object resting on top of the box is inside or outside.",
        question_type=QuestionType.SHORT_ANSWER,
        marks=2,
        difficulty=Difficulty.EASY,
        grade="1",
        chapter="Lesson 1: Spatial Understanding",
        topic="Spatial Understanding",
        answer="Outside",
        solution="Outside",
    )
    is_dup, reason = is_duplicate(q2, [q1])
    assert is_dup is True
    assert "concept" in reason.lower() or "same concept" in reason.lower() or "shared phrase" in reason.lower()


# ---------------------------------------------------------------------------
# 6. Intelligent Blueprint Chapter Allocation for Class 2 Prerequisite
# ---------------------------------------------------------------------------
def test_blueprint_chapter_allocation_class2_prerequisite_prevents_slot_mismatch():
    """
    Verify intelligent allocation for Class 2 Prerequisite (9 Class 1 chapters, 10 questions):
    - Section C (Long Answer, 3 questions, 5M each) receives operational chapters (Addition, Subtraction, Units & Tens).
    - Section A (MCQ, 4 questions, 1M each) receives foundational chapters (Spatial Understanding, Solids, Zero, Digits).
    - Lesson 1: Spatial Understanding is NEVER placed in Section C (5 Marks).
    - All 9 chapters are represented across the paper.
    """
    class_1_chapters = get_chapters_for_class(1)
    assert len(class_1_chapters) == 9

    bp = create_default_blueprint(
        num_questions=10,
        total_marks=25,
        question_types=[QuestionType.MULTIPLE_CHOICE, QuestionType.SHORT_ANSWER, QuestionType.LONG_ANSWER],
        grade="1",
    )
    assert len(bp.sections) == 3
    sec_a, sec_b, sec_c = bp.sections

    bp, explanation = assign_blueprint_chapter_coverage(bp, class_1_chapters)

    assert "All 9 eligible curriculum chapters are assessed" in explanation
    assert len(bp.allocated_chapters) == 9

    # Check Section C (Long Answer):
    # Must NOT contain 'Lesson 1: Spatial Understanding' or 'Lesson 4: Zero'!
    for ch in sec_c.chapter_allocation:
        assert "spatial" not in ch.lower(), f"Spatial Understanding was inappropriately allocated to Long Answer: {ch}"
        assert "zero" not in ch.lower(), f"Zero was inappropriately allocated to Long Answer: {ch}"
        assert is_computational_chapter(ch), f"Expected computational chapter in Long Answer, got: {ch}"

    # Check Section A (MCQ):
    # Contains foundational/introductory chapters
    sec_a_lower = [ch.lower() for ch in sec_a.chapter_allocation]
    assert any("spatial" in ch for ch in sec_a_lower)
    assert any("zero" in ch for ch in sec_a_lower) or any("solids" in ch for ch in sec_a_lower)

    # Check total coverage: all 9 chapters are represented
    all_assigned = []
    for s in bp.sections:
        all_assigned.extend(s.chapter_allocation)
    assert len(all_assigned) == 10
    assert set(all_assigned) == set(class_1_chapters)


# ---------------------------------------------------------------------------
# 7. Intelligent Blueprint Allocation Across All Classes (1 to 5) and Exams
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("class_num", [1, 2, 3, 4, 5])
def test_blueprint_final_exam_coverage_all_classes(class_num):
    """Verify balanced full-syllabus blueprint allocation for Final Examinations Classes 1 to 5."""
    chapters = get_chapters_for_class(class_num)
    bp = create_default_blueprint(
        num_questions=len(chapters),
        total_marks=len(chapters) * 2,
        question_types=[QuestionType.MULTIPLE_CHOICE, QuestionType.SHORT_ANSWER, QuestionType.LONG_ANSWER],
        grade=str(class_num),
    )
    bp, _ = assign_blueprint_chapter_coverage(bp, chapters)

    all_assigned = []
    for s in bp.sections:
        all_assigned.extend(s.chapter_allocation)

    assert len(all_assigned) == len(chapters)
    assert set(all_assigned) == set(chapters)


@pytest.mark.parametrize("class_num", [1, 2, 3, 4, 5])
def test_blueprint_midterm_exam_coverage_all_classes(class_num):
    """Verify first-half syllabus blueprint allocation for Mid-Semester Examinations Classes 1 to 5."""
    midterm_chapters = get_midterm_chapters_for_class(class_num)
    bp = create_default_blueprint(
        num_questions=len(midterm_chapters),
        total_marks=len(midterm_chapters) * 2,
        question_types=[QuestionType.MULTIPLE_CHOICE, QuestionType.SHORT_ANSWER],
        grade=str(class_num),
    )
    bp, _ = assign_blueprint_chapter_coverage(bp, midterm_chapters)

    all_assigned = []
    for s in bp.sections:
        all_assigned.extend(s.chapter_allocation)

    assert len(all_assigned) == len(midterm_chapters)
    assert set(all_assigned) == set(midterm_chapters)


# ---------------------------------------------------------------------------
# 8. Teacher and Student PDF Answer Separation
# ---------------------------------------------------------------------------
def test_pdf_export_student_vs_teacher_answer_separation():
    """Verify that student PDF omits all answers while teacher PDF includes solutions."""
    q1 = Question(
        question_id="q1",
        question_text="Which of the following shapes has three sides?",
        question_type=QuestionType.MULTIPLE_CHOICE,
        marks=1,
        difficulty=Difficulty.EASY,
        grade="1",
        topic="Shapes",
        chapter="Lesson 2: Solids Around Us",
        options=["Circle", "Triangle", "Square", "Rectangle"],
        correct_option="Triangle",
        answer="Triangle",
        solution="A triangle has 3 sides and 3 corners.",
    )
    q2 = Question(
        question_id="q2",
        question_text="Solve the addition: 5 + 3 = ___",
        question_type=QuestionType.SHORT_ANSWER,
        marks=2,
        difficulty=Difficulty.EASY,
        grade="1",
        topic="Addition",
        chapter="Lesson 5: Addition",
        answer="8",
        solution="5 + 3 = 8 [2 Marks]",
    )

    paper = Paper(
        paper_id="test_pdf_sep_01",
        title="Class 1 Mathematics Model Paper",
        subject="Mathematics",
        topic="Mathematics",
        grade="1",
        academic_grade="1",
        curriculum_grade="1",
        exam_type=ExamType.FINAL,
        total_marks=3,
        num_questions=2,
        difficulty=Difficulty.EASY,
        questions=[q1, q2],
        sections=[
            PaperSection(
                section_id="A",
                title="Section A: Multiple Choice Questions",
                question_type=QuestionType.MULTIPLE_CHOICE,
                num_questions=1,
                section_marks=1,
                instructions="Choose the correct option.",
                questions=[q1],
            ),
            PaperSection(
                section_id="B",
                title="Section B: Short Answer Questions",
                question_type=QuestionType.SHORT_ANSWER,
                num_questions=1,
                section_marks=2,
                instructions="Write the answer clearly.",
                questions=[q2],
            ),
        ],
    )

    # 1. Generate Student PDF (include_answers=False)
    student_pdf_bytes = generate_pdf(paper, include_answers=False)
    assert len(student_pdf_bytes) > 1000

    # 2. Generate Teacher PDF (include_answers=True)
    teacher_pdf_bytes = generate_pdf(paper, include_answers=True)
    assert len(teacher_pdf_bytes) > 1000
    # Teacher PDF is larger because it contains answers, solutions, and marking steps
    assert len(teacher_pdf_bytes) > len(student_pdf_bytes)

    # Check text content of PDFs using PyMuPDF (fitz)
    import fitz

    # Extract student text
    doc_student = fitz.open(stream=student_pdf_bytes, filetype="pdf")
    student_text = "".join(page.get_text() for page in doc_student)

    # Extract teacher text
    doc_teacher = fitz.open(stream=teacher_pdf_bytes, filetype="pdf")
    teacher_text = "".join(page.get_text() for page in doc_teacher)

    # Student text must NOT contain answer keywords
    assert "Correct Option:" not in student_text
    assert "Key Answer:" not in student_text
    assert "Worked Solution & Marking Steps:" not in student_text
    assert "A triangle has 3 sides and 3 corners." not in student_text

    # Teacher text MUST contain answers and marking guidance
    assert "Correct Option: Triangle" in teacher_text or "Key Answer: Triangle" in teacher_text
    assert "Worked Solution & Marking Steps:" in teacher_text
    assert "A triangle has 3 sides and 3 corners." in teacher_text
