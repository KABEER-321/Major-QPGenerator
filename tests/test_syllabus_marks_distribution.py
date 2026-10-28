# tests/test_syllabus_marks_distribution.py
# Automated tests for syllabus-wise marks and question distribution across all examination
# types (Prerequisite, Mid-Semester, and Final) and all supported classes (1-5).
# Specifically verifies prevention and rejection of the 1/1/23 marks distribution failure.

import pytest
from app.schemas import (
    ExamType,
    Question,
    QuestionType,
    Difficulty,
    Paper,
    PaperSection,
    ExamBlueprint,
    create_default_blueprint,
)
from app.curriculum import (
    get_chapters_for_class,
    get_midterm_chapters_for_class,
    get_chapter_pedagogical_weight,
    allocate_chapter_marks_and_questions,
    assign_blueprint_chapter_coverage,
    is_computational_chapter,
    is_foundational_chapter,
)
from app.validator import (
    validate_paper_chapter_marks_distribution,
    validate_full_paper,
)
from app.paper import assemble_paper
from app.schemas import PaperGenerateRequest


# ---------------------------------------------------------------------------
# 1. Catch the Exact 1/1/23 Marks Distribution Failure from Uploaded PDF
# ---------------------------------------------------------------------------
def test_rejects_exact_1_1_23_distribution_failure():
    """
    Simulate the exact Class 3 Mid-Semester 25-mark examination paper shown in the uploaded PDF:
    - 10 Questions total across 3 Sections:
      Section A: 4 x 1M MCQ (Q1: Shapes, Q2: Numbers, Q3: Addition, Q4: Addition)
      Section B: 3 x 2M SA (Q5: Addition, Q6: Addition, Q7: Addition)
      Section C: 3 x 5M LA (Q8: Addition, Q9: Addition, Q10: Addition)
    Result:
      Chapter 1: Shapes = 1 Mark (4%)
      Chapter 2: Numbers = 1 Mark (4%)
      Chapter 3: Addition = 23 Marks (92%)
    The validator MUST reject this paper and cite syllabus marks distribution failure.
    """
    permitted_chapters = [
        "Chapter 1: Shapes",
        "Chapter 2: Numbers",
        "Chapter 3: Addition",
    ]

    # Reconstruct the exact questions from the uploaded PDF
    q1 = Question(
        question_text="Which of the following is a plane figure with 4 sides, 4 corners, and 2 diagonals?",
        question_type=QuestionType.MULTIPLE_CHOICE,
        marks=1,
        difficulty=Difficulty.EASY,
        chapter="Chapter 1: Shapes",
        topic="Shapes",
        options=["Triangle", "Quadrilateral", "Cylinder", "Circle"],
        correct_option="Quadrilateral",
        answer="Quadrilateral",
        solution="A quadrilateral is a 2-D plane figure.",
    )
    q2 = Question(
        question_text="What is the standard numeral for 'Eight hundred ninety nine'?",
        question_type=QuestionType.MULTIPLE_CHOICE,
        marks=1,
        difficulty=Difficulty.EASY,
        chapter="Chapter 2: Numbers",
        topic="Numbers",
        options=["809", "890", "899", "989"],
        correct_option="899",
        answer="899",
        solution="Written as 899.",
    )
    q3 = Question(
        question_text="Find the sum of 43 and 35 by separating them into tens and units.",
        question_type=QuestionType.MULTIPLE_CHOICE,
        marks=1,
        difficulty=Difficulty.EASY,
        chapter="Chapter 3: Addition",
        topic="Addition",
        options=["75", "78", "87", "83"],
        correct_option="78",
        answer="78",
        solution="40+30=70, 3+5=8 -> 78.",
    )
    q4 = Question(
        question_text="What is the total sum when you add 23 and 64?",
        question_type=QuestionType.MULTIPLE_CHOICE,
        marks=1,
        difficulty=Difficulty.EASY,
        chapter="Chapter 3: Addition",
        topic="Addition",
        options=["85", "86", "87", "97"],
        correct_option="87",
        answer="87",
        solution="20+60=80, 3+4=7 -> 87.",
    )
    q5 = Question(
        question_text="Add the 3-digit numbers 342 and 156 using standard column addition.",
        question_type=QuestionType.SHORT_ANSWER,
        marks=2,
        difficulty=Difficulty.MEDIUM,
        chapter="Chapter 3: Addition",
        topic="Addition",
        answer="498",
        solution="Column addition: 342 + 156 = 498.",
    )
    q6 = Question(
        question_text="Add the numbers 512 and 221.",
        question_type=QuestionType.SHORT_ANSWER,
        marks=2,
        difficulty=Difficulty.MEDIUM,
        chapter="Chapter 3: Addition",
        topic="Addition",
        answer="733",
        solution="512 + 221 = 733.",
    )
    q7 = Question(
        question_text="Add 400 and 345.",
        question_type=QuestionType.SHORT_ANSWER,
        marks=2,
        difficulty=Difficulty.MEDIUM,
        chapter="Chapter 3: Addition",
        topic="Addition",
        answer="745",
        solution="400 + 345 = 745.",
    )
    q8 = Question(
        question_text="A school library has 342 English books and 256 Science books. (a) Express in expanded form. (b) Add columns. (c) Total books.",
        question_type=QuestionType.LONG_ANSWER,
        marks=5,
        difficulty=Difficulty.HARD,
        chapter="Chapter 3: Addition",
        topic="Addition",
        answer="(a) 300+40+2; (b) 500+90+8; (c) 598",
        solution="Worked steps for books addition.",
    )
    q9 = Question(
        question_text="A toy factory produced 524 red cars and 235 blue cars. (a) Expand. (b) Group. (c) Total.",
        question_type=QuestionType.LONG_ANSWER,
        marks=5,
        difficulty=Difficulty.HARD,
        chapter="Chapter 3: Addition",
        topic="Addition",
        answer="(a) 500+20+4; (b) 700+50+9; (c) 759",
        solution="Worked steps for toy cars addition.",
    )
    q10 = Question(
        question_text="Consider 684 and 213. (a) Write expanded form. (b) Set up standard addition. (c) Write final sum.",
        question_type=QuestionType.LONG_ANSWER,
        marks=5,
        difficulty=Difficulty.HARD,
        chapter="Chapter 3: Addition",
        topic="Addition",
        answer="Sum = 897",
        solution="Worked steps: 684 + 213 = 897.",
    )

    questions = [q1, q2, q3, q4, q5, q6, q7, q8, q9, q10]
    sec_a = PaperSection(section_id="A", title="Section A", question_type=QuestionType.MULTIPLE_CHOICE, num_questions=4, section_marks=4, questions=[q1, q2, q3, q4])
    sec_b = PaperSection(section_id="B", title="Section B", question_type=QuestionType.SHORT_ANSWER, num_questions=3, section_marks=6, questions=[q5, q6, q7])
    sec_c = PaperSection(section_id="C", title="Section C", question_type=QuestionType.LONG_ANSWER, num_questions=3, section_marks=15, questions=[q8, q9, q10])

    paper = Paper(
        paper_id="test_failed_paper",
        title="Class 3 Mathematics — Mid-Semester Examination",
        grade="Class 3",
        subject="Mathematics",
        topic="Mathematics",
        exam_type=ExamType.MIDTERM,
        academic_grade="Class 3",
        curriculum_grade="Class 3",
        permitted_units=permitted_chapters,
        forbidden_units=["Chapter 4: Subtraction", "Chapter 5: Multiplication"],
        difficulty=Difficulty.MEDIUM,
        questions=questions,
        num_questions=10,
        total_marks=25,
        sections=[sec_a, sec_b, sec_c],
    )

    is_ok, err_msg, actual_m, actual_q = validate_paper_chapter_marks_distribution(
        paper=paper,
        permitted_chapters=permitted_chapters,
    )

    # Must be strictly rejected!
    assert not is_ok, "Expected 1/1/23 marks distribution failure to be rejected, but it passed!"
    assert "distribution failure" in err_msg.lower()
    assert "addition" in err_msg.lower()
    assert actual_m["Chapter 3: Addition"] == 23
    assert actual_m["Chapter 1: Shapes"] == 1
    assert actual_m["Chapter 2: Numbers"] == 1

    # Full paper validation must also fail with this error
    full_ok, full_errors = validate_full_paper(paper)
    assert not full_ok
    assert any("distribution failure" in e.lower() for e in full_errors)


# ---------------------------------------------------------------------------
# 2. Balanced Blueprint Creation for Class 3 Mid-Semester (25 Marks, 10 Questions)
# ---------------------------------------------------------------------------
def test_class_3_midterm_blueprint_produces_5_8_12_allocation():
    """
    Verify that creating a blueprint for Class 3 Mid-Semester (25 marks, 10 questions)
    produces the balanced marks allocation specified in the prompt:
    Shapes = 5 marks, Numbers = 8 marks, Addition = 12 marks.
    """
    chapters = get_midterm_chapters_for_class(3)
    assert chapters == ["Chapter 1: Shapes", "Chapter 2: Numbers", "Chapter 3: Addition"]

    bp = create_default_blueprint(
        num_questions=10,
        total_marks=25,
        question_types=[QuestionType.MULTIPLE_CHOICE, QuestionType.SHORT_ANSWER, QuestionType.LONG_ANSWER],
        grade="Class 3",
    )
    bp, explanation = assign_blueprint_chapter_coverage(bp, chapters)

    # 1. Target marks check: Shapes=5, Numbers=8, Addition=12
    assert bp.chapter_marks_allocation == {
        "Chapter 1: Shapes": 5,
        "Chapter 2: Numbers": 8,
        "Chapter 3: Addition": 12,
    }

    # 2. Target questions check: Shapes=3, Numbers=3, Addition=4
    assert bp.chapter_questions_allocation == {
        "Chapter 1: Shapes": 3,
        "Chapter 2: Numbers": 3,
        "Chapter 3: Addition": 4,
    }

    # 3. Sum of section question allocations matches exactly
    sec_a, sec_b, sec_c = bp.sections
    assert len(sec_a.chapter_allocation) == 4
    assert len(sec_b.chapter_allocation) == 3
    assert len(sec_c.chapter_allocation) == 3

    # 4. Resulting marks per chapter from section slot assignment
    actual_marks = {}
    actual_qs = {}
    for sec in bp.sections:
        for idx, ch in enumerate(sec.chapter_allocation):
            m = sec.marks_distribution[idx]
            actual_marks[ch] = actual_marks.get(ch, 0) + m
            actual_qs[ch] = actual_qs.get(ch, 0) + 1

    assert actual_marks["Chapter 1: Shapes"] == 5
    assert actual_marks["Chapter 2: Numbers"] == 8
    assert actual_marks["Chapter 3: Addition"] == 12
    assert sum(actual_marks.values()) == 25

    # 5. Long Answer section does not contain Shapes
    assert "shapes" not in [ch.lower() for ch in sec_c.chapter_allocation]


# ---------------------------------------------------------------------------
# 3. Mid-Semester Examinations Balanced Across All Classes 1 to 5
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("class_num", [1, 2, 3, 4, 5])
def test_midterm_blueprint_marks_balanced_all_classes(class_num):
    """
    Verify that Mid-Semester blueprints across Classes 1 to 5 distribute marks and
    questions fairly across all first-half permitted chapters without neglect.
    """
    chapters = get_midterm_chapters_for_class(class_num)
    m = len(chapters)
    total_marks = 25
    total_qs = 10

    bp = create_default_blueprint(
        num_questions=total_qs,
        total_marks=total_marks,
        question_types=[QuestionType.MULTIPLE_CHOICE, QuestionType.SHORT_ANSWER, QuestionType.LONG_ANSWER],
        grade=f"Class {class_num}",
    )
    bp, _ = assign_blueprint_chapter_coverage(bp, chapters)

    # Every permitted chapter must be allocated questions and marks
    assert bp.chapter_marks_allocation is not None
    assert bp.chapter_questions_allocation is not None
    assert sum(bp.chapter_marks_allocation.values()) == total_marks
    assert sum(bp.chapter_questions_allocation.values()) == total_qs

    for ch in chapters:
        assert bp.chapter_questions_allocation[ch] >= 1, f"Chapter {ch} in Class {class_num} Midterm got 0 questions"
        assert bp.chapter_marks_allocation[ch] >= 1, f"Chapter {ch} in Class {class_num} Midterm got 0 marks"

    # No single chapter monopolizes marks
    for ch, marks in bp.chapter_marks_allocation.items():
        share = marks / total_marks
        assert share < 0.65, f"Chapter {ch} in Class {class_num} Midterm dominates with {share*100:.1f}% marks"


# ---------------------------------------------------------------------------
# 4. Prerequisite Examinations Balanced Across Full Previous-Class Syllabus
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("academic_class,previous_class", [(2, 1), (3, 2), (4, 3), (5, 4)])
def test_prerequisite_blueprint_marks_balanced(academic_class, previous_class):
    """
    Verify that Prerequisite blueprints (e.g. Class 2 Prerequisite testing Class 1)
    distribute questions and marks across the entire previous class syllabus.
    """
    prev_chapters = get_chapters_for_class(previous_class)
    m = len(prev_chapters)
    total_qs = max(m, 10)
    total_marks = total_qs * 2

    bp = create_default_blueprint(
        num_questions=total_qs,
        total_marks=total_marks,
        question_types=[QuestionType.MULTIPLE_CHOICE, QuestionType.SHORT_ANSWER, QuestionType.LONG_ANSWER],
        grade=f"Class {previous_class}",
    )
    bp, _ = assign_blueprint_chapter_coverage(bp, prev_chapters)

    assert bp.chapter_marks_allocation is not None
    assert sum(bp.chapter_marks_allocation.values()) == total_marks

    # Every chapter is represented
    for ch in prev_chapters:
        assert bp.chapter_questions_allocation[ch] >= 1
        assert bp.chapter_marks_allocation[ch] >= 1


# ---------------------------------------------------------------------------
# 5. Final Examinations Balanced Across Full Class Syllabus
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("class_num", [1, 2, 3, 4, 5])
def test_final_blueprint_marks_balanced(class_num):
    """
    Verify that Final examination blueprints across Classes 1 to 5 cover the 100% textbook syllabus
    with fair mark weights based on chapter learning objectives.
    """
    chapters = get_chapters_for_class(class_num)
    m = len(chapters)
    total_qs = m
    total_marks = m * 3

    bp = create_default_blueprint(
        num_questions=total_qs,
        total_marks=total_marks,
        question_types=[QuestionType.MULTIPLE_CHOICE, QuestionType.SHORT_ANSWER, QuestionType.LONG_ANSWER],
        grade=f"Class {class_num}",
    )
    bp, _ = assign_blueprint_chapter_coverage(bp, chapters)

    assert sum(bp.chapter_marks_allocation.values()) == total_marks
    assert sum(bp.chapter_questions_allocation.values()) == total_qs

    for ch in chapters:
        assert bp.chapter_questions_allocation[ch] == 1
        assert bp.chapter_marks_allocation[ch] >= 1


# ---------------------------------------------------------------------------
# 6. Fair Distribution When Requested Questions Are Fewer Than Chapters (N < M)
# ---------------------------------------------------------------------------
def test_fair_distribution_when_n_less_than_m():
    """
    Verify that when a teacher requests fewer questions than available chapters (e.g. 3 questions for 6 chapters),
    the system selects representative chapters, allocates marks fairly, and reports the chapters actually assessed.
    """
    chapters = get_chapters_for_class(2)  # 6 chapters
    bp = create_default_blueprint(
        num_questions=3,
        total_marks=10,
        question_types=[QuestionType.MULTIPLE_CHOICE, QuestionType.SHORT_ANSWER],
        grade="Class 2",
    )
    bp, explanation = assign_blueprint_chapter_coverage(bp, chapters)

    assert len(bp.allocated_chapters) == 3
    assert sum(bp.chapter_marks_allocation.values()) == 10
    assert sum(bp.chapter_questions_allocation.values()) == 3
    assert "3 chapters selected across the 6-chapter syllabus" in explanation
    assert "because requested question count (3) is smaller than available chapters (6)" in explanation


# ---------------------------------------------------------------------------
# 7. Mathematical Consistency Verification
# ---------------------------------------------------------------------------
def test_mathematical_consistency_audit():
    """
    Verify that section totals, question marks, chapter totals, and overall totals
    are strictly consistent and verified by the validator.
    """
    permitted = ["Chapter 1: Shapes", "Chapter 2: Numbers", "Chapter 3: Addition"]
    req = PaperGenerateRequest(
        grade="Class 3",
        subject="Mathematics",
        topic="Mathematics",
        exam_type=ExamType.MIDTERM,
        academic_grade="Class 3",
        curriculum_grade="Class 3",
        num_questions=10,
        total_marks=25,
        permitted_units=permitted,
    )

    bp = req.get_or_create_blueprint()
    bp, _ = assign_blueprint_chapter_coverage(bp, permitted)

    # Construct mock validated questions matching the blueprint
    mock_questions = []
    for sec in bp.sections:
        for idx, ch in enumerate(sec.chapter_allocation):
            m = sec.marks_distribution[idx]
            mock_questions.append(
                Question(
                    question_text=f"Math exercise for {ch} slot {idx+1}",
                    question_type=sec.question_type,
                    marks=m,
                    difficulty=Difficulty.MEDIUM,
                    grade="Class 3",
                    chapter=ch,
                    topic=ch,
                    options=["A", "B", "C", "D"] if sec.question_type == QuestionType.MULTIPLE_CHOICE else None,
                    correct_option="A" if sec.question_type == QuestionType.MULTIPLE_CHOICE else None,
                    answer="42",
                    solution="Step by step solution",
                )
            )

    paper = assemble_paper(request=req, questions=mock_questions, blueprint_id=bp.blueprint_id)

    # Validate paper
    is_valid, err_msg, ch_marks, ch_qs = validate_paper_chapter_marks_distribution(
        paper=paper,
        blueprint=bp,
        permitted_chapters=permitted,
    )

    assert is_valid, f"Expected mathematically consistent paper to pass, but got: {err_msg}"
    assert ch_marks == {"Chapter 1: Shapes": 5, "Chapter 2: Numbers": 8, "Chapter 3: Addition": 12}
    assert ch_qs == {"Chapter 1: Shapes": 3, "Chapter 2: Numbers": 3, "Chapter 3: Addition": 4}
    assert sum(ch_marks.values()) == paper.total_marks
    assert paper.chapter_marks_distribution == ch_marks
    assert paper.chapter_questions_distribution == ch_qs
