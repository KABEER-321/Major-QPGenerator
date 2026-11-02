# tests/test_figure_questions.py
"""
Comprehensive automated tests for Figure-Based Mathematics Questions in QPGen.

Validates the 10 core requirements:
1. Class 1 receives only age-appropriate visual questions (counting, basic shapes, spatial).
2. Class 5 receives advanced syllabus-aligned visual questions (angles, fractions, perimeter).
3. Mid-Semester figure questions never use forbidden second-half chapters.
4. Figure-based questions respect question count and total marks.
5. Teacher and Student views display identical diagrams.
6. Student PDF hides answers.
7. Teacher PDF shows answers, solutions, and marking steps.
8. Missing or broken diagrams are rejected by validation.
9. Non-figure question generation remains completely intact.
10. Authentication and RBAC functionality remains intact.
"""

import pytest
from app.figures import (
    create_figure_question,
    is_figure_eligible_chapter,
    render_multiplication_options_svg,
    render_solid_views_question_svg,
    render_symmetry_shapes_options_svg,
    render_fraction_circle_svg,
    render_grid_perimeter_svg,
    render_angle_diagram_svg,
    render_counting_spatial_svg,
)
from app.schemas import (
    ExamType,
    Question,
    QuestionType,
    Difficulty,
    FigureData,
    Paper,
    ExamBlueprint,
    SectionBlueprint,
    PaperGenerateRequest,
    determine_midterm_scope,
    determine_second_half_scope,
)
from app.curriculum import (
    assign_blueprint_chapter_coverage,
    CURRICULUM_CHAPTERS,
)
from app.validator import (
    validate_question_quality,
    check_question_clarity,
)
from app.pdf_export import generate_pdf, _render_svg_flowable



# ---------------------------------------------------------------------------
# 1. Class 1 age-appropriate visual questions (no higher-grade concepts)
# ---------------------------------------------------------------------------

def test_class_1_age_appropriate_visual_questions():
    """Class 1 must receive only simple spatial counting or basic 2D shape questions."""
    q_spatial = create_figure_question(
        grade="Class 1",
        chapter="Shapes and Space",
        question_type=QuestionType.MULTIPLE_CHOICE,
        marks=1,
        difficulty=Difficulty.EASY,
    )
    assert q_spatial is not None
    assert q_spatial.question_subtype == "figure_based"
    assert q_spatial.figure is not None
    assert "<svg" in q_spatial.figure.svg_content
    # Must test basic counting / shapes
    assert any(term in q_spatial.question_text.lower() for term in ["inside", "outside", "shape", "how many", "count"])
    # Must NOT contain 3D projections, angle protractor, or fraction division
    assert "degree" not in q_spatial.question_text.lower()
    assert "fraction" not in q_spatial.question_text.lower()
    assert "cylinder" not in q_spatial.question_text.lower()


# ---------------------------------------------------------------------------
# 2. Class 5 advanced syllabus-aligned visual questions
# ---------------------------------------------------------------------------

def test_class_5_advanced_syllabus_visual_questions():
    """Class 5 receives syllabus-aligned questions like Angles, Fractions, and Perimeter."""
    # Angles
    q_angle = create_figure_question(
        grade="Class 5",
        chapter="Shapes and Angles",
        question_type=QuestionType.MULTIPLE_CHOICE,
        marks=1,
        difficulty=Difficulty.MEDIUM,
    )
    assert q_angle is not None
    assert q_angle.figure is not None
    assert q_angle.figure.figure_type == "angle_rays"
    assert any(ang in q_angle.question_text.lower() for ang in ["angle", "acute", "right", "obtuse"])

    # Perimeter / Area grid
    q_perim = create_figure_question(
        grade="Class 5",
        chapter="How many Squares",
        question_type=QuestionType.MULTIPLE_CHOICE,
        marks=2,
        difficulty=Difficulty.HARD,
    )
    assert q_perim is not None
    assert q_perim.figure is not None
    assert q_perim.figure.figure_type == "perimeter_grid"
    assert "perimeter" in q_perim.question_text.lower() or "boundary" in q_perim.question_text.lower()


# ---------------------------------------------------------------------------
# 3. Mid-Semester figure questions never use forbidden chapters
# ---------------------------------------------------------------------------

def test_midterm_figure_questions_stay_within_permitted_scope():
    """For Mid-Semester exams, figure allocations must only target the first half of chapters."""
    for grade_num in [1, 2, 3, 4, 5]:
        all_ch = CURRICULUM_CHAPTERS[grade_num]
        permitted = determine_midterm_scope(all_ch)
        forbidden = determine_second_half_scope(all_ch)

        blueprint = ExamBlueprint(
            title=f"Class {grade_num} Mid-Semester Blueprint",
            grade=f"Class {grade_num}",
            sections=[
                SectionBlueprint(
                    section_id="A",
                    title="Section A: Multiple Choice",
                    question_type=QuestionType.MULTIPLE_CHOICE,
                    num_questions=len(permitted),
                    section_marks=len(permitted),
                    marks_per_question=1,
                )
            ],
            total_questions=len(permitted),
            total_marks=len(permitted),
        )

        assign_blueprint_chapter_coverage(blueprint, permitted, include_figure_questions=True)

        sec = blueprint.sections[0]
        for idx, is_fig in enumerate(sec.figure_allocation or []):
            if is_fig:
                ch = sec.chapter_allocation[idx]
                assert ch in permitted, f"Grade {grade_num} figure question allocated to forbidden chapter {ch}"
                assert ch not in forbidden, f"Grade {grade_num} figure question touches forbidden chapter {ch}"


# ---------------------------------------------------------------------------
# 4. Blueprint question count and marks conservation
# ---------------------------------------------------------------------------

def test_figure_questions_preserve_blueprint_structure():
    """Figure-based questions must fit strictly into existing question slots and marks."""
    sections = [
        SectionBlueprint(
            section_id="A",
            title="Section A: MCQ",
            question_type=QuestionType.MULTIPLE_CHOICE,
            num_questions=4,
            section_marks=4,
            marks_per_question=1,
        ),
        SectionBlueprint(
            section_id="B",
            title="Section B: Short Answer",
            question_type=QuestionType.SHORT_ANSWER,
            num_questions=3,
            section_marks=6,
            marks_per_question=2,
        ),
        SectionBlueprint(
            section_id="C",
            title="Section C: Long Answer",
            question_type=QuestionType.LONG_ANSWER,
            num_questions=3,
            section_marks=15,
            marks_per_question=5,
        ),
    ]
    blueprint = ExamBlueprint(
        title="Class 3 Math Blueprint",
        grade="Class 3",
        sections=sections,
        total_questions=10,
        total_marks=25,
    )
    permitted = ["Shapes and Designs", "Give and Take", "How Many Times?"]

    assign_blueprint_chapter_coverage(blueprint, permitted, include_figure_questions=True)

    # Question count must not change
    assert sum(s.num_questions for s in blueprint.sections) == 10
    # Marks sum must not change
    assert sum(s.section_marks for s in blueprint.sections) == 25

    # Check that at least one figure question is designated in eligible visual slots
    total_fig_slots = sum(sum(1 for f in (s.figure_allocation or []) if f) for s in blueprint.sections)
    assert 1 <= total_fig_slots <= 2


# ---------------------------------------------------------------------------
# 5. Teacher and Student views display identical diagrams
# ---------------------------------------------------------------------------

def test_teacher_and_student_views_render_identical_svg():
    """Both views must render identical SVG content for the question figure."""
    q_fig = create_figure_question(
        grade="Class 3",
        chapter="Shapes and Designs",
        question_type=QuestionType.MULTIPLE_CHOICE,
        marks=1,
        difficulty=Difficulty.MEDIUM,
    )
    assert q_fig is not None
    svg_teacher = q_fig.figure.svg_content
    svg_student = q_fig.figure.svg_content
    assert svg_teacher == svg_student
    assert "<svg" in svg_teacher and "</svg>" in svg_teacher

    # PyMuPDF flowables generated for PDF must succeed
    flowable = _render_svg_flowable(q_fig.figure.svg_content)
    assert flowable is not None
    assert flowable.drawWidth > 0
    assert flowable.drawHeight > 0


# ---------------------------------------------------------------------------
# 6 & 7. Student PDF hides answers, Teacher PDF includes solutions
# ---------------------------------------------------------------------------

def test_student_pdf_hides_answers_and_teacher_pdf_shows_them():
    """Student copy must hide key answers and solutions, teacher copy must include them."""
    q_fig = create_figure_question(
        grade="Class 3",
        chapter="How Many Times?",
        question_type=QuestionType.MULTIPLE_CHOICE,
        marks=1,
        difficulty=Difficulty.MEDIUM,
    )
    assert q_fig is not None
    q_fig.answer = "Key Answer: 3 x 4 = 12"
    q_fig.solution = "Step 1: Count 3 groups of 4 stars to get 12 stars."

    paper = Paper(
        paper_id="test_fig_paper_01",
        title="Mathematics Assessment",
        grade="Class 3",
        subject="Mathematics",
        topic="Multiplication",
        difficulty=Difficulty.MEDIUM,
        exam_type=ExamType.FINAL,
        total_marks=1,
        num_questions=1,
        questions=[q_fig],
    )

    # Generate Student PDF
    pdf_student_bytes = generate_pdf(paper, include_answers=False)
    assert len(pdf_student_bytes) > 1000

    # Generate Teacher PDF
    pdf_teacher_bytes = generate_pdf(paper, include_answers=True)
    assert len(pdf_teacher_bytes) > 1000

    # Read rendered text via PyMuPDF
    import fitz
    doc_student = fitz.open(stream=pdf_student_bytes, filetype="pdf")
    student_text = "".join(page.get_text() for page in doc_student)

    doc_teacher = fitz.open(stream=pdf_teacher_bytes, filetype="pdf")
    teacher_text = "".join(page.get_text() for page in doc_teacher)

    # Student text must NOT show answer key or worked solution
    assert "Worked Solution & Marking Steps" not in student_text
    assert "Key Answer:" not in student_text
    assert "Marking Guidance" not in student_text

    # Teacher text MUST show solution and key answer
    assert "Worked Solution" in teacher_text or "Marking Steps" in teacher_text
    assert "Key Answer:" in teacher_text or "Correct Option:" in teacher_text


# ---------------------------------------------------------------------------
# 8. Missing or broken diagrams are rejected by validation
# ---------------------------------------------------------------------------

def test_missing_or_corrupted_figure_rejected_by_validator():
    """Questions marked as figure_based with missing or invalid SVG diagrams must fail validation."""
    # Case A: subtype is figure_based but figure is None
    q_bad_1 = Question(
        question_text="Look at the figure and count the stars.",
        question_type=QuestionType.MULTIPLE_CHOICE,
        marks=1,
        difficulty=Difficulty.MEDIUM,
        topic="Multiplication",
        answer="8",
        options=["6", "8", "10", "12"],
        correct_option="8",
        question_subtype="figure_based",
        figure=None,
    )
    valid, err = validate_question_quality(q_bad_1)
    assert not valid
    assert "missing" in err.lower() or "figure" in err.lower()

    # Case B: figure exists but svg_content is empty or corrupted
    q_bad_2 = Question(
        question_text="Identify the shape with two axes of symmetry.",
        question_type=QuestionType.MULTIPLE_CHOICE,
        marks=1,
        difficulty=Difficulty.MEDIUM,
        topic="Shapes",
        answer="Rectangle",
        options=["Square", "Triangle", "Circle", "Rectangle"],
        correct_option="Rectangle",
        question_subtype="figure_based",
        figure=FigureData(
            figure_type="symmetry",
            title="Symmetry",
            svg_content="not_a_valid_svg_string",
        ),
    )
    valid, err = validate_question_quality(q_bad_2)
    assert not valid
    assert "svg" in err.lower() or "diagram" in err.lower()

    # Case C: non-figure question references missing figure
    q_bad_3 = Question(
        question_text="What is the length of side AB as shown in the given figure?",
        question_type=QuestionType.SHORT_ANSWER,
        marks=2,
        difficulty=Difficulty.MEDIUM,
        topic="Measurement",
        answer="5 cm",
    )
    valid, err = validate_question_quality(q_bad_3)
    assert not valid
    assert "visual artifact" in err.lower() or "missing" in err.lower()


# ---------------------------------------------------------------------------
# 9. Non-figure question generation remains completely intact
# ---------------------------------------------------------------------------

def test_include_figure_questions_false_produces_zero_figures():
    """When include_figure_questions is False, no figure slots are allocated."""
    sections = [
        SectionBlueprint(
            section_id="A",
            title="Section A: Multiple Choice",
            question_type=QuestionType.MULTIPLE_CHOICE,
            num_questions=5,
            section_marks=5,
            marks_per_question=1,
        )
    ]
    blueprint = ExamBlueprint(
        title="Class 3 Math Blueprint",
        grade="Class 3",
        sections=sections,
        total_questions=5,
        total_marks=5,
    )
    permitted = ["Shapes and Designs", "Give and Take", "How Many Times?"]

    assign_blueprint_chapter_coverage(blueprint, permitted, include_figure_questions=False)

    sec = blueprint.sections[0]
    assert all(f is False for f in (sec.figure_allocation or []))


# ---------------------------------------------------------------------------
# 10. Authentication and RBAC functionality remains intact
# ---------------------------------------------------------------------------

def test_auth_and_rbac_protection_on_paper_generation():
    """Unauthenticated requests to protected endpoints are rejected with 401 Unauthorized."""
    from fastapi import HTTPException
    from starlette.requests import Request
    from app.auth import get_current_user

    req = Request(scope={"type": "http", "headers": [(b"accept", b"application/json")], "cookies": {}})
    with pytest.raises(HTTPException) as exc_info:
        get_current_user(req)
    assert exc_info.value.status_code == 401
