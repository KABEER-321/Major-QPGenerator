# tests/test_structured_paper.py
# Comprehensive tests for structured question paper generation, blueprint solver,
# section-by-section allocation, marks distribution, and PDF exports.

import pytest
from app.schemas import (
    Paper,
    Question,
    QuestionType,
    Difficulty,
    ExamType,
    PaperSection,
    PaperGenerateRequest,
    create_default_blueprint,
)
from app.paper import assemble_paper
from app.pdf_export import generate_pdf


class TestStructuredPaperClass2:
    """Tests specifically targeting Class 2 Final Examination with 25 questions and 100 marks."""

    def test_class_2_final_exam_blueprint(self):
        """Verify blueprint solver for 25 questions, 100 marks across 3 question types."""
        bp = create_default_blueprint(
            num_questions=25,
            total_marks=100,
            question_types=[
                QuestionType.MULTIPLE_CHOICE,
                QuestionType.SHORT_ANSWER,
                QuestionType.LONG_ANSWER,
            ],
            overall_difficulty=Difficulty.MEDIUM,
            topic="Mathematics",
        )

        assert bp.total_questions == 25
        assert bp.total_marks == 100
        assert len(bp.sections) == 3

        sec_a, sec_b, sec_c = bp.sections[0], bp.sections[1], bp.sections[2]

        # Section A: Multiple Choice
        assert sec_a.question_type == QuestionType.MULTIPLE_CHOICE
        assert sec_a.section_id == "A"
        assert "Multiple Choice" in sec_a.title
        assert sec_a.num_questions == 10
        assert sec_a.section_marks == 20
        assert sec_a.marks_per_question == 2

        # Section B: Short Answer
        assert sec_b.question_type == QuestionType.SHORT_ANSWER
        assert sec_b.section_id == "B"
        assert "Short Answer" in sec_b.title
        assert sec_b.num_questions == 10
        assert sec_b.section_marks == 30
        assert sec_b.marks_per_question == 3

        # Section C: Long Answer / Problem Solving
        assert sec_c.question_type == QuestionType.LONG_ANSWER
        assert sec_c.section_id == "C"
        assert "Long Answer" in sec_c.title
        assert sec_c.num_questions == 5
        assert sec_c.section_marks == 50
        assert sec_c.marks_per_question == 10

        # Verify hierarchical marks progression
        assert sec_a.marks_per_question < sec_b.marks_per_question < sec_c.marks_per_question

        # Verify sums
        assert sec_a.num_questions + sec_b.num_questions + sec_c.num_questions == 25
        assert sec_a.section_marks + sec_b.section_marks + sec_c.section_marks == 100

    def test_class_2_paper_assembly_and_sections(self):
        """Verify full assembly of 25-question Class 2 paper with structured sections."""
        # 10 MCQs (2 marks each)
        mcqs = [
            Question(
                question_id=f"mcq_{i+1}",
                question_text=f"Which number comes next in the pattern: {i*2}, {i*2+2}, ___?",
                question_type=QuestionType.MULTIPLE_CHOICE,
                marks=2,
                difficulty=Difficulty.EASY,
                topic="Numbers and Patterns",
                answer=str(i*2 + 4),
                solution=f"Add 2 to previous number: {i*2+2} + 2 = {i*2+4}.",
                options=[str(i*2+3), str(i*2+4), str(i*2+5), str(i*2+6)],
                correct_option=str(i*2+4),
            )
            for i in range(10)
        ]

        # 10 Short Answer Questions (3 marks each)
        short_answers = [
            Question(
                question_id=f"sa_{i+1}",
                question_text=f"Find the sum of {10 + i*5} and {15 + i*3}.",
                question_type=QuestionType.SHORT_ANSWER,
                marks=3,
                difficulty=Difficulty.MEDIUM,
                topic="Addition and Subtraction",
                answer=str((10 + i*5) + (15 + i*3)),
                solution=f"{10 + i*5} + {15 + i*3} = {(10 + i*5) + (15 + i*3)}.",
            )
            for i in range(10)
        ]

        # 5 Long Answer Questions (10 marks each)
        long_answers = [
            Question(
                question_id=f"la_{i+1}",
                question_text=(
                    f"Ravi has {20 + i*4} marbles. His sister gives him {15 + i*2} more marbles. "
                    f"He then gives {10 + i} marbles to his friend. "
                    f"How many marbles does Ravi have now? Show all steps."
                ),
                question_type=QuestionType.LONG_ANSWER,
                marks=10,
                difficulty=Difficulty.HARD,
                topic="Word Problems",
                answer=f"{(20 + i*4) + (15 + i*2) - (10 + i)} marbles",
                solution=(
                    f"Step 1: Initial marbles = {20 + i*4}.\n"
                    f"Step 2: After sister gives marbles = {(20 + i*4) + (15 + i*2)}.\n"
                    f"Step 3: After giving to friend = {(20 + i*4) + (15 + i*2) - (10 + i)} marbles."
                ),
            )
            for i in range(5)
        ]

        all_questions = mcqs + short_answers + long_answers
        assert len(all_questions) == 25
        assert sum(q.marks for q in all_questions) == 100

        req = PaperGenerateRequest(
            grade="2",
            academic_grade="Class 2",
            curriculum_grade="Class 2",
            subject="Mathematics",
            exam_type=ExamType.FINAL,
            num_questions=25,
            total_marks=100,
            difficulty=Difficulty.MEDIUM,
            question_types=[
                QuestionType.MULTIPLE_CHOICE,
                QuestionType.SHORT_ANSWER,
                QuestionType.LONG_ANSWER,
            ],
        )

        paper = assemble_paper(request=req, questions=all_questions)

        assert paper.num_questions == 25
        assert paper.total_marks == 100
        assert len(paper.sections) == 3

        assert paper.sections[0].num_questions == 10
        assert paper.sections[0].section_marks == 20
        assert paper.sections[1].num_questions == 10
        assert paper.sections[1].section_marks == 30
        assert paper.sections[2].num_questions == 5
        assert paper.sections[2].section_marks == 50

        # Test PDF generation for both Teacher and Student
        pdf_teacher = generate_pdf(paper, include_answers=True)
        pdf_student = generate_pdf(paper, include_answers=False)

        assert isinstance(pdf_teacher, bytes)
        assert isinstance(pdf_student, bytes)
        assert len(pdf_teacher) > 8000
        assert len(pdf_student) > 6000
        # Teacher copy includes solutions and marking scheme
        assert len(pdf_teacher) >= len(pdf_student)

    def test_respects_single_question_type_selection(self):
        """When teacher selects only 1 question type, blueprint contains ONLY that type."""
        bp = create_default_blueprint(
            num_questions=10,
            total_marks=30,
            question_types=[QuestionType.SHORT_ANSWER],
            topic="Addition",
        )
        assert len(bp.sections) == 1
        assert bp.sections[0].question_type == QuestionType.SHORT_ANSWER
        assert bp.sections[0].num_questions == 10
        assert bp.sections[0].section_marks == 30
        assert bp.sections[0].marks_per_question == 3

    def test_respects_two_question_types_selection(self):
        """When teacher selects only 2 question types, blueprint contains ONLY those 2 types."""
        bp = create_default_blueprint(
            num_questions=15,
            total_marks=45,
            question_types=[QuestionType.MULTIPLE_CHOICE, QuestionType.SHORT_ANSWER],
            topic="Geometry",
        )
        assert len(bp.sections) == 2
        allowed_types = {QuestionType.MULTIPLE_CHOICE, QuestionType.SHORT_ANSWER}
        for sec in bp.sections:
            assert sec.question_type in allowed_types
            assert sec.question_type != QuestionType.LONG_ANSWER
        assert bp.total_questions == 15
        assert bp.total_marks == 45

    def test_impossible_marks_allocation_fails(self):
        """Requesting fewer marks than questions must fail loudly with ValueError."""
        with pytest.raises(ValueError, match="must be at least num_questions"):
            create_default_blueprint(
                num_questions=25,
                total_marks=20,
                question_types=[QuestionType.MULTIPLE_CHOICE],
            )
