# app/schemas.py
# Pydantic models for request validation and response data structures.
# These enforce the expected shape of questions, papers, and API requests.

from __future__ import annotations
import math
import re
import uuid
from datetime import datetime
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ExamType(str, Enum):
    """Supported examination types."""
    PREREQUISITE = "prerequisite"
    MIDTERM = "midterm"
    FINAL = "final"

    @property
    def display_name(self) -> str:
        names = {
            ExamType.PREREQUISITE: "Prerequisite Examination",
            ExamType.MIDTERM: "Mid-Semester Examination",
            ExamType.FINAL: "Final Examination",
        }
        return names.get(self, self.value.replace("_", " ").title())


# ---------------------------------------------------------------------------
# Grade and Syllabus Scope Helpers
# ---------------------------------------------------------------------------

def parse_grade_number(grade_str: str) -> Optional[int]:
    """Extract integer class/grade number from strings like '10', 'Class 9', 'Grade 8', '11th'."""
    if not grade_str:
        return None
    match = re.search(r'\b(\d{1,2})(?:st|nd|rd|th)?\b', grade_str.strip(), re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def get_previous_class(grade_str: str) -> str:
    """
    Determine the prerequisite curriculum class (academic class - 1).
    Rule: Prerequisite curriculum class = Selected academic class - 1.
    Raises ValueError if class cannot be determined or if class <= 1.
    """
    grade_num = parse_grade_number(grade_str)
    if grade_num is None:
        raise ValueError(
            f"Cannot determine previous academic class from '{grade_str}'. "
            "Please provide a standard class format (e.g. '10', 'Class 9', 'Grade 8')."
        )
    if grade_num <= 1:
        raise ValueError(
            f"Cannot determine prerequisite class for Class {grade_num}. "
            "Prerequisite assessments require a preceding academic class."
        )
    return str(grade_num - 1)


def format_grade_display(grade_str: str) -> str:
    """Format grade string cleanly as 'Class X'."""
    num = parse_grade_number(grade_str)
    if num is not None:
        return f"Class {num}"
    clean = grade_str.strip()
    return clean if clean.lower().startswith("class") else f"Class {clean}"


def extract_syllabus_units(text: str) -> list[str]:
    """
    Identify units, chapters, or major modules from syllabus text.
    Looks for headers like 'Unit 1: Real Numbers', 'Chapter 2 - Polynomials', etc.
    """
    if not text:
        return []
    pattern = re.compile(
        r'(?:^|\n)\s*(?:(?:Unit|Chapter|Module|Topic)\s*[\dIVXLCDM]+[:\.\-–—\s]+[^\n]+)',
        re.IGNORECASE
    )
    matches = [m.strip() for m in pattern.findall(text)]
    cleaned = []
    seen = set()
    for m in matches:
        line = " ".join(m.split())
        if line.lower() not in seen and len(line) > 3:
            seen.add(line.lower())
            cleaned.append(line)
    return cleaned


def determine_midterm_scope(units: list[str]) -> list[str]:
    """
    For mid-semester examinations, select approximately the first half of syllabus units.
    """
    if not units:
        return []
    half_count = math.ceil(len(units) / 2.0)
    return units[:half_count]


class QuestionType(str, Enum):
    """Supported question types for the first version."""
    SHORT_ANSWER = "short_answer"
    LONG_ANSWER = "long_answer"
    MULTIPLE_CHOICE = "multiple_choice"


class Difficulty(str, Enum):
    """Difficulty levels for generated questions."""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class ValidationStatus(str, Enum):
    """Tracks whether a question passed schema and consistency checks."""
    PENDING = "pending"        # Not yet validated (initial state)
    VALID = "valid"            # Passed all schema validations
    INVALID = "invalid"        # Failed one or more validations
    FIXED = "fixed"            # Was invalid but corrected (e.g. after retry)


class MathCheckStatus(str, Enum):
    """Status of automated mathematical expression verification."""
    VERIFIED = "verified"          # Expression or root was mathematically confirmed by SymPy
    DISCREPANCY = "discrepancy"    # Evaluated expression contradicted the stated answer
    UNSUPPORTED = "unsupported"    # Expression contains natural language, proofs, or unsupported syntax
    SKIPPED = "skipped"            # Question does not require symbolic checking


class MathVerificationResult(BaseModel):
    """Optional, limited mathematical verification result for supported expressions.

    As documented in the specification: SymPy provides limited checking of supported
    expressions; it does not claim to validate every question, geometry, or proof.
    """
    status: MathCheckStatus = Field(
        default=MathCheckStatus.SKIPPED,
        description="Outcome of SymPy expression validation",
    )
    details: str = Field(
        default="",
        description="Explanation of the verification result or mathematical steps checked",
    )
    expression_checked: Optional[str] = Field(
        default=None,
        description="The mathematical expression or equation extracted and tested",
    )


# ---------------------------------------------------------------------------
# Source reference — tracks which syllabus passage a question was based on
# ---------------------------------------------------------------------------

class SourceReference(BaseModel):
    """A reference to a specific passage from the uploaded document.
    This enables traceability from a generated question back to its
    source material, which is critical for curriculum alignment."""
    source_filename: str = Field(
        default="", description="Original filename of the uploaded document"
    )
    chunk_index: Optional[int] = Field(
        default=None, description="Index of the chunk in the document"
    )
    passage_preview: str = Field(
        default="",
        description="Short preview of the passage text (first ~200 chars)",
    )


# ---------------------------------------------------------------------------
# Individual question
# ---------------------------------------------------------------------------

class Question(BaseModel):
    """
    A single generated question.
    For MCQs, 'options' and 'correct_option' must be provided.
    For other types, they may be absent.
    """
    # --- Identity & context fields (populated during paper assembly) ---
    question_id: str = Field(
        default_factory=lambda: uuid.uuid4().hex[:10],
        description="Unique identifier for this question",
    )
    grade: str = Field(
        default="",
        description="Grade or class this question targets (e.g. '10', 'Class 8')",
    )
    subject: str = Field(
        default="Mathematics",
        description="Subject this question belongs to",
    )
    chapter: str = Field(
        default="",
        description="Specific chapter name (may differ from topic)",
    )

    # --- Core question fields (populated by Gemini) ---
    question_text: str = Field(..., min_length=1, description="The question text")
    question_type: QuestionType = Field(..., description="Type of question")
    marks: int = Field(..., gt=0, description="Marks for this question (must be positive)")
    difficulty: Difficulty = Field(..., description="Difficulty level")
    topic: str = Field(..., min_length=1, description="Chapter or topic this question belongs to")
    answer: str = Field(..., min_length=1, description="Expected answer")
    solution: str = Field(
        default="",
        description="Concise solution steps or marking guidance",
    )
    options: Optional[list[str]] = Field(
        default=None,
        description="Multiple-choice options (required for MCQs)",
    )
    correct_option: Optional[str] = Field(
        default=None,
        description="The correct option text (must match one of the options)",
    )

    # --- Traceability & validation fields ---
    source_references: list[SourceReference] = Field(
        default_factory=list,
        description="References to syllabus passages this question is grounded in",
    )
    validation_status: ValidationStatus = Field(
        default=ValidationStatus.PENDING,
        description="Whether this question passed schema validation checks",
    )
    math_verification: Optional[MathVerificationResult] = Field(
        default=None,
        description="Optional SymPy verification for supported mathematical expressions",
    )

    @model_validator(mode="after")
    def validate_mcq_fields(self) -> "Question":
        """Ensure MCQ questions have valid options and correct_option."""
        if self.question_type == QuestionType.MULTIPLE_CHOICE:
            if not self.options or len(self.options) < 2:
                raise ValueError(
                    "Multiple-choice questions must have at least 2 options."
                )
            if not self.correct_option:
                raise ValueError(
                    "Multiple-choice questions must specify a correct_option."
                )
            if self.correct_option not in self.options:
                raise ValueError(
                    f"correct_option '{self.correct_option}' must be one of the options: {self.options}"
                )
        return self


# ---------------------------------------------------------------------------
# Blueprint models
# ---------------------------------------------------------------------------

class SectionBlueprint(BaseModel):
    """
    Blueprint configuration for a single examination section.
    Enforces that questions within the section share a question type,
    and have consistent question counts and marks.
    """
    section_id: str = Field(..., min_length=1, description="Section identifier, e.g. 'A', 'B', 'Sec-1'")
    title: str = Field(..., min_length=1, description="Section title, e.g. 'Section A: Multiple Choice'")
    question_type: QuestionType = Field(..., description="Question type for all questions in this section")
    num_questions: int = Field(..., gt=0, le=50, description="Number of questions in this section")
    section_marks: int = Field(..., gt=0, le=500, description="Total marks allocated to this section")
    marks_per_question: Optional[int] = Field(default=None, gt=0, description="Uniform marks per question")
    marks_distribution: Optional[list[int]] = Field(default=None, description="Explicit marks per question")
    difficulty: Optional[Difficulty] = Field(default=None, description="Difficulty level for this section")
    topics: Optional[list[str]] = Field(default=None, description="Specific topics or subtopics for this section")

    @model_validator(mode="after")
    def validate_section_marks(self) -> "SectionBlueprint":
        if self.marks_distribution is not None:
            if len(self.marks_distribution) != self.num_questions:
                raise ValueError(
                    f"Section '{self.section_id}': marks_distribution length ({len(self.marks_distribution)}) "
                    f"must match num_questions ({self.num_questions})."
                )
            if any(m <= 0 for m in self.marks_distribution):
                raise ValueError(
                    f"Section '{self.section_id}': all marks in marks_distribution must be positive integers."
                )
            if sum(self.marks_distribution) != self.section_marks:
                raise ValueError(
                    f"Section '{self.section_id}': sum of marks_distribution ({sum(self.marks_distribution)}) "
                    f"does not equal section_marks ({self.section_marks})."
                )
            if self.marks_per_question is not None:
                if any(m != self.marks_per_question for m in self.marks_distribution):
                    raise ValueError(
                        f"Section '{self.section_id}': marks_distribution does not match marks_per_question ({self.marks_per_question})."
                    )
        elif self.marks_per_question is not None:
            if self.num_questions * self.marks_per_question != self.section_marks:
                raise ValueError(
                    f"Section '{self.section_id}': num_questions ({self.num_questions}) * "
                    f"marks_per_question ({self.marks_per_question}) = "
                    f"{self.num_questions * self.marks_per_question}, which does not match section_marks ({self.section_marks})."
                )
        else:
            if self.section_marks < self.num_questions:
                raise ValueError(
                    f"Section '{self.section_id}': section_marks ({self.section_marks}) cannot be less than "
                    f"num_questions ({self.num_questions}) because every question must have at least 1 mark."
                )
        return self

    def get_marks_distribution(self) -> list[int]:
        """Return the per-question marks list for this section."""
        if self.marks_distribution:
            return list(self.marks_distribution)
        if self.marks_per_question:
            return [self.marks_per_question] * self.num_questions
        base = self.section_marks // self.num_questions
        rem = self.section_marks % self.num_questions
        return [base + (1 if i < rem else 0) for i in range(self.num_questions)]


class ExamBlueprint(BaseModel):
    """
    Complete blueprint for an examination paper.
    Specifies sections, question counts, marks, and constraints.
    Validates that section totals strictly equal overall paper totals.
    """
    blueprint_id: str = Field(
        default_factory=lambda: uuid.uuid4().hex[:10],
        description="Unique identifier for this blueprint",
    )
    title: str = Field(default="Mathematics Examination Blueprint", description="Blueprint title")
    sections: list[SectionBlueprint] = Field(..., min_length=1, description="Sections defining the paper structure")
    total_questions: int = Field(..., gt=0, le=100, description="Total question count across all sections")
    total_marks: int = Field(..., gt=0, le=1000, description="Total paper marks across all sections")
    difficulty_distribution: Optional[dict[Difficulty, int]] = Field(
        default=None,
        description="Optional breakdown of question counts by difficulty",
    )

    @model_validator(mode="after")
    def validate_totals_and_consistency(self) -> "ExamBlueprint":
        # 1. Check unique section IDs
        sec_ids = [s.section_id for s in self.sections]
        if len(sec_ids) != len(set(sec_ids)):
            duplicates = [sid for sid in sec_ids if sec_ids.count(sid) > 1]
            raise ValueError(f"Duplicate section identifiers found in blueprint: {set(duplicates)}")

        # 2. Check question count sum
        sum_questions = sum(s.num_questions for s in self.sections)
        if sum_questions != self.total_questions:
            raise ValueError(
                f"Sum of section question counts ({sum_questions}) does not equal total_questions ({self.total_questions})."
            )

        # 3. Check section marks sum
        sum_marks = sum(s.section_marks for s in self.sections)
        if sum_marks != self.total_marks:
            raise ValueError(
                f"Sum of section marks ({sum_marks}) does not equal total_marks ({self.total_marks})."
            )

        # 4. Check difficulty distribution if provided
        if self.difficulty_distribution:
            sum_diff_q = sum(self.difficulty_distribution.values())
            if sum_diff_q != self.total_questions:
                raise ValueError(
                    f"Sum of questions in difficulty_distribution ({sum_diff_q}) does not equal total_questions ({self.total_questions})."
                )

        return self


def create_default_blueprint(
    num_questions: int,
    total_marks: int,
    question_types: list[QuestionType],
    overall_difficulty: Difficulty = Difficulty.MEDIUM,
    topic: str = "",
) -> ExamBlueprint:
    """
    Construct a valid, deterministic ExamBlueprint from basic paper parameters.
    Used when a request does not supply explicit sections or a blueprint.
    Guarantees that section counts sum to num_questions and section marks sum to total_marks.
    """
    if total_marks < num_questions:
        raise ValueError(
            f"total_marks ({total_marks}) must be at least num_questions ({num_questions}) "
            "because each question must have at least 1 mark."
        )

    # Dedup and preserve order of question types
    seen_types = []
    for qt in question_types:
        if qt not in seen_types:
            seen_types.append(qt)

    # Sort preferred order: MCQ first, then Short Answer, then Long Answer
    type_priority = {
        QuestionType.MULTIPLE_CHOICE: 0,
        QuestionType.SHORT_ANSWER: 1,
        QuestionType.LONG_ANSWER: 2,
    }
    ordered_types = sorted(seen_types, key=lambda t: type_priority.get(t, 99))

    # If more question types than questions, take only the first num_questions types
    active_types = ordered_types[:min(len(ordered_types), num_questions)]
    num_sections = len(active_types)

    # 1. Distribute questions across sections
    base_q = num_questions // num_sections
    rem_q = num_questions % num_sections
    section_q_counts = [base_q + (1 if i < rem_q else 0) for i in range(num_sections)]

    # 2. Distribute marks across sections
    allocated_marks = [q_cnt for q_cnt in section_q_counts]
    remaining_marks = total_marks - num_questions

    # Desired additional marks per question beyond 1
    desired_extra = {
        QuestionType.MULTIPLE_CHOICE: 0,
        QuestionType.SHORT_ANSWER: 1,
        QuestionType.LONG_ANSWER: 3,
    }

    # Pass 1: Try to satisfy desired weights from Long to Short to MCQ
    for pass_qt in [QuestionType.LONG_ANSWER, QuestionType.SHORT_ANSWER, QuestionType.MULTIPLE_CHOICE]:
        for idx, qt in enumerate(active_types):
            if qt == pass_qt and remaining_marks > 0:
                needed = desired_extra.get(qt, 0) * section_q_counts[idx]
                give = min(needed, remaining_marks)
                allocated_marks[idx] += give
                remaining_marks -= give

    # Pass 2: If remaining_marks still > 0, distribute to Long, then Short, then MCQ
    while remaining_marks > 0:
        distributed = False
        for pass_qt in [QuestionType.LONG_ANSWER, QuestionType.SHORT_ANSWER, QuestionType.MULTIPLE_CHOICE]:
            for idx, qt in enumerate(active_types):
                if qt == pass_qt and remaining_marks > 0:
                    allocated_marks[idx] += 1
                    remaining_marks -= 1
                    distributed = True
                    break
            if distributed:
                break

    # Build SectionBlueprint objects
    section_names = ["A", "B", "C", "D", "E"]
    title_map = {
        QuestionType.MULTIPLE_CHOICE: "Multiple Choice Questions",
        QuestionType.SHORT_ANSWER: "Short Answer Questions",
        QuestionType.LONG_ANSWER: "Long Answer Questions",
    }

    sections = []
    for i, qt in enumerate(active_types):
        sec_id = section_names[i] if i < len(section_names) else f"Sec-{i+1}"
        sec_title = f"Section {sec_id}: {title_map.get(qt, qt.value.replace('_', ' ').title())}"
        q_count = section_q_counts[i]
        sec_marks = allocated_marks[i]

        base_m = sec_marks // q_count
        rem_m = sec_marks % q_count
        dist = [base_m + (1 if k < rem_m else 0) for k in range(q_count)]
        marks_per_q = base_m if rem_m == 0 else None

        sections.append(
            SectionBlueprint(
                section_id=sec_id,
                title=sec_title,
                question_type=qt,
                num_questions=q_count,
                section_marks=sec_marks,
                marks_per_question=marks_per_q,
                marks_distribution=dist,
                difficulty=overall_difficulty,
                topics=[topic] if topic else None,
            )
        )

    return ExamBlueprint(
        title=f"Blueprint - {topic}" if topic else "Mathematics Examination Blueprint",
        sections=sections,
        total_questions=num_questions,
        total_marks=total_marks,
    )


# ---------------------------------------------------------------------------
# Paper generation request
# ---------------------------------------------------------------------------

class PaperGenerateRequest(BaseModel):
    """Request body for generating a question paper."""
    document_id: Optional[str] = Field(default="", description="Optional ID of document; defaults to auto-resolved textbook")
    grade: str = Field(..., min_length=1, description="Grade or class (e.g. '1', '2', '3', '4', '5')")
    subject: str = Field(default="Mathematics", description="Subject name")
    topic: Optional[str] = Field(default="Mathematics Curriculum", description="Chapter or topic to focus on")
    exam_type: ExamType = Field(
        default=ExamType.FINAL,
        description="Examination type: prerequisite, midterm, or final",
    )
    academic_grade: Optional[str] = Field(
        default=None,
        description="Academic grade/class enrolled by student (defaults to grade)",
    )
    curriculum_grade: Optional[str] = Field(
        default=None,
        description="Curriculum grade actually assessed",
    )
    syllabus_scope: Optional[str] = Field(
        default=None,
        description="Permitted syllabus scope/units (e.g. first half, specific units, or full syllabus)",
    )
    difficulty: Difficulty = Field(default=Difficulty.MEDIUM, description="Overall difficulty")
    question_types: list[QuestionType] = Field(
        default=[QuestionType.SHORT_ANSWER, QuestionType.LONG_ANSWER, QuestionType.MULTIPLE_CHOICE],
        description="Which question types to include",
    )
    num_questions: int = Field(..., gt=0, le=50, description="Number of questions to generate")
    total_marks: int = Field(..., gt=0, le=500, description="Desired total marks for the paper")
    num_versions: int = Field(default=1, ge=1, le=10, description="Number of paper versions to generate (1-10)")
    sections: Optional[list[SectionBlueprint]] = Field(default=None, description="Optional custom section blueprints")
    blueprint: Optional[ExamBlueprint] = Field(default=None, description="Optional complete exam blueprint")

    @field_validator("question_types")
    @classmethod
    def at_least_one_type(cls, v: list[QuestionType]) -> list[QuestionType]:
        if not v:
            raise ValueError("At least one question type must be specified.")
        return v

    @model_validator(mode="after")
    def validate_request_blueprint_consistency(self) -> "PaperGenerateRequest":
        if self.total_marks < self.num_questions:
            raise ValueError(
                f"total_marks ({self.total_marks}) cannot be less than num_questions ({self.num_questions})."
            )
        if self.blueprint is not None:
            if self.blueprint.total_questions != self.num_questions:
                raise ValueError(
                    f"Blueprint total_questions ({self.blueprint.total_questions}) does not match request num_questions ({self.num_questions})."
                )
            if self.blueprint.total_marks != self.total_marks:
                raise ValueError(
                    f"Blueprint total_marks ({self.blueprint.total_marks}) does not match request total_marks ({self.total_marks})."
                )
        elif self.sections is not None:
            sum_q = sum(s.num_questions for s in self.sections)
            if sum_q != self.num_questions:
                raise ValueError(
                    f"Sum of section num_questions ({sum_q}) does not match request num_questions ({self.num_questions})."
                )
            sum_m = sum(s.section_marks for s in self.sections)
            if sum_m != self.total_marks:
                raise ValueError(
                    f"Sum of section marks ({sum_m}) does not match request total_marks ({self.total_marks})."
                )
        return self

    def get_or_create_blueprint(self) -> ExamBlueprint:
        """Return the explicit blueprint or build a default one matching the request."""
        if self.blueprint is not None:
            return self.blueprint
        if self.sections is not None:
            return ExamBlueprint(
                title=f"{self.subject} - {self.topic} Blueprint",
                sections=self.sections,
                total_questions=self.num_questions,
                total_marks=self.total_marks,
            )
        return create_default_blueprint(
            num_questions=self.num_questions,
            total_marks=self.total_marks,
            question_types=self.question_types,
            overall_difficulty=self.difficulty,
            topic=self.topic,
        )


class BatchPaperGenerateRequest(BaseModel):
    """Request body for generating multiple unique paper versions."""
    document_id: Optional[str] = Field(default="", description="Optional ID of document; defaults to auto-resolved textbook")
    grade: str = Field(..., min_length=1, description="Grade or class (e.g. '1', '2', '3', '4', '5')")
    subject: str = Field(default="Mathematics", description="Subject name")
    topic: Optional[str] = Field(default="Mathematics Curriculum", description="Chapter or topic to focus on")
    exam_type: ExamType = Field(
        default=ExamType.FINAL,
        description="Examination type: prerequisite, midterm, or final",
    )
    academic_grade: Optional[str] = Field(
        default=None,
        description="Academic grade/class enrolled by student (defaults to grade)",
    )
    curriculum_grade: Optional[str] = Field(
        default=None,
        description="Curriculum grade actually assessed",
    )
    syllabus_scope: Optional[str] = Field(
        default=None,
        description="Permitted syllabus scope/units",
    )
    difficulty: Difficulty = Field(default=Difficulty.MEDIUM, description="Overall difficulty")
    num_versions: int = Field(default=2, ge=1, le=10, description="Number of unique paper versions to generate (1-10)")

    blueprint: Optional[ExamBlueprint] = Field(default=None, description="Explicit exam blueprint")
    sections: Optional[list[SectionBlueprint]] = Field(default=None, description="Explicit section blueprints")

    question_types: list[QuestionType] = Field(
        default=[QuestionType.SHORT_ANSWER, QuestionType.LONG_ANSWER, QuestionType.MULTIPLE_CHOICE],
        description="Which question types to include if blueprint is omitted",
    )
    num_questions: Optional[int] = Field(default=None, gt=0, le=50, description="Total questions per paper")
    total_marks: Optional[int] = Field(default=None, gt=0, le=500, description="Total marks per paper")

    @model_validator(mode="after")
    def validate_batch_blueprint(self) -> "BatchPaperGenerateRequest":
        if self.blueprint is not None:
            if self.num_questions is not None and self.blueprint.total_questions != self.num_questions:
                raise ValueError("Blueprint total_questions does not match request num_questions.")
            if self.total_marks is not None and self.blueprint.total_marks != self.total_marks:
                raise ValueError("Blueprint total_marks does not match request total_marks.")
            self.num_questions = self.blueprint.total_questions
            self.total_marks = self.blueprint.total_marks
        elif self.sections is not None:
            sum_q = sum(s.num_questions for s in self.sections)
            sum_m = sum(s.section_marks for s in self.sections)
            if self.num_questions is not None and sum_q != self.num_questions:
                raise ValueError(f"Sum of section questions ({sum_q}) does not match num_questions ({self.num_questions}).")
            if self.total_marks is not None and sum_m != self.total_marks:
                raise ValueError(f"Sum of section marks ({sum_m}) does not match total_marks ({self.total_marks}).")
            self.num_questions = sum_q
            self.total_marks = sum_m
        else:
            if self.num_questions is None or self.total_marks is None:
                raise ValueError("Must provide either 'blueprint', 'sections', or both 'num_questions' and 'total_marks'.")
            if self.total_marks < self.num_questions:
                raise ValueError(f"total_marks ({self.total_marks}) cannot be less than num_questions ({self.num_questions}).")
        return self

    def get_or_create_blueprint(self) -> ExamBlueprint:
        """Return the blueprint or construct a default blueprint."""
        if self.blueprint is not None:
            return self.blueprint
        if self.sections is not None:
            return ExamBlueprint(
                title=f"{self.subject} - {self.topic} Blueprint",
                sections=self.sections,
                total_questions=self.num_questions,  # type: ignore
                total_marks=self.total_marks,  # type: ignore
            )
        return create_default_blueprint(
            num_questions=self.num_questions,  # type: ignore
            total_marks=self.total_marks,  # type: ignore
            question_types=self.question_types,
            overall_difficulty=self.difficulty,
            topic=self.topic,
        )


# ---------------------------------------------------------------------------
# Paper and Section (output / stored)
# ---------------------------------------------------------------------------

class PaperSection(BaseModel):
    """A section of questions within an assembled paper."""
    section_id: str = Field(..., description="Section identifier, e.g. 'A'")
    title: str = Field(..., description="Section title")
    question_type: QuestionType = Field(..., description="Type of questions in this section")
    num_questions: int = Field(..., gt=0)
    section_marks: int = Field(..., gt=0)
    questions: list[Question] = Field(default_factory=list)


class Paper(BaseModel):
    """A complete generated question paper with metadata."""
    paper_id: str = Field(..., description="Unique identifier for this paper")
    title: str = Field(..., description="Paper title")
    grade: str
    subject: str
    topic: str
    exam_type: ExamType = Field(
        default=ExamType.FINAL,
        description="Examination type: prerequisite, midterm, or final",
    )
    academic_grade: Optional[str] = Field(
        default=None,
        description="Academic class of student",
    )
    curriculum_grade: Optional[str] = Field(
        default=None,
        description="Curriculum class syllabus used",
    )
    syllabus_scope: Optional[str] = Field(
        default=None,
        description="Applied syllabus scope or units",
    )
    difficulty: Difficulty
    created_at: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="ISO-format creation timestamp",
    )
    questions: list[Question]
    num_questions: int = Field(..., gt=0)
    total_marks: int = Field(..., gt=0)
    document_id: str = Field(default="", description="Source document ID")

    # Multi-version and blueprint metadata
    sections: list[PaperSection] = Field(default_factory=list, description="Sections in this paper")
    batch_id: Optional[str] = Field(default=None, description="Batch ID if part of multi-version generation")
    version_number: int = Field(default=1, ge=1, le=10, description="Version number within batch (1-10)")
    num_versions: int = Field(default=1, ge=1, le=10, description="Total versions in batch")
    blueprint_id: Optional[str] = Field(default=None, description="Blueprint identifier")

    @model_validator(mode="after")
    def check_counts_and_marks(self) -> "Paper":
        """Verify question count, total marks, and section consistency."""
        actual_count = len(self.questions)
        if actual_count != self.num_questions:
            raise ValueError(
                f"num_questions is {self.num_questions} but there are {actual_count} questions."
            )
        actual_marks = sum(q.marks for q in self.questions)
        if actual_marks != self.total_marks:
            raise ValueError(
                f"total_marks is {self.total_marks} but sum of question marks is {actual_marks}."
            )

        # Ensure question IDs are unique within the paper
        q_ids = [q.question_id for q in self.questions]
        if len(q_ids) != len(set(q_ids)):
            raise ValueError("Paper contains duplicate question IDs.")

        # If sections are defined, ensure strict consistency
        if self.sections:
            sec_q_count = sum(len(s.questions) for s in self.sections)
            if sec_q_count != actual_count:
                raise ValueError(
                    f"Total questions across sections ({sec_q_count}) does not match paper question count ({actual_count})."
                )
            sec_marks_count = sum(s.section_marks for s in self.sections)
            if sec_marks_count != self.total_marks:
                raise ValueError(
                    f"Total marks across sections ({sec_marks_count}) does not match paper total_marks ({self.total_marks})."
                )
            for s in self.sections:
                if len(s.questions) != s.num_questions:
                    raise ValueError(
                        f"Section '{s.section_id}' expected {s.num_questions} questions, but has {len(s.questions)}."
                    )
                sum_sec_q_marks = sum(q.marks for q in s.questions)
                if sum_sec_q_marks != s.section_marks:
                    raise ValueError(
                        f"Section '{s.section_id}' expected {s.section_marks} marks, but question marks sum to {sum_sec_q_marks}."
                    )
                for q in s.questions:
                    if q.question_type != s.question_type:
                        raise ValueError(
                            f"Question '{q.question_id}' in section '{s.section_id}' has type {q.question_type}, "
                            f"expected {s.question_type}."
                        )
        return self


# ---------------------------------------------------------------------------
# Batch response models
# ---------------------------------------------------------------------------

class BatchValidationSummary(BaseModel):
    """Summary of candidate pool and uniqueness validation across versions."""
    total_candidates_generated: int = Field(..., description="Total candidate questions generated across attempts")
    valid_candidates_count: int = Field(..., description="Candidates that passed schema and constraint validation")
    duplicate_candidates_rejected: int = Field(..., description="Duplicate or near-duplicate candidates rejected")
    math_verified_count: int = Field(default=0, description="Questions confirmed by SymPy mathematical checking")
    math_discrepancies_rejected: int = Field(default=0, description="Candidates rejected due to mathematical discrepancies")
    is_unique: bool = Field(..., description="Whether all selected questions across versions are distinct")
    all_versions_valid: bool = Field(..., description="Whether all versions satisfy blueprint constraints")
    details: str = Field(default="", description="Additional verification or summary details")


class BatchPaperResponse(BaseModel):
    """Response returned by POST /papers/generate-batch."""
    batch_id: str = Field(..., description="Unique identifier for this generation batch")
    num_versions_requested: int = Field(..., description="Number of versions requested")
    num_versions_generated: int = Field(..., description="Number of versions successfully generated")
    blueprint: ExamBlueprint = Field(..., description="The shared blueprint followed by all versions")
    papers: list[Paper] = Field(..., description="The generated paper objects")
    paper_ids: list[str] = Field(..., description="List of paper IDs for retrieval and PDF export")
    validation_summary: BatchValidationSummary = Field(..., description="Validation and uniqueness summary")
    created_at: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="ISO-format creation timestamp",
    )


# ---------------------------------------------------------------------------
# Document processing response
# ---------------------------------------------------------------------------

class DocumentInfo(BaseModel):
    """Returned after a document is uploaded and processed."""
    document_id: str
    filename: str
    num_chunks: int
    message: str


class DocumentRecord(BaseModel):
    """Full record of an uploaded document, persisted as JSON.
    Tracks everything needed to trace questions back to their source."""
    document_id: str = Field(..., description="Unique identifier for this document")
    original_filename: str = Field(..., description="User's original filename")
    safe_filename: str = Field(
        default="",
        description="Safe filename used for storage on disk",
    )
    grade: Optional[str] = Field(
        default=None,
        description="Academic grade or class this document belongs to (e.g. '10', '9')",
    )
    subject: Optional[str] = Field(
        default="Mathematics",
        description="Subject of this syllabus document",
    )
    units: Optional[list[str]] = Field(
        default=None,
        description="Identified chapters or units in the document",
    )
    syllabus_scope: Optional[str] = Field(
        default=None,
        description="Identified syllabus scope",
    )
    num_chunks: int = Field(..., ge=0, description="Number of text chunks created")
    num_pages: Optional[int] = Field(
        default=None,
        description="Number of pages (PDF) or paragraphs (DOCX) if available",
    )
    file_size_bytes: int = Field(
        default=0, ge=0, description="Size of the uploaded file in bytes"
    )
    uploaded_at: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="ISO-format timestamp of when the document was uploaded",
    )
    has_index: bool = Field(
        default=False,
        description="Whether a FAISS index has been built for this document",
    )
    content_hash: Optional[str] = Field(
        default=None,
        description="SHA-256 hash of the source file to detect modifications",
    )
    text_preview: str = Field(
        default="",
        description="First ~300 characters of extracted text for quick reference",
    )


class DocumentSummary(BaseModel):
    """Lightweight summary used when listing documents."""
    document_id: str
    original_filename: str
    num_chunks: int
    file_size_bytes: int
    uploaded_at: str
    has_index: bool
    grade: Optional[str] = None
    subject: Optional[str] = "Mathematics"
    units: Optional[list[str]] = None


class TextDocumentRequest(BaseModel):
    """Payload for submitting syllabus content directly as plain text."""
    title: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Title, chapter, or reference name for this text syllabus",
    )
    content: str = Field(
        ...,
        min_length=10,
        description="Syllabus, notes, or textbook chapter text content",
    )
    grade: Optional[str] = Field(
        default=None,
        description="Grade or class (e.g. '10', '9', 'Class 8')",
    )
    subject: Optional[str] = Field(
        default="Mathematics",
        description="Subject name",
    )


# ---------------------------------------------------------------------------
# Paper summary (for listing)
# ---------------------------------------------------------------------------

class PaperSummary(BaseModel):
    """Lightweight summary used when listing papers."""
    paper_id: str
    title: str
    grade: str
    subject: str
    topic: str
    exam_type: ExamType = ExamType.FINAL
    academic_grade: Optional[str] = None
    curriculum_grade: Optional[str] = None
    syllabus_scope: Optional[str] = None
    difficulty: Difficulty
    num_questions: int
    total_marks: int
    created_at: str
    version_number: int = 1
    num_versions: int = 1
    batch_id: Optional[str] = None
