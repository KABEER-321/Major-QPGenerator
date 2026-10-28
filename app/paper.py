# app/paper.py
# Paper assembly, JSON persistence, and blueprint checks.
# Papers are saved as JSON files under data/papers/ with a unique ID.

import json
import logging
import tempfile
import uuid
from datetime import datetime
from pathlib import Path

from typing import Optional
from app.config import settings
from app.schemas import (
    Paper,
    PaperSummary,
    Question,
    QuestionType,
    PaperGenerateRequest,
    PaperSection,
    ExamType,
    format_grade_display,
)

logger = logging.getLogger(__name__)


def assemble_paper(
    request: PaperGenerateRequest,
    questions: list[Question],
    sections: Optional[list[PaperSection]] = None,
    batch_id: Optional[str] = None,
    version_number: int = 1,
    num_versions: int = 1,
    blueprint_id: Optional[str] = None,
) -> Paper:
    """
    Assemble validated questions into a Paper object.
    The Paper validator will check that counts and marks are consistent.
    Also ensures each question carries the request's grade, subject, and topic
    as context (in case these were not already populated by the generator).
    """
    paper_id = uuid.uuid4().hex[:12]
    academic_class = request.academic_grade or request.grade
    curriculum_class = request.curriculum_grade or request.grade
    exam_type = request.exam_type or ExamType.FINAL

    grade_display = format_grade_display(academic_class)
    version_suffix = f" (Version {version_number})" if num_versions > 1 else ""
    title = f"{grade_display} {request.subject} — {exam_type.display_name}{version_suffix}"

    # Ensure every question has its context fields filled
    permitted = getattr(request, "permitted_units", None)
    for q in questions:
        if not q.grade:
            q.grade = academic_class
        if not q.subject or q.subject == "Mathematics":
            q.subject = request.subject
        if not q.chapter or q.chapter in ("Mathematics Curriculum", "General Math", "Mathematics"):
            q.chapter = (permitted[0] if permitted else request.topic) or "Mathematics"

    # If sections not provided, construct them using the request blueprint or group by question type
    if not sections:
        built_sections = []
        if hasattr(request, "get_or_create_blueprint"):
            try:
                bp = request.get_or_create_blueprint()
                cursor = 0
                candidate_sections = []
                type_mismatch = False
                for sec_bp in bp.sections:
                    sec_qs = questions[cursor : cursor + sec_bp.num_questions]
                    cursor += sec_bp.num_questions
                    if any(q.question_type != sec_bp.question_type for q in sec_qs):
                        type_mismatch = True
                        break
                    candidate_sections.append(
                        PaperSection(
                            section_id=sec_bp.section_id,
                            title=sec_bp.title,
                            question_type=sec_bp.question_type,
                            num_questions=len(sec_qs),
                            section_marks=sum(q.marks for q in sec_qs),
                            marks_per_question=sec_bp.marks_per_question,
                            instructions=getattr(sec_bp, "instructions", None),
                            questions=sec_qs,
                        )
                    )
                if not type_mismatch and candidate_sections:
                    if (
                        sum(len(s.questions) for s in candidate_sections) == len(questions)
                        and sum(s.section_marks for s in candidate_sections) == request.total_marks
                    ):
                        built_sections = candidate_sections
                        blueprint_id = blueprint_id or bp.blueprint_id
            except Exception as e:
                logger.debug("Blueprint-based sectioning omitted: %s", e)

        # Fallback: Group questions by their actual question type (MCQ -> Short Answer -> Long Answer)
        if not built_sections and questions:
            type_order = [
                QuestionType.MULTIPLE_CHOICE,
                QuestionType.SHORT_ANSWER,
                QuestionType.LONG_ANSWER,
            ]
            title_map = {
                QuestionType.MULTIPLE_CHOICE: "Multiple Choice Questions",
                QuestionType.SHORT_ANSWER: "Short Answer Questions",
                QuestionType.LONG_ANSWER: "Long Answer / Problem Solving Questions",
            }
            sec_letters = ["A", "B", "C", "D", "E"]
            letter_idx = 0

            # Stable sort/group by question type order
            for qt in type_order:
                matching_qs = [q for q in questions if q.question_type == qt]
                if matching_qs:
                    sid = sec_letters[letter_idx] if letter_idx < len(sec_letters) else f"Sec-{letter_idx+1}"
                    built_sections.append(
                        PaperSection(
                            section_id=sid,
                            title=f"Section {sid}: {title_map.get(qt, qt.value.title())}",
                            question_type=qt,
                            num_questions=len(matching_qs),
                            section_marks=sum(q.marks for q in matching_qs),
                            questions=matching_qs,
                        )
                    )
                    letter_idx += 1

        if built_sections:
            sections = built_sections

    distinct_covered = []
    ch_marks_dist: dict[str, int] = {}
    ch_qs_dist: dict[str, int] = {}
    for q in questions:
        c_name = q.chapter or q.topic or "Unknown"
        if permitted:
            for pu in permitted:
                if pu.lower() in c_name.lower() or c_name.lower() in pu.lower():
                    c_name = pu
                    break
        if c_name not in distinct_covered:
            distinct_covered.append(c_name)
        ch_marks_dist[c_name] = ch_marks_dist.get(c_name, 0) + q.marks
        ch_qs_dist[c_name] = ch_qs_dist.get(c_name, 0) + 1

    coverage_explanation = None
    if hasattr(request, "get_or_create_blueprint"):
        try:
            bp_obj = request.get_or_create_blueprint()
            coverage_explanation = getattr(bp_obj, "coverage_explanation", None)
        except Exception:
            pass

    paper = Paper(
        paper_id=paper_id,
        title=title,
        grade=academic_class,
        subject=request.subject,
        topic=request.topic,
        exam_type=exam_type,
        academic_grade=academic_class,
        curriculum_grade=curriculum_class,
        syllabus_scope=request.syllabus_scope,
        permitted_units=getattr(request, "permitted_units", None),
        forbidden_units=getattr(request, "forbidden_units", None),
        covered_chapters=distinct_covered,
        coverage_explanation=coverage_explanation,
        chapter_marks_distribution=ch_marks_dist,
        chapter_questions_distribution=ch_qs_dist,
        difficulty=request.difficulty,
        questions=questions,
        num_questions=request.num_questions,
        total_marks=request.total_marks,
        document_id=request.document_id,
        sections=sections or [],
        batch_id=batch_id,
        version_number=version_number,
        num_versions=num_versions,
        blueprint_id=blueprint_id,
    )
    return paper


def save_paper(paper: Paper) -> Path:
    """
    Save a paper as a JSON file using atomic write.
    Returns the path to the saved file.
    """
    file_path = settings.papers_dir / f"{paper.paper_id}.json"

    # Atomic write: write to a temp file first, then rename.
    # This reduces the risk of corrupted files if the process is interrupted.
    tmp_path: Optional[Path] = None
    try:
        dir_path = file_path.parent
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".json",
            dir=str(dir_path),
            delete=False,
            encoding="utf-8",
        ) as tmp:
            tmp.write(paper.model_dump_json(indent=2))
            tmp_path = Path(tmp.name)
        tmp_path.replace(file_path)
    except Exception:
        # Clean up temp file if rename failed
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise

    logger.info("Saved paper %s to %s", paper.paper_id, file_path)
    return file_path


def load_paper(paper_id: str) -> Paper:
    """
    Load a paper from its JSON file.
    Raises FileNotFoundError if it doesn't exist.
    Raises ValueError if the file is malformed.
    """
    # Prevent path traversal by sanitizing the paper_id
    safe_id = Path(paper_id).name
    file_path = settings.papers_dir / f"{safe_id}.json"

    if not file_path.exists():
        raise FileNotFoundError(f"Paper '{paper_id}' not found.")

    try:
        raw = file_path.read_text(encoding="utf-8")
        data = json.loads(raw)
        return Paper.model_validate(data)
    except (json.JSONDecodeError, Exception) as e:
        raise ValueError(f"Paper file '{paper_id}' is malformed: {e}")


def list_papers() -> list[PaperSummary]:
    """
    List all saved papers as lightweight summaries.
    Skips files that cannot be parsed.
    """
    summaries = []
    for file_path in sorted(settings.papers_dir.glob("*.json")):
        try:
            raw = file_path.read_text(encoding="utf-8")
            data = json.loads(raw)
            summaries.append(
                PaperSummary(
                    paper_id=data["paper_id"],
                    title=data["title"],
                    grade=data["grade"],
                    subject=data["subject"],
                    topic=data["topic"],
                    exam_type=data.get("exam_type", ExamType.FINAL),
                    academic_grade=data.get("academic_grade"),
                    curriculum_grade=data.get("curriculum_grade"),
                    syllabus_scope=data.get("syllabus_scope"),
                    difficulty=data["difficulty"],
                    num_questions=data["num_questions"],
                    total_marks=data["total_marks"],
                    created_at=data.get("created_at", ""),
                    version_number=data.get("version_number", 1),
                    num_versions=data.get("num_versions", 1),
                    batch_id=data.get("batch_id"),
                )
            )
        except Exception as e:
            logger.warning("Skipping malformed paper file %s: %s", file_path.name, e)
    return summaries


def delete_paper(paper_id: str) -> bool:
    """
    Delete a saved paper JSON file.
    Returns True if deleted, False if it did not exist.
    """
    safe_id = Path(paper_id).name
    file_path = settings.papers_dir / f"{safe_id}.json"
    if file_path.exists():
        file_path.unlink()
        logger.info("Deleted paper file %s", file_path)
        return True
    return False

