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
from app.schemas import Paper, PaperSummary, Question, PaperGenerateRequest, PaperSection

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
    version_suffix = f" (Version {version_number})" if num_versions > 1 else ""
    title = f"{request.subject} - {request.topic} ({request.grade}){version_suffix}"

    # Ensure every question has its context fields filled
    for q in questions:
        if not q.grade:
            q.grade = request.grade
        if not q.subject or q.subject == "Mathematics":
            q.subject = request.subject
        if not q.chapter:
            q.chapter = request.topic

    paper = Paper(
        paper_id=paper_id,
        title=title,
        grade=request.grade,
        subject=request.subject,
        topic=request.topic,
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

