# app/custom_syllabus.py
# Custom (teacher-defined) syllabus support.
#
# Pipeline (reuses the existing document/RAG stack):
#   1. Text comes from an uploaded PDF/DOCX (documents.extract_text) or manual entry.
#   2. parse_syllabus() detects chapters/units and their topics.
#   3. Chunks are tagged with exact chapter metadata and embedded into a FAISS
#      index (retrieval.build_faiss_index) so retrieval can be restricted strictly
#      to the chapters the teacher selected.
#   4. The syllabus is saved as JSON under data/custom_syllabi/ for later reuse.

from __future__ import annotations

import json
import logging
import re
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


from app.config import settings
from app.documents import split_text_into_chunks
from app.retrieval import build_faiss_index
from app.schemas import (
    CustomSyllabus,
    CustomSyllabusChapter,
    CustomSyllabusChapterOut,
)

logger = logging.getLogger(__name__)

MAX_CHAPTERS = 60
MAX_TOPICS_PER_CHAPTER = 60
MAX_TOPIC_LENGTH = 160

_KEYWORD_HEADING = re.compile(
    r"^\s*(?:unit|chapter|module|lesson|part|section)\s*(?:no\.?\s*)?([0-9]+|[ivxlcdm]+)\b\s*[:.\-–—)]*\s*(.*)$",
    re.IGNORECASE,
)
_NUMBERED_HEADING = re.compile(r"^\s*(\d{1,2})\s*[.)]\s+(\S.{1,150})$")
_SUB_NUMBERED = re.compile(r"^\s*\d{1,2}\.\d{1,2}(?:\.\d{1,2})*\s*[.)\-:]?\s*")
_BULLET = re.compile(r"^\s*(?:[•●▪■◦○\-–—*·]|\(?[a-zA-Z]\)|\(?[ivx]{1,4}\))\s+")
_TOPICS_PREFIX = re.compile(r"^\s*(?:topics?|sub-?topics?|contents?|concepts?|includes?)\s*[:\-–—]\s*", re.IGNORECASE)
_PAGE_NUMBER = re.compile(r"^\s*(?:page\s*)?\d{1,4}\s*$", re.IGNORECASE)


@dataclass
class ParsedChapter:
    title: str
    topics: list[str] = field(default_factory=list)
    body_lines: list[str] = field(default_factory=list)

    @property
    def body(self) -> str:
        return "\n".join(self.body_lines).strip()


def _clean_line(line: str) -> str:
    return " ".join(line.replace("\u00a0", " ").split())


def _is_keyword_heading(line: str) -> Optional[re.Match]:
    m = _KEYWORD_HEADING.match(line)
    if not m:
        return None
    # Avoid treating ordinary sentences starting with "Part of ..." as headings:
    # the identifier group must be a number or a roman numeral token.
    return m


def _split_topic_line(line: str) -> list[str]:
    """Turn one syllabus line into one or more topic strings."""
    text = _clean_line(line)
    if not text or _PAGE_NUMBER.match(text):
        return []
    had_prefix = bool(_TOPICS_PREFIX.match(text))
    text = _TOPICS_PREFIX.sub("", text)
    text = _SUB_NUMBERED.sub("", text)
    text = _BULLET.sub("", text).strip(" -–—•*·;,")
    if not text:
        return []
    if had_prefix or ";" in text:
        parts = re.split(r"[;,]", text) if had_prefix else re.split(r";", text)
    else:
        parts = [text]
    return [p.strip(" .-–—") for p in parts if len(p.strip(" .-–—")) >= 2]


def parse_syllabus(text: str) -> list[ParsedChapter]:
    """
    Detect chapters/units and topics in raw syllabus text.

    Strategy: (1) keyword headings ('Unit 1: ...', 'Chapter 2 - ...'); else
    (2) top-level numbered headings ('1. Real Numbers'); else (3) a single
    chapter that holds every non-empty line as a topic.
    """
    lines = [l for l in (text or "").splitlines()]
    cleaned = [_clean_line(l) for l in lines]

    def collect(heading_fn) -> list[ParsedChapter]:
        chapters: list[ParsedChapter] = []
        current: Optional[ParsedChapter] = None
        pending_title_for: Optional[ParsedChapter] = None
        for raw, line in zip(lines, cleaned):
            if not line:
                continue
            title = heading_fn(line)
            if title is not None:
                current = ParsedChapter(title=title[0], body_lines=[line])
                chapters.append(current)
                if title[1]:  # heading carries an empty title: take next line
                    pending_title_for = current
                continue
            if current is None:
                continue
            if pending_title_for is current and not current.topics and len(line) <= 120:
                current.title = f"{current.title} {line}".strip()
                current.body_lines[0] = current.title
                pending_title_for = None
                continue
            pending_title_for = None
            current.body_lines.append(line)
            for t in _split_topic_line(raw):
                if len(t) <= MAX_TOPIC_LENGTH:
                    current.topics.append(t)
        return chapters

    def keyword_fn(line: str):
        m = _is_keyword_heading(line)
        if not m:
            return None
        label = line.split(m.group(1), 1)[0].strip()  # e.g. 'Unit'
        ident = m.group(1)
        rest = m.group(2).strip(" :.-–—")
        if len(line) > 200:
            return None
        if rest:
            return (f"{label.title()} {ident}: {rest}" if label else rest, False)
        return (f"{label.title()} {ident}", True)

    def numbered_fn(line: str):
        m = _NUMBERED_HEADING.match(line)
        if not m or _SUB_NUMBERED.match(line):
            return None
        title = m.group(2).strip(" :.-–—")
        if len(title) > 100 or title.endswith((",", ";")):
            return None
        return (f"{int(m.group(1))}. {title}", False)

    chapters = collect(keyword_fn)
    if not chapters:
        chapters = collect(numbered_fn)
        # a lone numbered line is far more likely to be a list item than a chapter
        if len(chapters) < 2:
            chapters = []

    if not chapters:
        body = [l for l in cleaned if l]
        topics: list[str] = []
        for raw in lines:
            topics.extend(t for t in _split_topic_line(raw) if len(t) <= MAX_TOPIC_LENGTH)
        chapters = [ParsedChapter(title="Custom Syllabus", topics=topics, body_lines=body)]

    # Normalise: unique titles, unique topics, caps
    seen_titles: dict[str, int] = {}
    result: list[ParsedChapter] = []
    for ch in chapters[:MAX_CHAPTERS]:
        key = ch.title.lower()
        seen_titles[key] = seen_titles.get(key, 0) + 1
        if seen_titles[key] > 1:
            ch.title = f"{ch.title} ({seen_titles[key]})"
        uniq: list[str] = []
        seen_topics: set[str] = set()
        for t in ch.topics:
            if t.lower() not in seen_topics and t.lower() != ch.title.lower():
                seen_topics.add(t.lower())
                uniq.append(t)
        ch.topics = uniq[:MAX_TOPICS_PER_CHAPTER]
        result.append(ch)
    return result


def to_chapters(parsed: list[ParsedChapter]) -> list[CustomSyllabusChapter]:
    return [CustomSyllabusChapter(title=p.title, topics=p.topics, content=p.body) for p in parsed]


# ---------------------------------------------------------------------------
# Indexing (reuses existing chunking + FAISS)
# ---------------------------------------------------------------------------

def build_chapter_chunks(
    chapters: list[CustomSyllabusChapter],
    syllabus_id: str,
    name: str,
    grade: Optional[str],
    subject: str,
) -> list[dict]:
    """Chunk every chapter separately and tag chunks with exact chapter metadata."""
    chunks: list[dict] = []
    for ch_idx, ch in enumerate(chapters):
        header = f"{ch.title}\n"
        if ch.topics:
            header += "Topics: " + "; ".join(ch.topics) + "\n"
        text = header
        if ch.content and ch.content.strip() not in (ch.title, ""):
            text = header + "\n" + ch.content
        pieces = split_text_into_chunks(
            text, source_filename=f"{name} - {ch.title}", grade=grade, subject=subject
        )
        for piece in pieces:
            meta = piece["metadata"]
            meta.update({
                "doc_id": syllabus_id,
                "chapter": ch.title,
                "unit": ch.title,
                "chapter_index": ch_idx,
                "custom_syllabus": True,
                "is_first_half": True,
                "chunk_id": f"{syllabus_id}_ch{ch_idx}_k{len(chunks)}",
            })
            meta["chunk_index"] = len(chunks)
            chunks.append(piece)
    return chunks


def _index_syllabus(syl: CustomSyllabus) -> int:
    chunks = build_chapter_chunks(syl.chapters, syl.syllabus_id, syl.name, syl.grade, syl.subject)
    if not chunks:
        raise ValueError("The syllabus has no usable content to index.")
    build_faiss_index(chunks, syl.syllabus_id)
    return len(chunks)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def _syllabus_path(syllabus_id: str) -> Path:
    return settings.custom_syllabi_dir / f"{Path(syllabus_id).name}.json"


def save_custom_syllabus(syl: CustomSyllabus) -> Path:
    path = _syllabus_path(syl.syllabus_id)
    tmp: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", dir=str(path.parent), delete=False, encoding="utf-8"
        ) as f:
            f.write(syl.model_dump_json(indent=2))
            tmp = Path(f.name)
        tmp.replace(path)
    except Exception:
        if tmp is not None and tmp.exists():
            tmp.unlink(missing_ok=True)
        raise
    return path


def load_custom_syllabus(syllabus_id: str, owner_id: Optional[str] = None) -> CustomSyllabus:
    """Load a saved syllabus. Raises FileNotFoundError if missing or owned by someone else."""
    path = _syllabus_path(syllabus_id)
    if not path.exists():
        raise FileNotFoundError(f"Custom syllabus '{syllabus_id}' not found.")
    try:
        syl = CustomSyllabus.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception as e:
        raise ValueError(f"Custom syllabus '{syllabus_id}' is malformed: {e}")
    if owner_id is not None and syl.owner_id and syl.owner_id != owner_id:
        raise FileNotFoundError(f"Custom syllabus '{syllabus_id}' not found.")
    return syl


def list_custom_syllabi(owner_id: Optional[str] = None) -> list[CustomSyllabus]:
    out: list[CustomSyllabus] = []
    for p in sorted(settings.custom_syllabi_dir.glob("*.json")):
        try:
            syl = CustomSyllabus.model_validate_json(p.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning("Skipping malformed custom syllabus %s: %s", p.name, e)
            continue
        if owner_id is not None and syl.owner_id and syl.owner_id != owner_id:
            continue
        out.append(syl)
    out.sort(key=lambda s: s.created_at, reverse=True)
    return out


def delete_custom_syllabus(syllabus_id: str, owner_id: Optional[str] = None) -> bool:
    import shutil
    try:
        syl = load_custom_syllabus(syllabus_id, owner_id)
    except (FileNotFoundError, ValueError):
        return False
    _syllabus_path(syl.syllabus_id).unlink(missing_ok=True)
    idx = settings.indexes_dir / Path(syl.syllabus_id).name
    if idx.exists():
        shutil.rmtree(idx, ignore_errors=True)
    if syl.source_filename:
        for ext in (".pdf", ".docx"):
            (settings.uploads_dir / f"{syl.syllabus_id}{ext}").unlink(missing_ok=True)
    return True


# ---------------------------------------------------------------------------
# Create / update
# ---------------------------------------------------------------------------

def create_custom_syllabus(
    *,
    name: str,
    text: str,
    owner_id: str,
    grade: Optional[str] = None,
    subject: str = "Mathematics",
    source_type: str = "text",
    source_filename: Optional[str] = None,
    syllabus_id: Optional[str] = None,
) -> CustomSyllabus:
    """Parse text into chapters/topics, index it, and persist the record."""
    if not text or not text.strip():
        raise ValueError("Syllabus content cannot be empty.")
    parsed = parse_syllabus(text)
    chapters = to_chapters(parsed)
    if not chapters:
        raise ValueError("No chapters or topics could be extracted from the syllabus.")
    syl = CustomSyllabus(
        syllabus_id=syllabus_id or f"custom_{uuid.uuid4().hex[:12]}",
        owner_id=owner_id,
        name=name.strip(),
        grade=grade,
        subject=subject or "Mathematics",
        source_type=source_type,
        source_filename=source_filename,
        chapters=chapters,
    )
    syl.num_chunks = _index_syllabus(syl)
    syl.has_index = True
    save_custom_syllabus(syl)
    return syl


def update_custom_syllabus(
    syl: CustomSyllabus,
    name: Optional[str] = None,
    grade: Optional[str] = None,
    chapters: Optional[list[CustomSyllabusChapterOut]] = None,
) -> CustomSyllabus:
    """Rename or edit the chapter/topic structure; re-indexes if structure changed."""
    from datetime import datetime

    if name is not None:
        syl.name = name.strip()
    if grade is not None:
        syl.grade = grade or None
    if chapters is not None:
        if not chapters:
            raise ValueError("A syllabus must keep at least one chapter.")
        old_content = {c.title.strip().lower(): c.content for c in syl.chapters}
        new: list[CustomSyllabusChapter] = []
        seen: set[str] = set()
        for c in chapters:
            title = _clean_line(c.title)
            if not title:
                continue
            if title.lower() in seen:
                raise ValueError(f"Duplicate chapter title '{title}'.")
            seen.add(title.lower())
            topics, ts = [], set()
            for t in c.topics:
                t = _clean_line(t)
                if t and t.lower() not in ts:
                    ts.add(t.lower())
                    topics.append(t)
            new.append(CustomSyllabusChapter(
                title=title, topics=topics, content=old_content.get(title.lower(), "")
            ))
        if not new:
            raise ValueError("A syllabus must keep at least one chapter.")
        syl.chapters = new
    syl.updated_at = datetime.now().isoformat()
    if chapters is not None or name is not None or grade is not None:
        syl.num_chunks = _index_syllabus(syl)
        syl.has_index = True
    save_custom_syllabus(syl)
    return syl


# ---------------------------------------------------------------------------
# Selection resolution for question-paper generation
# ---------------------------------------------------------------------------

class SelectionError(ValueError):
    """Raised when the chapter/topic selection is empty or does not match the syllabus."""


def resolve_selection(
    syl: CustomSyllabus,
    selected_chapters: Optional[list[str]],
    selected_topics: Optional[dict[str, list[str]]],
) -> tuple[list[str], dict[str, list[str]]]:
    """
    Validate the teacher's selection against the saved syllabus.
    Returns (ordered chapter titles, {chapter: [topics]}) where an empty topic
    list means 'all topics of the chapter'.
    """
    by_lower = {c.title.strip().lower(): c for c in syl.chapters}
    requested = list(selected_chapters or [])
    if not requested and selected_topics:
        requested = list(selected_topics.keys())
    if not requested:
        raise SelectionError("Select at least one chapter from the custom syllabus.")

    chosen: list[str] = []
    for title in requested:
        ch = by_lower.get(title.strip().lower())
        if ch is None:
            raise SelectionError(f"Chapter '{title}' is not part of custom syllabus '{syl.name}'.")
        if ch.title not in chosen:
            chosen.append(ch.title)
    chosen.sort(key=lambda t: [c.title for c in syl.chapters].index(t))

    topics_out: dict[str, list[str]] = {}
    for title in chosen:
        ch = by_lower[title.strip().lower()]
        wanted = None
        for k, v in (selected_topics or {}).items():
            if k.strip().lower() == title.strip().lower():
                wanted = v
        if not wanted:
            topics_out[title] = []
            continue
        valid = {t.strip().lower(): t for t in ch.topics}
        picked: list[str] = []
        for t in wanted:
            real = valid.get(t.strip().lower())
            if real is None:
                raise SelectionError(f"Topic '{t}' is not part of chapter '{title}'.")
            if real not in picked:
                picked.append(real)
        # all topics picked is the same as no restriction
        topics_out[title] = [] if len(picked) == len(ch.topics) else picked
    return chosen, topics_out


def build_scope_text(syl_name: str, chapters: list[str], topics: dict[str, list[str]], limit: int = 1800) -> str:
    """Human/LLM-readable scope statement used in prompts, paper metadata and PDFs."""
    parts = []
    for ch in chapters:
        tl = topics.get(ch) or []
        parts.append(f"{ch} (topics: {', '.join(tl)})" if tl else ch)
    text = (
        f"Custom syllabus '{syl_name}' - ONLY these selected chapters/topics, nothing outside them: "
        + "; ".join(parts)
    )
    return text if len(text) <= limit else text[: limit - 3] + "..."
