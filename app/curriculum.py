# app/curriculum.py
"""
Permanent curriculum textbook management for Classes 1 through 5.
Stores textbooks permanently in data/curriculum/class_{N}/textbook.pdf.
Extracts chapters based on textbook structure, chunks with chapter metadata,
and manages persistent FAISS indexes.
"""

from __future__ import annotations

import hashlib
import logging
import math
from pathlib import Path
from typing import Any, Optional

import fitz  # PyMuPDF
import re
from langchain_core.documents import Document

from app.config import settings
from app.documents_store import (
    DocumentRecord,
    load_document_record,
    save_document_record,
)
from app.retrieval import build_faiss_index
from app.schemas import parse_grade_number

logger = logging.getLogger(__name__)


def clean_text(text: str) -> str:
    """Normalize whitespace and strip unprintable control characters."""
    if not text:
        return ""
    return re.sub(r"[ \t]+", " ", text).strip()

# Complete chapter tables of contents for Classes 1 to 5
CURRICULUM_CHAPTERS: dict[int, list[str]] = {
    1: [
        "Lesson 1: Spatial Understanding",
        "Lesson 2: Solids Around Us",
        "Lesson 3: Digits (1-9)",
        "Lesson 4: Zero",
        "Lesson 5: Addition (Sum not more than 9)",
        "Lesson 6: Subtraction",
        "Lesson 7: Number 10",
        "Lesson 8: Units and Tens",
        "Lesson 9: Number 11 to 20",
    ],
    2: [
        "Lesson 1: Shapes and Spatial Understanding",
        "Lesson 2: Numbers",
        "Lesson 3: Addition",
        "Lesson 4: Subtraction",
        "Lesson 5: Multiplication",
        "Lesson 6: Division",
    ],
    3: [
        "Chapter 1: Shapes",
        "Chapter 2: Numbers",
        "Chapter 3: Addition",
        "Chapter 4: Subtraction",
        "Chapter 5: Multiplication",
    ],
    4: [
        "Chapter 1: Perimeter and Area of simple Geometrical figures",
        "Chapter 2: Numbers",
        "Chapter 3: Addition",
        "Chapter 4: Subtraction",
        "Chapter 5: Multiplication",
        "Chapter 6: Division",
        "Chapter 7: Circles",
        "Chapter 8: Mental Arithmetic",
        "Chapter 9: Fractional Numbers",
    ],
    5: [
        "Unit 1: 5-digit numbers",
        "Unit 2: Addition",
        "Unit 3: Subtraction",
        "Unit 4: Factors and multiples",
        "Unit 5: Fractions",
        "Unit 6: Angles",
        "Unit 7: Circles",
        "Unit 8: Length",
        "Unit 9: Perimeter and Area",
        "Unit 10: Data Handling",
    ],
}

# Page mappings (1-based printed page number to chapter) based on textbook TOCs
# Offset: the textbook front-matter ends at page 10 (PDF page 10 is index 9).
# Lessons start on printed page 1 (PDF page index 10).
_PDF_PAGE_OFFSET = 10


def get_textbook_path(grade: int | str) -> Path:
    """Return the filesystem path to the stored textbook for the given grade."""
    g_num = parse_grade_number(str(grade))
    if g_num is None or g_num < 1 or g_num > 5:
        raise ValueError(f"Invalid curriculum grade '{grade}'. Must be between 1 and 5.")
    return settings.curriculum_dir / f"class_{g_num}" / "textbook.pdf"


def get_curriculum_doc_id(grade: int | str) -> str:
    """Return the deterministic document ID for a grade's textbook."""
    g_num = parse_grade_number(str(grade))
    if g_num is None:
        raise ValueError(f"Cannot parse grade from '{grade}'")
    return f"curriculum_class_{g_num}"


def extract_chapter_number(chapter_str: str) -> tuple[int, str]:
    """
    Extract the numerical chapter index from strings like 'Lesson 1: ...',
    'Chapter 10: ...', or 'Unit 5 - ...' for stable textbook-order sorting.
    Returns (chapter_number, raw_string). Unnumbered chapters sort at the end (9999).
    """
    if not chapter_str:
        return (9999, "")
    # Check for keywords Lesson/Chapter/Unit followed by digits
    m = re.search(r'\b(?:lesson|chapter|unit|module|topic)\s*(\d+)', chapter_str, re.IGNORECASE)
    if m:
        return (int(m.group(1)), chapter_str)
    # Check for leading number like '1. Spatial Understanding'
    m2 = re.match(r'^\s*(\d+)[\.:\-\s]', chapter_str)
    if m2:
        return (int(m2.group(1)), chapter_str)
    return (9999, chapter_str)


def order_chapters_by_textbook(chapters: list[str]) -> list[str]:
    """
    Order chapters strictly according to textbook sequence (1, 2, ... 10).
    Never sorts alphabetically (which incorrectly places Unit 10 before Unit 2)
    and never uses an unordered collection.
    """
    if not chapters:
        return []
    # If the input list is already strictly monotonically ordered by chapter number,
    # preserve original list order verbatim
    nums = [extract_chapter_number(ch)[0] for ch in chapters]
    if all(nums[i] <= nums[i + 1] for i in range(len(nums) - 1)) and all(n != 9999 for n in nums):
        return list(chapters)
    # Otherwise sort numerically by textbook chapter number
    return sorted(chapters, key=extract_chapter_number)


def get_chapters_for_class(grade: int | str) -> list[str]:
    """Return the list of chapters/units for the given grade in strict textbook order."""
    g_num = parse_grade_number(str(grade))
    if g_num is None:
        return []
    raw = CURRICULUM_CHAPTERS.get(g_num, [])
    return order_chapters_by_textbook(raw)


def get_midterm_chapters_for_class(grade: int | str) -> list[str]:
    """
    Return the first half of chapters for the midterm examination in textbook order.
    Uses consistent ceil rounding for odd chapter counts (e.g. 9 -> 5, 10 -> 5).
    """
    all_ch = get_chapters_for_class(grade)
    if not all_ch:
        return []
    half_count = math.ceil(len(all_ch) / 2.0)
    return all_ch[:half_count]


def get_second_half_chapters_for_class(grade: int | str) -> list[str]:
    """
    Return the remaining (forbidden for midterm) chapters in textbook order.
    """
    all_ch = get_chapters_for_class(grade)
    if not all_ch:
        return []
    half_count = math.ceil(len(all_ch) / 2.0)
    return all_ch[half_count:]


def compute_file_hash(path: Path) -> str:
    """Compute sha256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def _get_chapter_for_printed_page(grade_num: int, printed_page: int) -> tuple[str, bool]:
    """
    Given a printed page number (1-based from textbook body), determine the chapter
    and whether it is in the first half of the syllabus.
    """
    chapters = CURRICULUM_CHAPTERS[grade_num]
    midterm_count = math.ceil(len(chapters) / 2)

    # Page intervals from the textbook Table of Contents
    if grade_num == 1:
        cuts = [(15, 0), (24, 1), (62, 2), (66, 3), (77, 4), (85, 5), (98, 6), (103, 7), (118, 8)]
    elif grade_num == 2:
        cuts = [(38, 0), (86, 1), (105, 2), (137, 3), (143, 4), (150, 5)]
    elif grade_num == 3:
        cuts = [(17, 0), (66, 1), (95, 2), (123, 3), (150, 4)]
    elif grade_num == 4:
        cuts = [(12, 0), (46, 1), (61, 2), (71, 3), (92, 4), (110, 5), (118, 6), (138, 7), (158, 8)]
    elif grade_num == 5:
        cuts = [(15, 0), (24, 1), (37, 2), (47, 3), (68, 4), (85, 5), (94, 6), (103, 7), (125, 8), (143, 9)]
    else:
        cuts = []

    for max_page, ch_idx in cuts:
        if printed_page <= max_page:
            ch_name = chapters[ch_idx]
            is_first_half = ch_idx < midterm_count
            return ch_name, is_first_half

    # Fallback to last chapter
    ch_idx = len(chapters) - 1
    return chapters[ch_idx], False


def extract_textbook_chunks(grade_num: int) -> list[Document]:
    """
    Extract textbook PDF pages grouped by chapter, and split into coherent chunks
    annotated with grade, subject, chapter, and whether the chapter is in the first half.
    """
    pdf_path = get_textbook_path(grade_num)
    if not pdf_path.exists():
        raise FileNotFoundError(f"Textbook for Class {grade_num} not found at {pdf_path}")

    doc = fitz.open(pdf_path)
    chapters = CURRICULUM_CHAPTERS.get(grade_num, [])
    midterm_count = math.ceil(len(chapters) / 2)

    # Group page texts by chapter
    ch_pages: dict[str, list[tuple[int, str]]] = {ch: [] for ch in chapters}
    for pdf_pno in range(_PDF_PAGE_OFFSET, len(doc)):
        printed_page = pdf_pno - _PDF_PAGE_OFFSET + 1
        page_text = clean_text(doc[pdf_pno].get_text())
        if page_text and len(page_text.strip()) > 10:
            ch_name, _ = _get_chapter_for_printed_page(grade_num, printed_page)
            if ch_name in ch_pages:
                ch_pages[ch_name].append((printed_page, page_text))
            elif chapters:
                ch_pages[chapters[-1]].append((printed_page, page_text))
    doc.close()

    chunks: list[Document] = []
    chunk_counter = 0

    chunk_size = 1200
    chunk_overlap = 150

    for ch_idx, ch_name in enumerate(chapters):
        pages_in_ch = ch_pages.get(ch_name, [])
        if not pages_in_ch:
            continue

        is_first_half = ch_idx < midterm_count
        combined_text = "\n\n".join(f"[Page {pno}]\n{txt}" for pno, txt in pages_in_ch)

        start = 0
        text_len = len(combined_text)
        step = chunk_size - chunk_overlap

        while start < text_len:
            sub = combined_text[start : start + chunk_size].strip()
            if len(sub) > 30:
                chunk_id = f"c{grade_num}_ch{ch_idx + 1}_k{chunk_counter}"
                metadata = {
                    "doc_id": get_curriculum_doc_id(grade_num),
                    "chunk_id": chunk_id,
                    "grade": f"Class {grade_num}",
                    "subject": "Mathematics",
                    "chapter": ch_name,
                    "unit": ch_name,
                    "is_first_half": is_first_half,
                    "source": f"Class {grade_num} Mathematics - {ch_name}",
                }
                chunks.append(Document(page_content=sub, metadata=metadata))
                chunk_counter += 1
            start += step

    return chunks


def index_curriculum_textbook(grade_num: int, force: bool = False) -> DocumentRecord:
    """
    Index a single class textbook if missing or modified.
    Builds and saves the FAISS index and stores the DocumentRecord.
    """
    pdf_path = get_textbook_path(grade_num)
    if not pdf_path.exists():
        raise FileNotFoundError(f"Textbook for Class {grade_num} does not exist at {pdf_path}")

    doc_id = get_curriculum_doc_id(grade_num)
    index_file = settings.indexes_dir / doc_id / "index.faiss"
    current_hash = compute_file_hash(pdf_path)

    # Check if already indexed and unchanged
    if not force and index_file.exists():
        try:
            rec = load_document_record(doc_id)
            if (getattr(rec, "content_hash", None) == current_hash or rec.has_index) and rec.num_chunks > 0:
                logger.info("Class %d textbook already indexed with %d chunks. Reusing.", grade_num, rec.num_chunks)
                return rec
        except Exception as e:
            logger.warning("Could not reuse existing record for Class %d: %s. Re-indexing.", grade_num, e)

    logger.info("Extracting and building FAISS index for Class %d textbook...", grade_num)
    chunks = extract_textbook_chunks(grade_num)
    if not chunks:
        raise ValueError(f"No text chunks could be extracted from Class {grade_num} textbook.")

    build_faiss_index(chunks, doc_id)

    record = DocumentRecord(
        document_id=doc_id,
        original_filename="textbook.pdf",
        safe_filename=f"class_{grade_num}_textbook.pdf",
        file_size=pdf_path.stat().st_size,
        content_hash=current_hash,
        num_chunks=len(chunks),
        chunk_ids=[c.metadata["chunk_id"] for c in chunks],
        has_index=True,
        grade=f"Class {grade_num}",
        subject="Mathematics",
        units=CURRICULUM_CHAPTERS.get(grade_num, []),
    )
    save_document_record(record)
    logger.info("Successfully indexed Class %d textbook (%d chunks).", grade_num, len(chunks))
    return record


def ensure_all_curriculum_textbooks_indexed() -> dict[int, bool]:
    """
    Ensure all Classes 1-5 textbooks are indexed. Called on application startup.
    Returns a dict mapping grade -> success status.
    """
    results: dict[int, bool] = {}
    for grade in range(1, 6):
        try:
            pdf_path = get_textbook_path(grade)
            if pdf_path.exists():
                index_curriculum_textbook(grade, force=False)
                results[grade] = True
            else:
                logger.warning("Textbook for Class %d not found at %s", grade, pdf_path)
                results[grade] = False
        except Exception as e:
            logger.error("Failed to index Class %d textbook: %s", grade, e)
            results[grade] = False
    return results


def get_curriculum_status() -> list[dict[str, Any]]:
    """
    Get the readiness status of all 5 class textbooks for the UI.
    """
    status_list = []
    for grade in range(1, 6):
        pdf_path = get_textbook_path(grade)
        doc_id = get_curriculum_doc_id(grade)
        index_file = settings.indexes_dir / doc_id / "index.faiss"

        is_available = pdf_path.exists()
        is_indexed = is_available and index_file.exists()

        num_chunks = 0
        if is_indexed:
            try:
                rec = load_document_record(doc_id)
                num_chunks = rec.num_chunks
            except Exception:
                num_chunks = 0

        chapters = get_chapters_for_class(grade)
        midterm_chapters = get_midterm_chapters_for_class(grade)

        status_list.append({
            "grade": grade,
            "class_name": f"Class {grade}",
            "available": is_available,
            "indexed": is_indexed,
            "file_size": pdf_path.stat().st_size if is_available else 0,
            "total_chapters": len(chapters),
            "chapters": chapters,
            "midterm_chapters": midterm_chapters,
            "total_chunks": num_chunks,
            "document_id": doc_id,
        })
    return status_list
