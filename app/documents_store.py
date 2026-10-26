# app/documents_store.py
# Persistence layer for document metadata.
# Each uploaded document gets a JSON record saved alongside the file itself.
# This enables listing, retrieving, and tracing documents without re-scanning.

import json
import logging
import tempfile
from pathlib import Path

from app.config import settings
from app.schemas import DocumentRecord, DocumentSummary

logger = logging.getLogger(__name__)


def save_document_record(record: DocumentRecord) -> Path:
    """
    Save a document metadata record as a JSON file.
    Uses atomic write (write to temp, then rename) to reduce corruption risk.

    Returns the path to the saved file.
    """
    file_path = settings.uploads_dir / f"{record.document_id}.json"

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
            tmp.write(record.model_dump_json(indent=2))
            tmp_path = Path(tmp.name)
        tmp_path.replace(file_path)
    except Exception:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise

    logger.info("Saved document record %s to %s", record.document_id, file_path)
    return file_path


def load_document_record(document_id: str) -> DocumentRecord:
    """
    Load a document metadata record from its JSON file.
    Raises FileNotFoundError if it doesn't exist.
    Raises ValueError if the file is malformed.
    """
    # Prevent path traversal
    safe_id = Path(document_id).name
    file_path = settings.uploads_dir / f"{safe_id}.json"

    if not file_path.exists():
        raise FileNotFoundError(f"Document '{document_id}' not found.")

    try:
        raw = file_path.read_text(encoding="utf-8")
        data = json.loads(raw)
        return DocumentRecord.model_validate(data)
    except (json.JSONDecodeError, Exception) as e:
        raise ValueError(f"Document record '{document_id}' is malformed: {e}")


def list_document_records() -> list[DocumentSummary]:
    """
    List all saved document records as lightweight summaries.
    Skips files that cannot be parsed.
    """
    summaries = []
    for file_path in sorted(settings.uploads_dir.glob("*.json")):
        try:
            raw = file_path.read_text(encoding="utf-8")
            data = json.loads(raw)
            summaries.append(
                DocumentSummary(
                    document_id=data["document_id"],
                    original_filename=data["original_filename"],
                    num_chunks=data["num_chunks"],
                    file_size_bytes=data.get("file_size_bytes", 0),
                    uploaded_at=data.get("uploaded_at", ""),
                    has_index=data.get("has_index", False),
                )
            )
        except Exception as e:
            logger.warning("Skipping malformed document record %s: %s", file_path.name, e)
    return summaries


def delete_document_record(document_id: str) -> bool:
    """
    Delete a document record JSON file.
    Returns True if deleted, False if it didn't exist.
    """
    safe_id = Path(document_id).name
    file_path = settings.uploads_dir / f"{safe_id}.json"
    if file_path.exists():
        file_path.unlink()
        return True
    return False


def delete_document_and_assets(document_id: str) -> bool:
    """
    Delete a document record JSON file along with its uploaded file and FAISS index.
    Returns True if record was found and deleted, False otherwise.
    """
    safe_id = Path(document_id).name
    record_file = settings.uploads_dir / f"{safe_id}.json"
    if not record_file.exists():
        return False

    safe_filename = None
    try:
        data = json.loads(record_file.read_text(encoding="utf-8"))
        safe_filename = data.get("safe_filename")
    except Exception:
        pass

    # Delete the JSON record
    record_file.unlink(missing_ok=True)

    # Delete the uploaded file
    if safe_filename:
        (settings.uploads_dir / safe_filename).unlink(missing_ok=True)
    else:
        for ext in (".pdf", ".docx", ".txt"):
            (settings.uploads_dir / f"{safe_id}{ext}").unlink(missing_ok=True)

    # Delete the FAISS index directory if present
    index_dir = settings.indexes_dir / safe_id
    if index_dir.exists() and index_dir.is_dir():
        import shutil
        shutil.rmtree(index_dir, ignore_errors=True)

    logger.info("Deleted document %s and associated assets", safe_id)
    return True

