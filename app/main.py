# app/main.py
# FastAPI application entry point with all API routes.
# This is the file you run with: uvicorn app.main:app --reload
#
# WARNING: This prototype has NO user authentication.
# Do NOT deploy it as a public production service.
# It is designed for local development or a trusted private environment.

import logging
import uuid
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import Response, HTMLResponse

from app.config import settings
from app.schemas import (
    DocumentInfo,
    DocumentRecord,
    DocumentSummary,
    TextDocumentRequest,
    PaperGenerateRequest,
    BatchPaperGenerateRequest,
    Paper,
    PaperSummary,
    BatchPaperResponse,
    BatchValidationSummary,
    MathCheckStatus,
)
from app.documents import (
    validate_file_extension,
    extract_text,
    split_text_into_chunks,
    count_pages,
)
from app.retrieval import build_faiss_index, retrieve_relevant_chunks
from app.generator import generate_questions, generate_candidate_pool
from app.duplicates import select_version_questions, CandidatePoolShortageError
from app.paper import (
    assemble_paper,
    save_paper,
    load_paper,
    list_papers,
    delete_paper,
)
from app.pdf_export import generate_pdf
from app.documents_store import (
    save_document_record,
    load_document_record,
    list_document_records,
    delete_document_and_assets,
)

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

INDEX_HTML_PATH = Path(__file__).resolve().parent / "templates" / "index.html"

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="QPGen — Question Paper Generator",
    description=(
        "AI-based curriculum-aligned mathematics question paper generator. "
        "Upload a syllabus or textbook, then generate exam papers grounded "
        "in the uploaded content using Google Gemini."
    ),
    version="0.1.0",
)


# ---------------------------------------------------------------------------
# GET / — interactive web UI for teachers
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse, tags=["Web UI"])
def teacher_dashboard():
    """Serve the interactive Teacher Portal web UI."""
    if INDEX_HTML_PATH.exists():
        return HTMLResponse(content=INDEX_HTML_PATH.read_text(encoding="utf-8"))
    return HTMLResponse(
        content="""
        <html>
            <body style='font-family:sans-serif;padding:40px;background:#0a0e17;color:white;'>
                <h2>QPGen API is Running</h2>
                <p>Interactive docs available at <a style='color:#6366f1;' href='/docs'>/docs</a>.</p>
            </body>
        </html>
        """
    )


# ---------------------------------------------------------------------------
# GET /health — basic health check
# ---------------------------------------------------------------------------

@app.get("/health", tags=["System"])
def health_check():
    """Return a simple health status."""
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# POST /documents/upload — upload a PDF/DOCX, extract text, build index
# ---------------------------------------------------------------------------

@app.post("/documents/upload", response_model=DocumentInfo, tags=["Documents"])
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a PDF or DOCX file.
    Extracts text, splits it into chunks, creates embeddings,
    and builds a FAISS index for later retrieval.
    """
    # 1. Validate file extension
    try:
        extension = validate_file_extension(file.filename or "unknown")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # 2. Read and check file size
    contents = await file.read()
    if len(contents) > settings.MAX_UPLOAD_SIZE_BYTES:
        max_mb = settings.MAX_UPLOAD_SIZE_BYTES / (1024 * 1024)
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {max_mb:.0f} MB.",
        )

    # 3. Save the file with a safe generated name
    document_id = uuid.uuid4().hex[:12]
    safe_filename = f"{document_id}{extension}"
    file_path = settings.uploads_dir / safe_filename

    file_path.write_bytes(contents)
    logger.info("Saved uploaded file as %s", file_path)

    # 4. Extract text
    try:
        text = extract_text(file_path, extension)
    except ValueError as e:
        # Clean up the saved file on failure
        file_path.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=str(e))

    # 5. Split into chunks
    chunks = split_text_into_chunks(text, source_filename=file.filename or safe_filename)
    logger.info("Split document into %d chunks", len(chunks))

    # 6. Build FAISS index (this calls the embedding model)
    index_built = False
    try:
        build_faiss_index(chunks, document_id)
        index_built = True
    except Exception as e:
        logger.error("Failed to build FAISS index: %s", e)
        raise HTTPException(
            status_code=502,
            detail=f"Failed to create embeddings. Check your GOOGLE_API_KEY. Error: {e}",
        )

    # 7. Count pages for metadata
    num_pages = count_pages(file_path, extension)

    # 8. Save document record for tracking and traceability
    text_preview = text[:300].strip()
    if len(text) > 300:
        text_preview += "…"

    doc_record = DocumentRecord(
        document_id=document_id,
        original_filename=file.filename or safe_filename,
        safe_filename=safe_filename,
        num_chunks=len(chunks),
        num_pages=num_pages,
        file_size_bytes=len(contents),
        has_index=index_built,
        text_preview=text_preview,
    )
    save_document_record(doc_record)

    return DocumentInfo(
        document_id=document_id,
        filename=file.filename or safe_filename,
        num_chunks=len(chunks),
        message="Document uploaded and indexed successfully.",
    )


# ---------------------------------------------------------------------------
# POST /documents/text — submit syllabus content directly as plain text
# ---------------------------------------------------------------------------

@app.post("/documents/text", response_model=DocumentInfo, tags=["Documents"])
def upload_text_document(request: TextDocumentRequest):
    """
    Submit syllabus or textbook content directly as plain text.
    Splits into chunks, creates embeddings, builds FAISS index,
    and stores a document record for question generation.
    """
    clean_text = request.content.strip()
    if not clean_text:
        raise HTTPException(status_code=400, detail="Document text cannot be empty.")

    # 1. Generate document ID and safe filename
    document_id = uuid.uuid4().hex[:12]
    safe_filename = f"{document_id}.txt"
    file_path = settings.uploads_dir / safe_filename

    # 2. Save raw text file
    file_path.write_text(clean_text, encoding="utf-8")
    logger.info("Saved text document as %s", file_path)

    # 3. Split into chunks
    display_name = f"{request.title.strip()}.txt"
    chunks = split_text_into_chunks(clean_text, source_filename=display_name)
    logger.info("Split text document into %d chunks", len(chunks))

    # 4. Build FAISS index
    index_built = False
    try:
        build_faiss_index(chunks, document_id)
        index_built = True
    except Exception as e:
        logger.error("Failed to build FAISS index for text document: %s", e)
        file_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=502,
            detail=f"Failed to create embeddings. Check your GOOGLE_API_KEY. Error: {e}",
        )

    # 5. Estimate pages (~3000 chars per typical page)
    num_pages = max(1, len(clean_text) // 3000 + (1 if len(clean_text) % 3000 else 0))

    # 6. Save document record
    text_preview = clean_text[:300].strip()
    if len(clean_text) > 300:
        text_preview += "…"

    doc_record = DocumentRecord(
        document_id=document_id,
        original_filename=display_name,
        safe_filename=safe_filename,
        num_chunks=len(chunks),
        num_pages=num_pages,
        file_size_bytes=len(clean_text.encode("utf-8")),
        has_index=index_built,
        text_preview=text_preview,
    )
    save_document_record(doc_record)

    return DocumentInfo(
        document_id=document_id,
        filename=display_name,
        num_chunks=len(chunks),
        message="Text syllabus uploaded and indexed successfully.",
    )


# ---------------------------------------------------------------------------
# GET /documents — list all uploaded documents
# ---------------------------------------------------------------------------

@app.get("/documents", response_model=list[DocumentSummary], tags=["Documents"])
def get_documents():
    """List all uploaded and processed documents."""
    return list_document_records()


# ---------------------------------------------------------------------------
# GET /documents/{document_id} — retrieve one document record
# ---------------------------------------------------------------------------

@app.get("/documents/{document_id}", response_model=DocumentRecord, tags=["Documents"])
def get_document(document_id: str):
    """Retrieve metadata for a single uploaded document."""
    try:
        return load_document_record(document_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


# ---------------------------------------------------------------------------
# DELETE /documents/{document_id} — delete document and its index
# ---------------------------------------------------------------------------

@app.delete("/documents/{document_id}", tags=["Documents"])
def delete_document(document_id: str):
    """Delete an uploaded document, its metadata record, and its FAISS index."""
    deleted = delete_document_and_assets(document_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Document '{document_id}' not found.")
    return {"message": f"Document '{document_id}' and all associated assets deleted successfully."}


# ---------------------------------------------------------------------------
# POST /papers/generate — generate a question paper
# ---------------------------------------------------------------------------

@app.post("/papers/generate", response_model=Paper, tags=["Papers"])
def generate_paper(request: PaperGenerateRequest):
    """
    Generate a question paper from an uploaded document.
    Retrieves relevant passages, calls Gemini to generate questions,
    validates the output, assembles the paper, and saves it as JSON.
    """
    # 1. Retrieve relevant passages
    try:
        passages = retrieve_relevant_chunks(request.document_id, request.topic)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    # 2. Generate questions with Gemini
    try:
        questions = generate_questions(request, passages)
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))
    except Exception as e:
        logger.error("Question generation failed: %s", e)
        raise HTTPException(
            status_code=500,
            detail=f"Question generation failed unexpectedly: {e}",
        )

    # 3. Assemble and save the paper
    try:
        blueprint = request.get_or_create_blueprint()
        paper = assemble_paper(
            request=request,
            questions=questions,
            blueprint_id=blueprint.blueprint_id,
        )
        save_paper(paper)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    return paper


# ---------------------------------------------------------------------------
# POST /papers/generate-batch — generate multiple unique paper versions
# ---------------------------------------------------------------------------

@app.post("/papers/generate-batch", response_model=BatchPaperResponse, tags=["Papers"])
def generate_batch_papers(request: BatchPaperGenerateRequest):
    """
    Generate multiple unique paper versions in a single request sharing the same blueprint.
    Retrieves syllabus passages, builds a candidate question pool, validates mathematical
    correctness with SymPy, detects duplicates, selects disjoint sets of questions for each
    version without replacement, and saves each paper to JSON.
    """
    blueprint = request.get_or_create_blueprint()
    batch_id = uuid.uuid4().hex[:10]

    # 1. Retrieve relevant passages
    try:
        passages = retrieve_relevant_chunks(request.document_id, request.topic)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    # 2. Generate candidate question pool
    try:
        candidates, dups_rejected, math_discrepancies = generate_candidate_pool(
            blueprint=blueprint,
            passages=passages,
            grade=request.grade,
            subject=request.subject,
            topic=request.topic,
            num_versions=request.num_versions,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))
    except Exception as e:
        logger.error("Candidate generation failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Candidate generation failed: {e}")

    # 3. Select questions without replacement across versions
    try:
        versions_questions, versions_sections = select_version_questions(
            candidate_pool=candidates,
            blueprint=blueprint,
            num_versions=request.num_versions,
        )
    except CandidatePoolShortageError as e:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "CandidatePoolShortageError",
                "message": str(e),
                "section_id": e.section_id,
                "needed": e.needed,
                "available": e.available,
            },
        )

    # 4. Assemble and persist each version
    saved_papers: list[Paper] = []
    paper_ids: list[str] = []

    for v_idx in range(request.num_versions):
        v_num = v_idx + 1
        v_qs = versions_questions[v_idx]
        v_secs = versions_sections[v_idx]

        single_req = PaperGenerateRequest(
            document_id=request.document_id,
            grade=request.grade,
            subject=request.subject,
            topic=request.topic,
            difficulty=request.difficulty,
            num_questions=blueprint.total_questions,
            total_marks=blueprint.total_marks,
            num_versions=request.num_versions,
        )

        paper = assemble_paper(
            request=single_req,
            questions=v_qs,
            sections=v_secs,
            batch_id=batch_id,
            version_number=v_num,
            num_versions=request.num_versions,
            blueprint_id=blueprint.blueprint_id,
        )
        save_paper(paper)
        saved_papers.append(paper)
        paper_ids.append(paper.paper_id)

    # 5. Validation summary
    math_verified = sum(
        1 for p in saved_papers for q in p.questions
        if q.math_verification and q.math_verification.status == MathCheckStatus.VERIFIED
    )
    all_q_ids = [q.question_id for p in saved_papers for q in p.questions]
    is_unique = len(all_q_ids) == len(set(all_q_ids))

    summary = BatchValidationSummary(
        total_candidates_generated=len(candidates) + dups_rejected + math_discrepancies,
        valid_candidates_count=len(candidates),
        duplicate_candidates_rejected=dups_rejected,
        math_verified_count=math_verified,
        math_discrepancies_rejected=math_discrepancies,
        is_unique=is_unique,
        all_versions_valid=all(p.num_questions == blueprint.total_questions for p in saved_papers),
        details=(
            f"Generated {len(saved_papers)} unique versions ({blueprint.total_questions} questions, "
            f"{blueprint.total_marks} marks each) under blueprint '{blueprint.title}'."
        ),
    )

    return BatchPaperResponse(
        batch_id=batch_id,
        num_versions_requested=request.num_versions,
        num_versions_generated=len(saved_papers),
        blueprint=blueprint,
        papers=saved_papers,
        paper_ids=paper_ids,
        validation_summary=summary,
    )


# ---------------------------------------------------------------------------
# GET /papers — list all saved papers
# ---------------------------------------------------------------------------

@app.get("/papers", response_model=list[PaperSummary], tags=["Papers"])
def get_papers():
    """List all saved question papers."""
    return list_papers()


# ---------------------------------------------------------------------------
# GET /papers/{paper_id} — retrieve one saved paper
# ---------------------------------------------------------------------------

@app.get("/papers/{paper_id}", response_model=Paper, tags=["Papers"])
def get_paper(paper_id: str):
    """Retrieve a single saved paper by its ID."""
    try:
        return load_paper(paper_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


# ---------------------------------------------------------------------------
# GET /papers/{paper_id}/pdf — download the paper as a PDF
# ---------------------------------------------------------------------------

@app.get("/papers/{paper_id}/pdf", tags=["Papers"])
def download_paper_pdf(paper_id: str, include_answers: bool = True):
    """
    Download a generated paper as a PDF.
    Set include_answers=false for a student version without solutions.
    """
    try:
        paper = load_paper(paper_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    pdf_bytes = generate_pdf(paper, include_answers=include_answers)

    filename = f"{paper.title.replace(' ', '_')}_{paper.paper_id}.pdf"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ---------------------------------------------------------------------------
# DELETE /papers/{paper_id} — delete a saved question paper
# ---------------------------------------------------------------------------

@app.delete("/papers/{paper_id}", tags=["Papers"])
def delete_paper_endpoint(paper_id: str):
    """Delete a saved question paper."""
    deleted = delete_paper(paper_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Paper '{paper_id}' not found.")
    return {"message": f"Paper '{paper_id}' deleted successfully."}

