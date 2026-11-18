# app/main.py
# FastAPI application entry point with all API routes.
# Protected with enterprise authentication, session tokens, and admin user controls.

import io
import logging
import re
import uuid
import zipfile
from typing import Optional
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, Response, status, Depends, Query
from fastapi.responses import Response, HTMLResponse, RedirectResponse

from app.config import settings
from app.auth import (
    User,
    UserRole,
    LoginRequest,
    UserCreateRequest,
    UserStatusUpdateRequest,
    UserResetPasswordRequest,
    COOKIE_NAME,
    init_auth_db,
    get_db,
    check_rate_limit,
    record_login_attempt,
    verify_password,
    create_user_session,
    delete_session,
    extract_session_id,
    get_user_for_session,
    get_current_user,
    require_admin,
    require_teacher,
    list_all_users,
    create_teacher_account,
    update_user_status,
    reset_user_password,
    AuthenticationRedirectException,
)
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
    ExamType,
    parse_grade_number,
    get_previous_class,
    format_grade_display,
    extract_syllabus_units,
    determine_midterm_scope,
    determine_second_half_scope,
    CustomSyllabusOut,
    CustomSyllabusSummary,
    CustomSyllabusUpdateRequest,
    SET_LABELS,
    format_set_or_student_label,
)
from app.custom_syllabus import (
    SelectionError,
    build_scope_text,
    create_custom_syllabus,
    delete_custom_syllabus,
    list_custom_syllabi,
    load_custom_syllabus,
    resolve_selection,
    update_custom_syllabus,
)
from app.documents import (
    validate_file_extension,
    extract_text,
    split_text_into_chunks,
    count_pages,
)
from app.retrieval import build_faiss_index, retrieve_relevant_chunks
from app.generator import generate_questions, generate_candidate_pool
from app.duplicates import (
    select_version_questions,
    CandidatePoolShortageError,
    validate_batch_papers_uniqueness,
)
from app.paper import (
    assemble_paper,
    save_paper,
    load_paper,
    list_papers,
    delete_paper,
)
from app.validator import validate_full_paper
from app.pdf_export import generate_pdf
from app.documents_store import (
    save_document_record,
    load_document_record,
    list_document_records,
    delete_document_and_assets,
    find_best_document_for_grade,
    find_documents_by_class_and_subject,
)
from app.curriculum import (
    CURRICULUM_CHAPTERS,
    get_textbook_path,
    get_curriculum_doc_id,
    get_chapters_for_class,
    get_midterm_chapters_for_class,
    get_second_half_chapters_for_class,
    ensure_all_curriculum_textbooks_indexed,
    get_curriculum_status,
    resolve_authoritative_syllabus_scope,
    assign_blueprint_chapter_coverage,
    order_chapters_by_textbook,
    SyllabusScopeInfo,
)

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

INDEX_HTML_PATH = Path(__file__).resolve().parent / "templates" / "index.html"
LOGIN_HTML_PATH = Path(__file__).resolve().parent / "templates" / "login.html"
ADMIN_HTML_PATH = Path(__file__).resolve().parent / "templates" / "admin.html"

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


@app.exception_handler(AuthenticationRedirectException)
async def auth_redirect_handler(request: Request, exc: AuthenticationRedirectException):
    """Redirect unauthenticated browser requests directly to the login portal."""
    return RedirectResponse(url="/login", status_code=status.HTTP_303_SEE_OTHER)


@app.on_event("startup")
def on_startup():
    """Ensure authentication database and curriculum textbooks are indexed on application start."""
    logger.info("Initializing authentication database...")
    init_auth_db()
    logger.info("Checking QPGen permanent curriculum textbooks...")
    try:
        ensure_all_curriculum_textbooks_indexed()
    except Exception as e:
        logger.warning("Startup curriculum indexing notice: %s", e)


# ---------------------------------------------------------------------------
# Authentication Endpoints (Login, Logout, Profile, Admin User Controls)
# ---------------------------------------------------------------------------

@app.get("/login", response_class=HTMLResponse, tags=["Authentication"])
def login_page(request: Request):
    """Serve the login page or redirect to dashboard if already authenticated."""
    session_id = extract_session_id(request)
    if session_id:
        with get_db() as conn:
            user = get_user_for_session(conn, session_id)
            if user and user.is_active:
                if user.role == UserRole.ADMIN:
                    return RedirectResponse(url="/admin", status_code=status.HTTP_303_SEE_OTHER)
                return RedirectResponse(url="/", status_code=status.HTTP_303_SEE_OTHER)
    if LOGIN_HTML_PATH.exists():
        return HTMLResponse(content=LOGIN_HTML_PATH.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h2>QPGen Login</h2><p>Login template missing.</p>")


@app.post("/auth/login", tags=["Authentication"])
def auth_login(req: LoginRequest, request: Request, response: Response):
    """
    Authenticate teacher or administrator credentials.
    Rate-limited to protect against brute-force attacks.
    Sets secure HttpOnly session cookie on success.
    """
    ip_addr = request.client.host if request.client else "unknown"
    ident = req.identifier.strip().lower()

    with get_db() as conn:
        # Check rate limiting (5 failed attempts per 15 mins)
        if check_rate_limit(conn, ip_addr, ident):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Too many failed login attempts. Please wait 15 minutes before trying again."
            )

        cur = conn.execute(
            """
            SELECT id, username, email, password_hash, full_name, role, is_active, created_at, last_login_at
            FROM users
            WHERE username = ? OR email = ?;
            """,
            (ident, ident)
        )
        row = cur.fetchone()

        if not row or not verify_password(req.password, row["password_hash"]):
            record_login_attempt(conn, ip_addr, ident, success=False)
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid username/email or password.",
            )

        if not row["is_active"]:
            record_login_attempt(conn, ip_addr, ident, success=False)
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Your account has been deactivated. Please contact your administrator.",
            )

        record_login_attempt(conn, ip_addr, ident, success=True)
        user_agent = request.headers.get("user-agent", "")
        session_id = create_user_session(conn, row["id"], ip_addr, user_agent)

        # Set secure HttpOnly session cookie
        response.set_cookie(
            key=COOKIE_NAME,
            value=session_id,
            max_age=settings.SESSION_MAX_AGE_SECONDS,
            httponly=True,
            samesite="lax",
            path="/",
        )

        user_profile = User(
            id=row["id"],
            username=row["username"],
            email=row["email"],
            full_name=row["full_name"],
            role=UserRole(row["role"]),
            is_active=bool(row["is_active"]),
            created_at=row["created_at"],
            last_login_at=row["last_login_at"],
        )

        return {
            "status": "ok",
            "message": "Login successful",
            "session_id": session_id,
            "user": user_profile,
        }


@app.post("/auth/logout", tags=["Authentication"])
def auth_logout(request: Request, response: Response):
    """Log out user, invalidate session in database, and clear cookie."""
    session_id = extract_session_id(request)
    if session_id:
        with get_db() as conn:
            delete_session(conn, session_id)
    response.delete_cookie(key=COOKIE_NAME, path="/")
    return {"status": "ok", "message": "Logged out successfully."}


@app.get("/auth/me", response_model=User, tags=["Authentication"])
def auth_me(current_user: User = Depends(get_current_user)):
    """Return currently authenticated user profile."""
    return current_user


# ---------------------------------------------------------------------------
# Administrator User Management (Admin Only — No Public Registration)
# ---------------------------------------------------------------------------

@app.get("/admin", response_class=HTMLResponse, tags=["Admin"])
def admin_dashboard(admin: User = Depends(require_admin)):
    """Serve the dedicated Access Management workspace for administrators."""
    if ADMIN_HTML_PATH.exists():
        return HTMLResponse(content=ADMIN_HTML_PATH.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h2>Access Management</h2><p>Admin template missing.</p>")


@app.get("/admin/users", response_model=list[User], tags=["Admin"])
def admin_get_users(admin: User = Depends(require_admin)):
    """List all user accounts for administration. Never includes password hashes."""
    with get_db() as conn:
        return list_all_users(conn)


@app.post("/admin/users", response_model=User, status_code=status.HTTP_201_CREATED, tags=["Admin"])
def admin_create_teacher(req: UserCreateRequest, admin: User = Depends(require_admin)):
    """Administrator-only endpoint to provision a new teacher account."""
    with get_db() as conn:
        return create_teacher_account(conn, req)


@app.patch("/admin/users/{user_id}/status", response_model=User, tags=["Admin"])
def admin_set_user_status(user_id: str, req: UserStatusUpdateRequest, admin: User = Depends(require_admin)):
    """Activate or deactivate a user account. Deactivation immediately invalidates active sessions."""
    with get_db() as conn:
        return update_user_status(conn, user_id=user_id, is_active=req.is_active, acting_admin_id=admin.id)


@app.post("/admin/users/{user_id}/reset-password", tags=["Admin"])
def admin_reset_password(user_id: str, req: UserResetPasswordRequest, admin: User = Depends(require_admin)):
    """Reset a teacher account password and terminate their active sessions."""
    with get_db() as conn:
        reset_user_password(conn, user_id=user_id, new_password=req.new_password)
    return {"status": "ok", "message": "Password reset successfully. The teacher must log in again."}


# Explicit rejection of public registration
@app.api_route("/register", methods=["GET", "POST"], include_in_schema=False)
@app.api_route("/signup", methods=["GET", "POST"], include_in_schema=False)
def no_public_registration():
    """Explicitly reject all public registration attempts."""
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="Public registration is disabled. Teacher accounts are created exclusively by administrators.",
    )


# ---------------------------------------------------------------------------
# GET /curriculum/status — check availability and indexing of Class 1-5 textbooks
# ---------------------------------------------------------------------------

@app.get("/curriculum/status", tags=["Curriculum"])
def get_curriculum_readiness(current_user: User = Depends(require_teacher)):
    """Return availability and indexing readiness status of the 5 stored class textbooks."""
    return get_curriculum_status()


# ---------------------------------------------------------------------------
# GET / — interactive web UI for teachers
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse, tags=["Web UI"])
def teacher_dashboard(current_user: User = Depends(get_current_user)):
    """Serve the interactive Teacher Portal web UI (requires authentication)."""
    if current_user.role == UserRole.ADMIN:
        return RedirectResponse(url="/admin", status_code=status.HTTP_303_SEE_OTHER)
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
async def upload_document(
    file: UploadFile = File(...),
    grade: Optional[str] = Form(None),
    subject: Optional[str] = Form("Mathematics"),
    current_user: User = Depends(require_teacher),
):
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

    # 5. Extract metadata & units
    parsed_grade = grade
    if not parsed_grade:
        num = parse_grade_number(file.filename or "")
        if num is not None:
            parsed_grade = str(num)

    units = extract_syllabus_units(text)

    # 6. Split into chunks
    chunks = split_text_into_chunks(
        text,
        source_filename=file.filename or safe_filename,
        grade=parsed_grade,
        subject=subject or "Mathematics",
    )
    logger.info("Split document into %d chunks", len(chunks))

    # 7. Build FAISS index (this calls the embedding model)
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

    # 8. Count pages for metadata
    num_pages = count_pages(file_path, extension)

    # 9. Save document record for tracking and traceability
    text_preview = text[:300].strip()
    if len(text) > 300:
        text_preview += "…"

    doc_record = DocumentRecord(
        document_id=document_id,
        original_filename=file.filename or safe_filename,
        safe_filename=safe_filename,
        grade=parsed_grade,
        subject=subject or "Mathematics",
        units=units if units else None,
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
def upload_text_document(request: TextDocumentRequest, current_user: User = Depends(require_teacher)):
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

    # 3. Extract metadata & units
    parsed_grade = request.grade
    if not parsed_grade:
        num = parse_grade_number(request.title)
        if num is not None:
            parsed_grade = str(num)

    units = extract_syllabus_units(clean_text)

    # 4. Split into chunks
    display_name = f"{request.title.strip()}.txt"
    chunks = split_text_into_chunks(
        clean_text,
        source_filename=display_name,
        grade=parsed_grade,
        subject=request.subject or "Mathematics",
    )
    logger.info("Split text document into %d chunks", len(chunks))

    # 5. Build FAISS index
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

    # 6. Estimate pages (~3000 chars per typical page)
    num_pages = max(1, len(clean_text) // 3000 + (1 if len(clean_text) % 3000 else 0))

    # 7. Save document record
    text_preview = clean_text[:300].strip()
    if len(clean_text) > 300:
        text_preview += "…"

    doc_record = DocumentRecord(
        document_id=document_id,
        original_filename=display_name,
        safe_filename=safe_filename,
        grade=parsed_grade,
        subject=request.subject or "Mathematics",
        units=units if units else None,
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
def get_documents(current_user: User = Depends(require_teacher)):
    """List all uploaded and processed documents."""
    return list_document_records()


# ---------------------------------------------------------------------------
# GET /documents/{document_id} — retrieve one document record
# ---------------------------------------------------------------------------

@app.get("/documents/{document_id}", response_model=DocumentRecord, tags=["Documents"])
def get_document(document_id: str, current_user: User = Depends(require_teacher)):
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
def delete_document(document_id: str, current_user: User = Depends(require_teacher)):
    """Delete an uploaded document, its metadata record, and its FAISS index."""
    deleted = delete_document_and_assets(document_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Document '{document_id}' not found.")
    return {"message": f"Document '{document_id}' and all associated assets deleted successfully."}


# ---------------------------------------------------------------------------
# Custom Syllabus — upload / manual entry, extraction, save & reuse
# ---------------------------------------------------------------------------

def _syllabus_out(syl) -> CustomSyllabusOut:
    return CustomSyllabusOut(
        syllabus_id=syl.syllabus_id,
        name=syl.name,
        grade=syl.grade,
        subject=syl.subject,
        source_type=syl.source_type,
        source_filename=syl.source_filename,
        chapters=[{"title": c.title, "topics": c.topics} for c in syl.chapters],
        num_chunks=syl.num_chunks,
        has_index=syl.has_index,
        created_at=syl.created_at,
        updated_at=syl.updated_at,
    )


@app.post("/syllabi/custom", response_model=CustomSyllabusOut, tags=["Custom Syllabus"])
async def create_custom_syllabus_endpoint(
    name: str = Form(...),
    file: Optional[UploadFile] = File(None),
    text: Optional[str] = Form(None),
    grade: Optional[str] = Form(None),
    subject: Optional[str] = Form("Mathematics"),
    current_user: User = Depends(require_teacher),
):
    """
    Create a reusable custom syllabus from an uploaded PDF/DOCX or manually typed text.
    Chapters and topics are extracted, the content is indexed in FAISS, and the
    syllabus is saved so it can be reused for later papers.
    """
    if not name.strip():
        raise HTTPException(status_code=400, detail="Please provide a name for the custom syllabus.")
    has_file = file is not None and bool(file.filename)
    has_text = bool(text and text.strip())
    if not has_file and not has_text:
        raise HTTPException(status_code=400, detail="Upload a PDF/DOCX file or enter the syllabus content manually.")

    syllabus_id = f"custom_{uuid.uuid4().hex[:12]}"
    source_type, source_filename, saved_path = "text", None, None

    if has_file:
        try:
            extension = validate_file_extension(file.filename or "unknown")
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        contents = await file.read()
        if len(contents) > settings.MAX_UPLOAD_SIZE_BYTES:
            max_mb = settings.MAX_UPLOAD_SIZE_BYTES / (1024 * 1024)
            raise HTTPException(status_code=413, detail=f"File too large. Maximum size is {max_mb:.0f} MB.")
        saved_path = settings.uploads_dir / f"{syllabus_id}{extension}"
        saved_path.write_bytes(contents)
        try:
            body_text = extract_text(saved_path, extension)
        except ValueError as e:
            saved_path.unlink(missing_ok=True)
            raise HTTPException(status_code=422, detail=str(e))
        except Exception as e:
            saved_path.unlink(missing_ok=True)
            raise HTTPException(status_code=422, detail=f"Could not read the uploaded file: {e}")
        source_type, source_filename = extension.lstrip("."), file.filename
    else:
        body_text = text.strip()

    try:
        syl = create_custom_syllabus(
            name=name,
            text=body_text,
            owner_id=current_user.id,
            grade=grade,
            subject=subject or "Mathematics",
            source_type=source_type,
            source_filename=source_filename,
            syllabus_id=syllabus_id,
        )
    except ValueError as e:
        if saved_path:
            saved_path.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        if saved_path:
            saved_path.unlink(missing_ok=True)
        logger.error("Custom syllabus indexing failed: %s", e)
        raise HTTPException(
            status_code=502,
            detail=f"Failed to create embeddings. Check your GOOGLE_API_KEY. Error: {e}",
        )
    return _syllabus_out(syl)


@app.get("/syllabi/custom", response_model=list[CustomSyllabusSummary], tags=["Custom Syllabus"])
def list_custom_syllabi_endpoint(current_user: User = Depends(require_teacher)):
    """List the current teacher's saved custom syllabi."""
    return [
        CustomSyllabusSummary(
            syllabus_id=s.syllabus_id,
            name=s.name,
            grade=s.grade,
            subject=s.subject,
            source_type=s.source_type,
            num_chapters=len(s.chapters),
            num_topics=sum(len(c.topics) for c in s.chapters),
            created_at=s.created_at,
        )
        for s in list_custom_syllabi(current_user.id)
    ]


@app.get("/syllabi/custom/{syllabus_id}", response_model=CustomSyllabusOut, tags=["Custom Syllabus"])
def get_custom_syllabus_endpoint(syllabus_id: str, current_user: User = Depends(require_teacher)):
    """Retrieve a saved custom syllabus with its chapters and topics."""
    try:
        return _syllabus_out(load_custom_syllabus(syllabus_id, current_user.id))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.put("/syllabi/custom/{syllabus_id}", response_model=CustomSyllabusOut, tags=["Custom Syllabus"])
def update_custom_syllabus_endpoint(
    syllabus_id: str,
    req: CustomSyllabusUpdateRequest,
    current_user: User = Depends(require_teacher),
):
    """Rename a saved syllabus or edit its extracted chapters/topics (re-indexes on change)."""
    try:
        syl = load_custom_syllabus(syllabus_id, current_user.id)
        return _syllabus_out(update_custom_syllabus(syl, req.name, req.grade, req.chapters))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to update the syllabus index: {e}")


@app.delete("/syllabi/custom/{syllabus_id}", tags=["Custom Syllabus"])
def delete_custom_syllabus_endpoint(syllabus_id: str, current_user: User = Depends(require_teacher)):
    """Delete a saved custom syllabus together with its FAISS index."""
    if not delete_custom_syllabus(syllabus_id, current_user.id):
        raise HTTPException(status_code=404, detail=f"Custom syllabus '{syllabus_id}' not found.")
    return {"message": f"Custom syllabus '{syllabus_id}' deleted successfully."}


def resolve_custom_syllabus_and_passages(
    request: PaperGenerateRequest | BatchPaperGenerateRequest,
    current_user: User,
) -> tuple[DocumentRecord, list[object], list[str]]:
    """
    Resolve a saved custom syllabus + the teacher's chapter/topic selection and retrieve
    passages from ONLY the selected chapters (exact chapter-metadata match).
    """
    try:
        syl = load_custom_syllabus(request.custom_syllabus_id, current_user.id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    try:
        chapters, topics = resolve_selection(syl, request.selected_chapters, request.selected_topics)
    except SelectionError as e:
        raise HTTPException(status_code=422, detail=str(e))

    grade_label = format_grade_display(request.academic_grade or request.grade)
    request.exam_type = ExamType.FINAL
    request.academic_grade = grade_label
    request.curriculum_grade = grade_label
    request.document_id = syl.syllabus_id
    request.custom_syllabus_name = syl.name
    request.permitted_units = chapters
    request.forbidden_units = []
    request.syllabus_scope = build_scope_text(syl.name, chapters, topics)
    selected_topic_names = [t for ch in chapters for t in (topics.get(ch) or [])]
    topic_text = ", ".join(selected_topic_names) if selected_topic_names else ", ".join(chapters)
    request.topic = topic_text if len(topic_text) <= 250 else topic_text[:247] + "..."

    query = f"{', '.join(chapters)}. {', '.join(selected_topic_names)}".strip()
    try:
        raw = retrieve_relevant_chunks(
            document_id=syl.syllabus_id,
            query=query,
            permitted_units=chapters,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    allowed = set(chapters)
    passages = [d for d in raw if (d.metadata or {}).get("chapter") in allowed]
    if not passages:
        raise HTTPException(
            status_code=422,
            detail="No content was found for the selected chapters in this custom syllabus.",
        )

    doc = DocumentRecord(
        document_id=syl.syllabus_id,
        original_filename=syl.source_filename or f"{syl.name}.txt",
        safe_filename=syl.syllabus_id,
        grade=syl.grade,
        subject=syl.subject,
        units=[c.title for c in syl.chapters],
        num_chunks=syl.num_chunks,
        has_index=syl.has_index,
    )
    return doc, passages, []


# ---------------------------------------------------------------------------
# Examination Scope & Curriculum Resolution Helper
# ---------------------------------------------------------------------------

def resolve_curriculum_and_passages(
    request: PaperGenerateRequest | BatchPaperGenerateRequest,
    current_user: Optional[User] = None,
) -> tuple[DocumentRecord, list[object], list[str]]:
    """
    Resolve curriculum class, examination scope, target document, and retrieved passages
    based on the examination type (prerequisite, midterm, or final).
    Uses the 5 permanently stored textbooks (Classes 1-5) by default.
    """
    if getattr(request, "custom_syllabus_id", None):
        if current_user is None:
            raise HTTPException(status_code=401, detail="Authentication required for custom syllabus.")
        return resolve_custom_syllabus_and_passages(request, current_user)

    exam_type = request.exam_type or ExamType.FINAL
    academic_class = format_grade_display(request.academic_grade or request.grade)
    acad_num = parse_grade_number(academic_class)
    is_auto_curriculum = not request.document_id or request.document_id.startswith("curriculum_class_")

    forbidden_units: list[str] = []

    if exam_type == ExamType.PREREQUISITE:
        if acad_num is not None and acad_num <= 1:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Prerequisite Examination is not available for {academic_class} "
                    "unless a previous-level curriculum textbook is provided."
                ),
            )

        try:
            prev_class = get_previous_class(academic_class)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))

        prev_num = parse_grade_number(prev_class)
        target_grade_str = format_grade_display(prev_class)

        # Check if previous class textbook exists in curriculum (Classes 1-5)
        tb_exists = False
        if prev_num is not None and 1 <= prev_num <= 5:
            tb_path = get_textbook_path(prev_num)
            tb_exists = tb_path.exists()

        if is_auto_curriculum and tb_exists and prev_num is not None:
            target_doc_id = get_curriculum_doc_id(prev_num)
            try:
                target_doc = load_document_record(target_doc_id)
            except FileNotFoundError:
                from app.curriculum import index_curriculum_textbook
                target_doc = index_curriculum_textbook(prev_num)
            all_ch = get_chapters_for_class(prev_num)
        else:
            # Check uploaded documents for previous class
            alt_doc = find_best_document_for_grade(grade=prev_class, subject=request.subject)
            if alt_doc is None and request.document_id:
                try:
                    selected_doc = load_document_record(request.document_id)
                    sel_num = parse_grade_number(selected_doc.grade or selected_doc.original_filename)
                    if sel_num == prev_num:
                        alt_doc = selected_doc
                except FileNotFoundError:
                    pass

            if alt_doc is not None:
                target_doc = alt_doc
                all_ch = order_chapters_by_textbook(alt_doc.units) if alt_doc.units else [f"Class {prev_class} Curriculum"]
            else:
                raise HTTPException(
                    status_code=422,
                    detail=(
                        f"Prerequisite Examination requires syllabus for Class {prev_class} {request.subject} "
                        f"(for students enrolled in {academic_class}). "
                        f"The selected document is for a different class, and no syllabus document was found for Class {prev_class}. "
                        f"Please upload or select the Class {prev_class} syllabus document."
                    ),
                )

        request.curriculum_grade = target_grade_str
        request.academic_grade = academic_class
        request.permitted_units = all_ch
        request.forbidden_units = []
        request.document_id = target_doc.document_id
        if not request.syllabus_scope:
            request.syllabus_scope = (
                f"Full syllabus of {target_grade_str} Mathematics "
                f"({len(all_ch)} chapters: {', '.join(all_ch)}) "
                f"[Prerequisite for {academic_class}]"
            )

        query_topic = (
            request.topic
            if (request.topic and request.topic not in ("Mathematics Curriculum", "General Math"))
            else f"{target_grade_str} Mathematics"
        )
        try:
            passages = retrieve_relevant_chunks(
                document_id=target_doc.document_id,
                query=query_topic,
                first_half_only=False,
                permitted_units=all_ch,
            )
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))

        return target_doc, passages, []

    elif exam_type == ExamType.MIDTERM:
        curr_grade_str = academic_class
        request.curriculum_grade = curr_grade_str
        request.academic_grade = academic_class

        if is_auto_curriculum and acad_num is not None and acad_num in CURRICULUM_CHAPTERS:
            tb_path = get_textbook_path(acad_num)
            if not tb_path.exists():
                raise HTTPException(
                    status_code=422,
                    detail=f"Textbook for Class {acad_num} not found at {tb_path}. Please provide the textbook.",
                )
            target_doc_id = get_curriculum_doc_id(acad_num)
            try:
                target_doc = load_document_record(target_doc_id)
            except FileNotFoundError:
                from app.curriculum import index_curriculum_textbook
                target_doc = index_curriculum_textbook(acad_num)
            request.document_id = target_doc.document_id
            permitted = get_midterm_chapters_for_class(acad_num)
            forbidden = get_second_half_chapters_for_class(acad_num)
            all_ch = get_chapters_for_class(acad_num)
        else:
            try:
                target_doc = load_document_record(request.document_id)
            except FileNotFoundError:
                target_doc = DocumentRecord(
                    document_id=request.document_id,
                    original_filename=f"{request.document_id}.pdf",
                    safe_filename=f"{request.document_id}.pdf",
                    grade=request.grade,
                    subject=request.subject,
                    num_chunks=1,
                    has_index=True,
                )
            if target_doc.units and len(target_doc.units) > 1:
                all_ch = order_chapters_by_textbook(target_doc.units)
                permitted = determine_midterm_scope(all_ch)
                forbidden = determine_second_half_scope(all_ch)
            else:
                all_ch = [request.topic or "Midterm Topics"]
                permitted = all_ch
                forbidden = []

        request.permitted_units = permitted
        request.forbidden_units = forbidden
        if not request.syllabus_scope:
            request.syllabus_scope = (
                f"First half of {curr_grade_str} syllabus ({len(permitted)} chapters: {', '.join(permitted)})"
            )

        query_topic = (
            request.topic
            if (request.topic and request.topic not in ("Mathematics Curriculum", "General Math"))
            else (", ".join(permitted) if permitted else f"{curr_grade_str} Mathematics first half")
        )
        try:
            passages = retrieve_relevant_chunks(
                document_id=target_doc.document_id,
                query=query_topic,
                first_half_only=True,
                permitted_units=permitted,
            )
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))

        return target_doc, passages, forbidden

    else:  # Final Examination
        curr_grade_str = academic_class
        request.curriculum_grade = curr_grade_str
        request.academic_grade = academic_class

        if is_auto_curriculum and acad_num is not None and acad_num in CURRICULUM_CHAPTERS:
            tb_path = get_textbook_path(acad_num)
            if not tb_path.exists():
                raise HTTPException(
                    status_code=422,
                    detail=f"Textbook for Class {acad_num} not found at {tb_path}. Please provide the textbook.",
                )
            target_doc_id = get_curriculum_doc_id(acad_num)
            try:
                target_doc = load_document_record(target_doc_id)
            except FileNotFoundError:
                from app.curriculum import index_curriculum_textbook
                target_doc = index_curriculum_textbook(acad_num)
            request.document_id = target_doc.document_id
            all_ch = get_chapters_for_class(acad_num)
        else:
            try:
                target_doc = load_document_record(request.document_id)
            except FileNotFoundError:
                target_doc = DocumentRecord(
                    document_id=request.document_id,
                    original_filename=f"{request.document_id}.pdf",
                    safe_filename=f"{request.document_id}.pdf",
                    grade=request.grade,
                    subject=request.subject,
                    num_chunks=1,
                    has_index=True,
                )
            all_ch = order_chapters_by_textbook(target_doc.units) if target_doc.units else ([request.topic] if request.topic else [f"{academic_class} Mathematics"])

        request.permitted_units = all_ch
        request.forbidden_units = []
        if not request.syllabus_scope:
            request.syllabus_scope = (
                f"Full syllabus of {curr_grade_str} Mathematics ({len(all_ch)} chapters: {', '.join(all_ch)})"
            )

        query_topic = (
            request.topic
            if (request.topic and request.topic not in ("Mathematics Curriculum", "General Math"))
            else f"{curr_grade_str} Mathematics"
        )
        try:
            passages = retrieve_relevant_chunks(
                document_id=target_doc.document_id,
                query=query_topic,
                first_half_only=False,
                permitted_units=all_ch,
            )
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))

        return target_doc, passages, []


# ---------------------------------------------------------------------------
# POST /papers/generate — generate a question paper
# ---------------------------------------------------------------------------

@app.post("/papers/generate", response_model=Paper, tags=["Papers"])
def generate_paper(request: PaperGenerateRequest, current_user: User = Depends(require_teacher)):
    """
    Generate a question paper from an uploaded document.
    Retrieves relevant passages based on the examination type, calls Gemini,
    validates the output, assembles the paper, and saves it as JSON.
    """
    target_doc, passages, forbidden_units = resolve_curriculum_and_passages(request, current_user)

    # Pre-generate the structured examination blueprint and assign chapter coverage
    blueprint = request.get_or_create_blueprint()
    assign_blueprint_chapter_coverage(
        blueprint,
        request.permitted_units or [],
        include_figure_questions=getattr(request, "include_figure_questions", True),
    )

    # Generate questions with Gemini
    try:
        questions = generate_questions(
            request=request,
            passages=passages,
            blueprint=blueprint,
            permitted_units=request.permitted_units,
            forbidden_units=forbidden_units,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))
    except Exception as e:
        logger.error("Question generation failed: %s", e)
        raise HTTPException(
            status_code=500,
            detail=f"Question generation failed unexpectedly: {e}",
        )

    # Syllabus scope validation: check no forbidden units included in mid-semester
    if forbidden_units:
        for q in questions:
            for fu in forbidden_units:
                if fu.lower() in q.question_text.lower() or fu.lower() in q.topic.lower():
                    logger.warning("Question '%s' touched forbidden mid-semester unit '%s'", q.question_id, fu)

    # Assemble and save the paper
    try:
        paper = assemble_paper(
            request=request,
            questions=questions,
            blueprint_id=blueprint.blueprint_id,
        )
        is_valid, validation_errors = validate_full_paper(paper)
        if not is_valid:
            logger.warning("Paper %s generated with validation issues: %s", paper.paper_id, validation_errors)
            critical_errors = [e for e in validation_errors if "forbidden" in e.lower() or "not belong to permitted" in e.lower()]
            if critical_errors:
                raise HTTPException(
                    status_code=422,
                    detail=f"Generated paper failed syllabus scope validation: {'; '.join(critical_errors)}"
                )
        save_paper(paper)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    return paper


# ---------------------------------------------------------------------------
# POST /papers/generate-batch — generate multiple unique paper versions
# ---------------------------------------------------------------------------

@app.post("/papers/generate-batch", response_model=BatchPaperResponse, tags=["Papers"])
def generate_batch_papers(request: BatchPaperGenerateRequest, current_user: User = Depends(require_teacher)):
    """
    Generate multiple unique paper versions in a single request sharing the same blueprint.
    Retrieves syllabus passages according to examination type, builds candidate pool,
    validates correctness, and selects disjoint sets of questions.
    """
    blueprint = request.get_or_create_blueprint()
    batch_id = uuid.uuid4().hex[:10]

    target_doc, passages, forbidden_units = resolve_curriculum_and_passages(request, current_user)
    assign_blueprint_chapter_coverage(
        blueprint,
        request.permitted_units or [],
        include_figure_questions=getattr(request, "include_figure_questions", True),
    )

    # Generate candidate question pool
    try:
        candidates, dups_rejected, math_discrepancies = generate_candidate_pool(
            blueprint=blueprint,
            passages=passages,
            grade=request.grade,
            subject=request.subject,
            topic=request.topic,
            num_versions=request.num_versions,
            exam_type=request.exam_type,
            academic_grade=request.academic_grade,
            curriculum_grade=request.curriculum_grade,
            syllabus_scope=request.syllabus_scope,
            permitted_units=request.permitted_units,
            forbidden_units=forbidden_units,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))
    except Exception as e:
        logger.error("Candidate generation failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Candidate generation failed: {e}")

    # Select questions without replacement across sets. If the pool cannot support all
    # requested sets without repeating questions, build as many fully-unique sets as
    # possible and tell the teacher — never silently duplicate questions.
    warnings: list[str] = []
    requested_sets = request.num_versions
    shortage: Optional[CandidatePoolShortageError] = None
    versions_questions = versions_sections = None
    sets_built = 0
    lowest_k = 1 if request.allow_fewer_sets else requested_sets
    for k in range(requested_sets, lowest_k - 1, -1):
        try:
            versions_questions, versions_sections = select_version_questions(
                candidate_pool=candidates,
                blueprint=blueprint,
                num_versions=k,
            )
            sets_built = k
            break
        except CandidatePoolShortageError as e:
            if shortage is None:
                shortage = e
    if sets_built == 0:
        e = shortage
        raise HTTPException(
            status_code=422,
            detail={
                "error": "CandidatePoolShortageError",
                "message": (
                    f"Not enough unique questions to build {requested_sets} sets without repeating "
                    f"questions. {e} Reduce the number of sets or widen the syllabus."
                    if not request.allow_fewer_sets
                    else f"Not enough unique questions could be generated to build even one paper. {e}"
                ),
                "section_id": e.section_id,
                "needed": e.needed,
                "available": e.available,
            },
        )
    if sets_built < requested_sets:
        warnings.append(
            f"Only {sets_built} of the {requested_sets} requested sets could be generated with fully unique "
            f"questions ({shortage}). "
            "No questions were duplicated across sets. Reduce the number of sets, widen the syllabus "
            "selection, or generate again to get more unique questions."
        )
    mode = getattr(request, "generation_mode", "sets")
    students_roster = getattr(request, "students", None) or []
    # If student roster was provided, default mode to individual
    if students_roster:
        mode = "individual"

    set_labels: list[str] = []
    for v_idx in range(sets_built):
        st = students_roster[v_idx] if v_idx < len(students_roster) else None
        set_labels.append(format_set_or_student_label(v_idx, mode=mode, student=st))

    # Assemble and persist each set
    saved_papers: list[Paper] = []
    paper_ids: list[str] = []

    for v_idx in range(sets_built):
        v_num = v_idx + 1
        v_qs = versions_questions[v_idx]
        v_secs = versions_sections[v_idx]
        st = students_roster[v_idx] if v_idx < len(students_roster) else None
        st_name = st.student_name if st else None
        st_roll = st.roll_number if st else None

        single_req = PaperGenerateRequest(
            document_id=request.document_id,
            grade=request.grade,
            subject=request.subject,
            topic=request.topic,
            exam_type=request.exam_type,
            academic_grade=request.academic_grade,
            curriculum_grade=request.curriculum_grade,
            syllabus_scope=request.syllabus_scope,
            permitted_units=request.permitted_units,
            forbidden_units=request.forbidden_units,
            difficulty=request.difficulty,
            num_questions=blueprint.total_questions,
            total_marks=blueprint.total_marks,
            num_versions=sets_built,
            custom_syllabus_id=request.custom_syllabus_id,
            custom_syllabus_name=request.custom_syllabus_name,
        )

        paper = assemble_paper(
            request=single_req,
            questions=v_qs,
            sections=v_secs,
            batch_id=batch_id,
            version_number=v_num,
            num_versions=sets_built,
            blueprint_id=blueprint.blueprint_id,
            set_label=set_labels[v_idx],
            student_name=st_name,
            roll_number=st_roll,
            generation_mode=mode,
        )
        is_valid, validation_errors = validate_full_paper(paper)
        if not is_valid:
            logger.warning("Batch paper %s has validation warnings: %s", set_labels[v_idx], validation_errors)
        save_paper(paper)
        saved_papers.append(paper)
        paper_ids.append(paper.paper_id)

    # 5. Validation summary
    math_verified = sum(
        1 for p in saved_papers for q in p.questions
        if q.math_verification and q.math_verification.status == MathCheckStatus.VERIFIED
    )
    is_unique, uniq_violations = validate_batch_papers_uniqueness(saved_papers)
    if not is_unique:
        logger.warning("Uniqueness validation issues in batch: %s", uniq_violations)
        if not request.allow_fewer_sets:
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "UniquenessValidationError",
                    "message": f"Could not guarantee 100% unique questions across all requested papers: {'; '.join(uniq_violations[:3])}. Reduce the number of sets or widen the syllabus selection.",
                    "violations": uniq_violations,
                },
            )
        else:
            warnings.append(f"Uniqueness notice: {'; '.join(uniq_violations[:2])}")

    summary = BatchValidationSummary(
        total_candidates_generated=len(candidates) + dups_rejected + math_discrepancies,
        valid_candidates_count=len(candidates),
        duplicate_candidates_rejected=dups_rejected,
        math_verified_count=math_verified,
        math_discrepancies_rejected=math_discrepancies,
        is_unique=is_unique,
        all_versions_valid=all(p.num_questions == blueprint.total_questions for p in saved_papers),
        details=(
            f"Generated {len(saved_papers)} unique papers ({blueprint.total_questions} questions, "
            f"{blueprint.total_marks} marks each) under blueprint '{blueprint.title}'."
            + (f" Uniqueness notices: {'; '.join(uniq_violations)}" if uniq_violations else "")
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
        set_labels=set_labels,
        students=students_roster if students_roster else None,
        generation_mode=mode,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# GET /papers — list all saved papers
# ---------------------------------------------------------------------------

@app.get("/papers", response_model=list[PaperSummary], tags=["Papers"])
def get_papers(current_user: User = Depends(require_teacher)):
    """List all saved question papers."""
    return list_papers()


# ---------------------------------------------------------------------------
# GET /papers/{paper_id} — retrieve one saved paper
# ---------------------------------------------------------------------------

@app.get("/papers/{paper_id}", response_model=Paper, tags=["Papers"])
def get_paper(paper_id: str, current_user: User = Depends(require_teacher)):
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
def download_paper_pdf(paper_id: str, include_answers: bool = True, current_user: User = Depends(require_teacher)):
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

    safe_title = (
        paper.title.replace("—", "-")
        .replace("–", "-")
        .replace(" ", "_")
        .encode("ascii", "ignore")
        .decode()
        .strip("_")
    )
    if not safe_title:
        safe_title = f"paper_{paper.paper_id}"
    version = "solutions" if include_answers else "student"
    filename = f"{safe_title}_{version}.pdf"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ---------------------------------------------------------------------------
# GET /papers/batch/{batch_id}/pdfs — download every set of a batch as one ZIP of PDFs
# ---------------------------------------------------------------------------

@app.get("/papers/batch/{batch_id}/pdfs", tags=["Papers"])
def download_batch_pdfs(batch_id: str, include_answers: bool = True, current_user: User = Depends(require_teacher)):
    """Download a separate PDF for every set (Set A, B, C...) or student of a batch, bundled in a ZIP."""
    summaries = [p for p in list_papers() if p.batch_id == batch_id]
    if not summaries:
        raise HTTPException(status_code=404, detail=f"No papers found for batch '{batch_id}'.")
    summaries.sort(key=lambda p: p.version_number)
    kind = "solutions" if include_answers else "student"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        index_entries: list[str] = []
        for summ in summaries:
            try:
                paper = load_paper(summ.paper_id)
            except (FileNotFoundError, ValueError):
                continue

            if paper.student_name:
                clean_name = re.sub(r"[^a-zA-Z0-9_\-]", "_", paper.student_name).strip("_")
                roll_part = f"_Roll_{paper.roll_number}" if paper.roll_number else ""
                filename = f"Student_{paper.version_number:02d}_{clean_name}{roll_part}_{kind}.pdf"
            elif str(paper.set_label or "").lower().startswith("student "):
                clean_lbl = str(paper.set_label).replace(" ", "_")
                filename = f"{clean_lbl}_{kind}.pdf"
            else:
                label = paper.set_label or str(paper.version_number)
                filename = f"Set_{label}_{kind}.pdf"

            zf.writestr(filename, generate_pdf(paper, include_answers=include_answers))
            st_info = f"Student: {paper.student_name or 'Unassigned'}" + (f" | Roll No: {paper.roll_number}" if paper.roll_number else "")
            index_entries.append(f"{paper.version_number}. [{paper.set_label or f'Version {paper.version_number}'}] {st_info} -> {filename}")

        if include_answers and any(getattr(s, "student_name", None) for s in summaries) and index_entries:
            index_doc = "EXAMINATION PAPERS & ANSWER KEYS BATCH INDEX\n" + "=" * 55 + "\n\n" + "\n".join(index_entries)
            zf.writestr("Batch_Master_Index.txt", index_doc.encode("utf-8"))

    zip_filename = (
        f"individual_student_papers_{kind}.zip"
        if any(s.student_name or str(s.set_label).startswith("Student") for s in summaries)
        else f"question_paper_sets_{kind}.zip"
    )
    return Response(
        content=buf.getvalue(),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{zip_filename}"'},
    )


# ---------------------------------------------------------------------------
# DELETE /papers/{paper_id} — delete a saved question paper
# ---------------------------------------------------------------------------

@app.delete("/papers/{paper_id}", tags=["Papers"])
def delete_paper_endpoint(paper_id: str, current_user: User = Depends(require_teacher)):
    """Delete a saved question paper."""
    deleted = delete_paper(paper_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Paper '{paper_id}' not found.")
    return {"message": f"Paper '{paper_id}' deleted successfully."}


# ===========================================================================
# STUDENT PERFORMANCE & PERSONALIZED PRACTICE WORKSHEETS
# ===========================================================================

from pydantic import BaseModel, Field

from app.performance import (
    EvaluationModuleAdapter,
    EvaluationResultRecord,
    PerformanceCategory,
    StudentPerformanceProfile,
    StudentTopicScore,
    generate_simulated_profiles,
    list_performance_profiles,
    load_performance_profile,
    save_performance_profile,
)
from app.worksheets import (
    PersonalizedWorksheet,
    WorksheetGenerateRequest,
    allocate_worksheet_questions,
    generate_personalized_worksheet,
    list_worksheets,
    load_worksheet,
)
from app.worksheet_pdf import generate_worksheet_pdf, generate_worksheets_zip


class BatchWorksheetGenerateRequest(BaseModel):
    grade: str = Field(..., description="Target class/grade")
    subject: str = Field(default="Mathematics", description="Subject")
    student_profiles: list[StudentPerformanceProfile] = Field(..., description="List of student performance profiles")
    num_questions: int = Field(default=10, ge=3, le=30, description="Questions per worksheet")
    include_figures: bool = Field(default=True, description="Whether to include visual figure questions")
    custom_syllabus_id: Optional[str] = Field(default=None)


class BatchWorksheetZipRequest(BaseModel):
    worksheet_ids: list[str] = Field(..., min_length=1)
    include_answers: bool = Field(default=False)


@app.get("/worksheets/simulated-performance", response_model=list[StudentPerformanceProfile], tags=["Personalized Worksheets"])
def get_simulated_performance(
    grade: str = Query(default="5", description="Class grade 1-5"),
    count: int = Query(default=3, ge=1, le=10, description="Number of sample students"),
    assessment_id: Optional[str] = Query(default=None, description="Assessment ID reference"),
    current_user: User = Depends(require_teacher),
):
    """
    Generate realistic simulated performance data across curriculum chapters.
    Clearly labeled with is_simulated=True (DEMO / SIMULATED DATA).
    """
    students_demo = [
        {"student_name": "Aarav Sharma", "roll_number": "101"},
        {"student_name": "Diya Patel", "roll_number": "102"},
        {"student_name": "Mohammed Zaid", "roll_number": "103"},
        {"student_name": "Priya Verma", "roll_number": "104"},
        {"student_name": "Rohan Gupta", "roll_number": "105"},
    ]
    target_students = students_demo[:count]
    profiles = generate_simulated_profiles(
        grade=grade,
        students=target_students,
        assessment_id=assessment_id or f"assessment_grade_{grade}",
        assessment_title=f"Class {grade} Mid-Semester Assessment (Demo)",
    )
    for p in profiles:
        save_performance_profile(p)
    return profiles


@app.post("/worksheets/analyze-profile", response_model=StudentPerformanceProfile, tags=["Personalized Worksheets"])
def analyze_student_profile(
    profile: StudentPerformanceProfile,
    current_user: User = Depends(require_teacher),
):
    """
    Calculate and recompute topic-wise percentages, categories, strengths, and focus areas.
    """
    profile.recompute_summaries()
    save_performance_profile(profile)
    return profile


@app.post("/worksheets/generate", response_model=PersonalizedWorksheet, tags=["Personalized Worksheets"])
def generate_single_worksheet(
    req: WorksheetGenerateRequest,
    current_user: User = Depends(require_teacher),
):
    """
    Generate an individualized practice worksheet tailored to an individual student's performance.
    """
    try:
        ws = generate_personalized_worksheet(req)
        return ws
    except Exception as e:
        logger.error("Worksheet generation failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to generate worksheet: {e}")


@app.post("/worksheets/generate-batch", tags=["Personalized Worksheets"])
def generate_batch_worksheets(
    req: BatchWorksheetGenerateRequest,
    current_user: User = Depends(require_teacher),
):
    """
    Generate personalized practice worksheets for multiple students in a classroom.
    """
    if not req.student_profiles:
        raise HTTPException(status_code=400, detail="Must provide at least one student profile.")

    batch_id = uuid.uuid4().hex[:10]
    worksheets: list[PersonalizedWorksheet] = []

    for prof in req.student_profiles:
        single_req = WorksheetGenerateRequest(
            grade=req.grade,
            subject=req.subject,
            student_profile=prof,
            num_questions=req.num_questions,
            include_figures=req.include_figures,
            custom_syllabus_id=req.custom_syllabus_id,
        )
        ws = generate_personalized_worksheet(single_req)
        worksheets.append(ws)

    return {
        "batch_id": batch_id,
        "total_generated": len(worksheets),
        "worksheet_ids": [w.worksheet_id for w in worksheets],
        "worksheets": worksheets,
    }


@app.get("/worksheets", tags=["Personalized Worksheets"])
def list_saved_worksheets(
    grade: Optional[str] = Query(default=None),
    current_user: User = Depends(require_teacher),
):
    """List all saved personalized practice worksheets."""
    return list_worksheets(grade=grade)


@app.get("/worksheets/{worksheet_id}", response_model=PersonalizedWorksheet, tags=["Personalized Worksheets"])
def get_worksheet_by_id(
    worksheet_id: str,
    current_user: User = Depends(require_teacher),
):
    """Retrieve a saved personalized worksheet."""
    ws = load_worksheet(worksheet_id)
    if not ws:
        raise HTTPException(status_code=404, detail=f"Worksheet '{worksheet_id}' not found.")
    return ws


@app.get("/worksheets/{worksheet_id}/pdf", tags=["Personalized Worksheets"])
def download_worksheet_pdf(
    worksheet_id: str,
    include_answers: bool = Query(default=False, description="False for student practice sheet, True for teacher key"),
    current_user: User = Depends(require_teacher),
):
    """Download a worksheet as a PDF."""
    ws = load_worksheet(worksheet_id)
    if not ws:
        raise HTTPException(status_code=404, detail=f"Worksheet '{worksheet_id}' not found.")

    pdf_bytes = generate_worksheet_pdf(ws, include_answers=include_answers)
    safe_name = ws.student_name.replace(" ", "_")
    suffix = "solutions" if include_answers else "worksheet"
    filename = f"Worksheet_{safe_name}_{suffix}.pdf"

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'inline; filename="{filename}"'},
    )


@app.post("/worksheets/batch-zip", tags=["Personalized Worksheets"])
def download_worksheets_zip(
    req: BatchWorksheetZipRequest,
    current_user: User = Depends(require_teacher),
):
    """Download multiple student worksheets as a single ZIP bundle."""
    worksheets: list[PersonalizedWorksheet] = []
    for wid in req.worksheet_ids:
        ws = load_worksheet(wid)
        if ws:
            worksheets.append(ws)

    if not worksheets:
        raise HTTPException(status_code=404, detail="No matching worksheets found for download.")

    zip_bytes = generate_worksheets_zip(worksheets, include_answers=req.include_answers)
    suffix = "teacher_solutions" if req.include_answers else "student_worksheets"
    filename = f"Personalized_Worksheets_{suffix}_{datetime.now().strftime('%Y%m%d')}.zip"

    return Response(
        content=zip_bytes,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.delete("/worksheets/{worksheet_id}", tags=["Personalized Worksheets"])
def delete_worksheet_endpoint(
    worksheet_id: str,
    current_user: User = Depends(require_teacher),
):
    """Delete a saved worksheet."""
    f_path = settings.worksheets_dir / f"{worksheet_id}.json"
    if not f_path.exists():
        raise HTTPException(status_code=404, detail=f"Worksheet '{worksheet_id}' not found.")
    try:
        f_path.unlink()
        return {"message": f"Worksheet '{worksheet_id}' deleted successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete worksheet: {e}")


