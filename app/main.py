# app/main.py
# FastAPI application entry point with all API routes.
# Protected with enterprise authentication, session tokens, and admin user controls.

import logging
import uuid
from typing import Optional
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, Response, status, Depends
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
# Examination Scope & Curriculum Resolution Helper
# ---------------------------------------------------------------------------

def resolve_curriculum_and_passages(
    request: PaperGenerateRequest | BatchPaperGenerateRequest,
) -> tuple[DocumentRecord, list[object], list[str]]:
    """
    Resolve curriculum class, examination scope, target document, and retrieved passages
    based on the examination type (prerequisite, midterm, or final).
    Uses the 5 permanently stored textbooks (Classes 1-5) by default.
    """
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
    target_doc, passages, forbidden_units = resolve_curriculum_and_passages(request)

    # Pre-generate the structured examination blueprint and assign chapter coverage
    blueprint = request.get_or_create_blueprint()
    assign_blueprint_chapter_coverage(blueprint, request.permitted_units or [])

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

    target_doc, passages, forbidden_units = resolve_curriculum_and_passages(request)
    assign_blueprint_chapter_coverage(blueprint, request.permitted_units or [])

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

    # Select questions without replacement across versions
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

    # Assemble and persist each version
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
            exam_type=request.exam_type,
            academic_grade=request.academic_grade,
            curriculum_grade=request.curriculum_grade,
            syllabus_scope=request.syllabus_scope,
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
        is_valid, validation_errors = validate_full_paper(paper)
        if not is_valid:
            logger.warning("Batch paper version %d has validation warnings: %s", v_num, validation_errors)
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
# DELETE /papers/{paper_id} — delete a saved question paper
# ---------------------------------------------------------------------------

@app.delete("/papers/{paper_id}", tags=["Papers"])
def delete_paper_endpoint(paper_id: str, current_user: User = Depends(require_teacher)):
    """Delete a saved question paper."""
    deleted = delete_paper(paper_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Paper '{paper_id}' not found.")
    return {"message": f"Paper '{paper_id}' deleted successfully."}

