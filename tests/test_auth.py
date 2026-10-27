# tests/test_auth.py
# Comprehensive security and authentication tests for QPGen.
# Verifies secure login, session tokens, rate limiting, no public registration,
# and administrator user-management controls.

import pytest
from starlette.testclient import TestClient
from app.main import app
from app.config import settings
from app.auth import (
    init_auth_db,
    get_db,
    hash_password,
    verify_password,
    COOKIE_NAME,
    UserRole,
    get_current_user,
    require_teacher,
    require_admin,
)


@pytest.fixture(autouse=True)
def setup_auth_test_env(tmp_path, monkeypatch):
    """Isolate authentication database in a temporary directory for tests."""
    app.dependency_overrides.pop(get_current_user, None)
    app.dependency_overrides.pop(require_teacher, None)
    app.dependency_overrides.pop(require_admin, None)
    test_db = tmp_path / "test_users.db"
    monkeypatch.setattr(settings, "ADMIN_USERNAME", "admin")
    monkeypatch.setattr(settings, "ADMIN_EMAIL", "admin@qpgen.local")
    monkeypatch.setattr(settings, "ADMIN_PASSWORD", "AdminTestPass2026!")
    monkeypatch.setattr(settings, "RATE_LIMIT_FAILED_LOGINS", 5)
    monkeypatch.setattr(settings, "RATE_LIMIT_WINDOW_SECONDS", 900)
    monkeypatch.setattr(settings, "USERS_DB_PATH", test_db)
    monkeypatch.setattr("app.main.ensure_all_curriculum_textbooks_indexed", lambda: None)
    init_auth_db(test_db)
    yield
    app.dependency_overrides.pop(get_current_user, None)
    app.dependency_overrides.pop(require_teacher, None)
    app.dependency_overrides.pop(require_admin, None)


@pytest.fixture
def auth_client():
    """Test client without default dependency overrides so real auth is tested."""
    return TestClient(app)


# ---------------------------------------------------------------------------
# 1. Password Hashing & Verification
# ---------------------------------------------------------------------------
def test_password_hashing_and_verification():
    raw_pass = "MySecretPassword123!"
    hashed = hash_password(raw_pass)

    assert hashed.startswith("pbkdf2_sha256$200000$")
    assert raw_pass not in hashed
    assert verify_password(raw_pass, hashed) is True
    assert verify_password("WrongPassword!", hashed) is False
    assert verify_password("", hashed) is False


# ---------------------------------------------------------------------------
# 2. Login Page Availability
# ---------------------------------------------------------------------------
def test_login_page_renders(auth_client):
    resp = auth_client.get("/login")
    assert resp.status_code == 200
    assert "Sign in to your account" in resp.text
    assert "No Public Registration" in resp.text


# ---------------------------------------------------------------------------
# 3. Successful Login & Session Cookie
# ---------------------------------------------------------------------------
def test_successful_admin_login(auth_client):
    payload = {
        "identifier": "admin",
        "password": "AdminTestPass2026!",
    }
    resp = auth_client.post("/auth/login", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    assert data["status"] == "ok"
    assert data["user"]["username"] == "admin"
    assert data["user"]["role"] == "admin"
    assert "password" not in data["user"]
    assert "password_hash" not in data["user"]

    # Verify session cookie was set
    assert COOKIE_NAME in resp.cookies
    session_token = resp.cookies[COOKIE_NAME]
    assert len(session_token) > 20

    # Verify /auth/me returns current user
    me_resp = auth_client.get("/auth/me", cookies={COOKIE_NAME: session_token})
    assert me_resp.status_code == 200
    assert me_resp.json()["username"] == "admin"


# ---------------------------------------------------------------------------
# 4. Incorrect Credentials & Rate Limiting
# ---------------------------------------------------------------------------
def test_invalid_credentials_and_rate_limiting(auth_client):
    # Wrong password
    bad_payload = {"identifier": "admin", "password": "WrongPassword"}
    resp = auth_client.post("/auth/login", json=bad_payload)
    assert resp.status_code == 401
    assert "Invalid username/email or password" in resp.json()["detail"]

    # Trigger rate limit (5 failed attempts)
    for _ in range(4):
        auth_client.post("/auth/login", json=bad_payload)

    # 6th attempt must be blocked by rate limiter
    blocked_resp = auth_client.post("/auth/login", json=bad_payload)
    assert blocked_resp.status_code == 429
    assert "Too many failed login attempts" in blocked_resp.json()["detail"]


# ---------------------------------------------------------------------------
# 5. Protected Endpoint Access Rejection
# ---------------------------------------------------------------------------
def test_unauthenticated_api_requests_return_401(auth_client):
    # Direct access to protected APIs without credentials must return 401
    resp = auth_client.get("/papers", headers={"Accept": "application/json"})
    assert resp.status_code == 401

    resp_curr = auth_client.get("/curriculum/status", headers={"Accept": "application/json"})
    assert resp_curr.status_code == 401

    resp_gen = auth_client.post(
        "/papers/generate",
        json={"grade": "1", "subject": "Mathematics", "num_questions": 1, "total_marks": 2},
        headers={"Accept": "application/json"},
    )
    assert resp_gen.status_code == 401


def test_unauthenticated_browser_request_redirects_to_login(auth_client):
    # HTML browser navigation to root dashboard must redirect to /login
    resp = auth_client.get("/", headers={"Accept": "text/html,application/xhtml+xml"}, follow_redirects=False)
    assert resp.status_code in (302, 303, 307)
    assert resp.headers["location"] == "/login"


# ---------------------------------------------------------------------------
# 6. Logout and Session Invalidation
# ---------------------------------------------------------------------------
def test_logout_invalidates_session(auth_client):
    # Log in first
    login_resp = auth_client.post("/auth/login", json={"identifier": "admin", "password": "AdminTestPass2026!"})
    assert login_resp.status_code == 200
    token = login_resp.cookies[COOKIE_NAME]

    # Log out
    logout_resp = auth_client.post("/auth/logout", cookies={COOKIE_NAME: token})
    assert logout_resp.status_code == 200

    # Old token must now be invalid
    after_resp = auth_client.get("/auth/me", cookies={COOKIE_NAME: token}, headers={"Accept": "application/json"})
    assert after_resp.status_code == 401


# ---------------------------------------------------------------------------
# 7. No Public Registration
# ---------------------------------------------------------------------------
def test_public_registration_is_strictly_forbidden(auth_client):
    # Attempting to call /register or /signup must return 404
    for endpoint in ["/register", "/signup"]:
        assert auth_client.get(endpoint).status_code == 404
        assert auth_client.post(endpoint, json={"username": "hacker"}).status_code == 404


# ---------------------------------------------------------------------------
# 8. Administrator User Management (Create, Deactivate, Reset Password)
# ---------------------------------------------------------------------------
def test_admin_can_manage_teacher_accounts():
    app.dependency_overrides.pop(get_current_user, None)
    admin_client = TestClient(app)
    teacher_client = TestClient(app)

    # 1. Log in as admin
    admin_login = admin_client.post("/auth/login", json={"identifier": "admin", "password": "AdminTestPass2026!"})
    assert admin_login.status_code == 200

    # 2. Create teacher account
    new_teacher_data = {
        "username": "teacher_sharma",
        "email": "sharma@school.edu",
        "full_name": "R. Sharma",
        "password": "TeacherPass2026!",
        "role": "teacher",
    }
    create_resp = admin_client.post("/admin/users", json=new_teacher_data)
    assert create_resp.status_code == 201
    created_user = create_resp.json()
    teacher_id = created_user["id"]
    assert created_user["username"] == "teacher_sharma"
    assert created_user["role"] == "teacher"
    assert created_user["is_active"] is True

    # 3. Verify teacher can log in
    t_login = teacher_client.post("/auth/login", json={"identifier": "teacher_sharma", "password": "TeacherPass2026!"})
    assert t_login.status_code == 200

    # 4. Teacher cannot access admin endpoints (403 Forbidden)
    t_forbidden = teacher_client.get("/admin/users")
    assert t_forbidden.status_code == 403

    # 5. Admin can deactivate teacher
    deact_resp = admin_client.patch(f"/admin/users/{teacher_id}/status", json={"is_active": False})
    assert deact_resp.status_code == 200
    assert deact_resp.json()["is_active"] is False

    # 6. Deactivated teacher cannot use existing session
    t_blocked_session = teacher_client.get("/auth/me", headers={"Accept": "application/json"})
    assert t_blocked_session.status_code == 401

    # 7. Deactivated teacher cannot log in
    t_blocked_login = teacher_client.post("/auth/login", json={"identifier": "teacher_sharma", "password": "TeacherPass2026!"})
    assert t_blocked_login.status_code == 403
    assert "deactivated" in t_blocked_login.json()["detail"].lower()

    # 8. Admin reactivates teacher and resets password
    admin_client.patch(f"/admin/users/{teacher_id}/status", json={"is_active": True})
    reset_resp = admin_client.post(
        f"/admin/users/{teacher_id}/reset-password",
        json={"new_password": "NewTeacherPass2026!"},
    )
    assert reset_resp.status_code == 200

    # 9. Teacher logs in with new password
    new_login = teacher_client.post("/auth/login", json={"identifier": "teacher_sharma", "password": "NewTeacherPass2026!"})
    assert new_login.status_code == 200


# ---------------------------------------------------------------------------
# 9. Strict Role Separation: Admin Forbidden from Question Papers & PDFs
# ---------------------------------------------------------------------------
def test_admin_strictly_forbidden_from_paper_generation_and_downloads():
    """Verify administrator cannot call teacher-only APIs or download papers."""
    app.dependency_overrides.pop(get_current_user, None)
    from app.auth import require_teacher
    app.dependency_overrides.pop(require_teacher, None)

    admin_client = TestClient(app)
    admin_login = admin_client.post("/auth/login", json={"identifier": "admin", "password": "AdminTestPass2026!"})
    assert admin_login.status_code == 200

    # 1. POST /papers/generate -> 403 Forbidden
    resp = admin_client.post(
        "/papers/generate",
        json={"grade": "1", "subject": "Mathematics", "num_questions": 1, "total_marks": 2},
    )
    assert resp.status_code == 403
    assert "restricted to teacher accounts" in resp.json()["detail"].lower()

    # 2. POST /papers/generate-batch -> 403 Forbidden
    resp = admin_client.post(
        "/papers/generate-batch",
        json={"grade": "1", "subject": "Mathematics", "num_versions": 2, "num_questions": 1, "total_marks": 2},
    )
    assert resp.status_code == 403
    assert "restricted to teacher accounts" in resp.json()["detail"].lower()

    # 3. GET /papers -> 403 Forbidden
    resp = admin_client.get("/papers")
    assert resp.status_code == 403
    assert "restricted to teacher accounts" in resp.json()["detail"].lower()

    # 4. GET /papers/{id} -> 403 Forbidden
    resp = admin_client.get("/papers/paper-123")
    assert resp.status_code == 403
    assert "restricted to teacher accounts" in resp.json()["detail"].lower()

    # 5. GET /papers/{id}/pdf -> 403 Forbidden
    resp = admin_client.get("/papers/paper-123/pdf?include_answers=true")
    assert resp.status_code == 403
    assert "restricted to teacher accounts" in resp.json()["detail"].lower()

    resp_student = admin_client.get("/papers/paper-123/pdf?include_answers=false")
    assert resp_student.status_code == 403
    assert "restricted to teacher accounts" in resp_student.json()["detail"].lower()

    # 6. DELETE /papers/{id} -> 403 Forbidden
    resp = admin_client.delete("/papers/paper-123")
    assert resp.status_code == 403
    assert "restricted to teacher accounts" in resp.json()["detail"].lower()

    # 7. GET /curriculum/status -> 403 Forbidden
    resp = admin_client.get("/curriculum/status")
    assert resp.status_code == 403
    assert "restricted to teacher accounts" in resp.json()["detail"].lower()

    # 8. Document APIs -> 403 Forbidden
    resp = admin_client.get("/documents")
    assert resp.status_code == 403
    assert "restricted to teacher accounts" in resp.json()["detail"].lower()


# ---------------------------------------------------------------------------
# 10. Role-Based Navigation & Page Redirection
# ---------------------------------------------------------------------------
def test_role_based_page_redirection_and_access():
    """Verify administrator and teacher browser navigation rules."""
    app.dependency_overrides.pop(get_current_user, None)
    from app.auth import require_teacher
    app.dependency_overrides.pop(require_teacher, None)

    admin_client = TestClient(app)
    admin_client.post("/auth/login", json={"identifier": "admin", "password": "AdminTestPass2026!"})

    # Admin visits / -> Redirected to /admin
    resp_root = admin_client.get("/", follow_redirects=False)
    assert resp_root.status_code in (302, 303, 307)
    assert resp_root.headers["location"] == "/admin"

    # Admin visits /admin -> 200 OK
    resp_admin = admin_client.get("/admin")
    assert resp_admin.status_code == 200
    assert "Access Management" in resp_admin.text

    # Provision a teacher
    admin_client.post("/admin/users", json={
        "username": "teacher_verma",
        "email": "verma@school.edu",
        "full_name": "A. Verma",
        "password": "TeacherPass2026!",
        "role": "teacher",
    })

    teacher_client = TestClient(app)
    teacher_client.post("/auth/login", json={"identifier": "teacher_verma", "password": "TeacherPass2026!"})

    # Teacher visits / -> 200 OK (Teacher Portal)
    resp_t_root = teacher_client.get("/")
    assert resp_t_root.status_code == 200
    assert "QPGen" in resp_t_root.text

    # Teacher visits /admin -> 403 Forbidden
    resp_t_admin = teacher_client.get("/admin")
    assert resp_t_admin.status_code == 403
    assert "administrator privileges required" in resp_t_admin.json()["detail"].lower()

    # Teacher visits /admin/users -> 403 Forbidden
    resp_t_users = teacher_client.get("/admin/users")
    assert resp_t_users.status_code == 403

    # Teacher can access curriculum and papers
    resp_curr = teacher_client.get("/curriculum/status")
    assert resp_curr.status_code == 200

    resp_papers = teacher_client.get("/papers")
    assert resp_papers.status_code == 200

