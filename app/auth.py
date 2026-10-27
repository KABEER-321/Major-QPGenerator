# app/auth.py
# Enterprise authentication, session management, and admin user controls for QPGen.
# Protects all dashboards and sensitive APIs with zero public registration.

from __future__ import annotations

import os
import sqlite3
import secrets
import hashlib
import time
import logging
import uuid
from enum import Enum
from pathlib import Path
from typing import Optional
from datetime import datetime, timezone

from pydantic import BaseModel, Field, field_validator
from fastapi import Request, HTTPException, status, Depends
from app.config import settings

logger = logging.getLogger("qpgen.auth")

COOKIE_NAME = "qpgen_session"


class UserRole(str, Enum):
    ADMIN = "admin"
    TEACHER = "teacher"


class User(BaseModel):
    """Safe user profile representation — never exposes password hashes."""
    id: str
    username: str
    email: str
    full_name: str
    role: UserRole
    is_active: bool
    created_at: str
    last_login_at: Optional[str] = None


class LoginRequest(BaseModel):
    identifier: str = Field(..., description="Username or email address")
    password: str = Field(..., min_length=1, description="Account password")


class UserCreateRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=50, description="Unique username")
    email: str = Field(..., min_length=5, max_length=100, description="Unique email address")
    full_name: str = Field(..., min_length=2, max_length=100, description="Teacher full name")
    password: str = Field(..., min_length=8, description="Initial password")
    role: UserRole = Field(default=UserRole.TEACHER, description="Account role")

    @field_validator("username")
    @classmethod
    def validate_username(cls, v: str) -> str:
        clean = v.strip().lower()
        if not clean.replace("_", "").replace("-", "").isalnum():
            raise ValueError("Username can only contain alphanumeric characters, underscores, and hyphens.")
        return clean

    @field_validator("email")
    @classmethod
    def validate_email(cls, v: str) -> str:
        clean = v.strip().lower()
        if "@" not in clean or "." not in clean:
            raise ValueError("Please provide a valid email address.")
        return clean


class UserStatusUpdateRequest(BaseModel):
    is_active: bool


class UserResetPasswordRequest(BaseModel):
    new_password: str = Field(..., min_length=8, description="New account password")


class AuthenticationRedirectException(Exception):
    """Raised when an unauthenticated HTML browser navigation occurs."""
    pass


# ---------------------------------------------------------------------------
# Password Hashing & Verification (PBKDF2-HMAC-SHA256, 200,000 rounds)
# ---------------------------------------------------------------------------
def hash_password(plain_password: str) -> str:
    """Securely hash a password using PBKDF2 with HMAC-SHA256 and a random salt."""
    if not plain_password:
        raise ValueError("Password cannot be empty.")
    salt = secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", plain_password.encode("utf-8"), salt, 200000)
    return f"pbkdf2_sha256$200000${salt.hex()}${dk.hex()}"


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plaintext password against the stored PBKDF2 hash using constant-time comparison."""
    if not plain_password or not hashed_password:
        return False
    try:
        parts = hashed_password.split("$")
        if len(parts) != 4 or parts[0] != "pbkdf2_sha256":
            return False
        iterations = int(parts[1])
        salt = bytes.fromhex(parts[2])
        expected_dk = bytes.fromhex(parts[3])
        actual_dk = hashlib.pbkdf2_hmac("sha256", plain_password.encode("utf-8"), salt, iterations)
        return secrets.compare_digest(actual_dk, expected_dk)
    except Exception as e:
        logger.error("Error during password verification: %s", e)
        return False


# ---------------------------------------------------------------------------
# SQLite Database & Schema Initialization
# ---------------------------------------------------------------------------
def get_db(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """Get a SQLite database connection with row factory configured."""
    target_path = db_path or settings.users_db_path
    target_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(target_path), timeout=10.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_auth_db(db_path: Optional[Path] = None) -> None:
    """Initialize database tables and create initial administrator if not present."""
    with get_db(db_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                full_name TEXT NOT NULL,
                role TEXT NOT NULL,
                is_active INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                last_login_at TEXT
            );
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                created_at REAL NOT NULL,
                expires_at REAL NOT NULL,
                ip_address TEXT,
                user_agent TEXT,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            );
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id);
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_sessions_expires_at ON sessions(expires_at);
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS login_attempts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ip_address TEXT NOT NULL,
                identifier TEXT NOT NULL,
                timestamp REAL NOT NULL,
                success INTEGER NOT NULL
            );
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_login_attempts_lookup 
            ON login_attempts(identifier, ip_address, timestamp);
        """)

        # Ensure an initial administrator exists
        cur = conn.execute("SELECT id FROM users WHERE role = 'admin' LIMIT 1;")
        admin_row = cur.fetchone()
        if not admin_row:
            admin_user = (settings.ADMIN_USERNAME or "admin").strip().lower()
            admin_email = (settings.ADMIN_EMAIL or "admin@qpgen.local").strip().lower()
            admin_pass = settings.ADMIN_PASSWORD

            if not admin_pass:
                # Generate a secure random password if none was supplied in .env
                admin_pass = secrets.token_urlsafe(16)
                cred_file = Path(__file__).resolve().parent.parent / ".admin_initial_credentials.txt"
                try:
                    cred_file.write_text(
                        f"QPGen Initial Administrator Setup\n"
                        f"Generated at: {datetime.now(timezone.utc).isoformat()}\n"
                        f"Username: {admin_user}\n"
                        f"Email: {admin_email}\n"
                        f"Password: {admin_pass}\n\n"
                        f"KEEP THIS FILE SECURE. You can change your password in the Admin Panel.\n",
                        encoding="utf-8"
                    )
                    logger.info("[AUTH] Initial administrator created. Credentials written to %s", cred_file.name)
                except Exception as ex:
                    logger.warning("Could not write initial admin credentials file: %s", ex)
            else:
                logger.info("[AUTH] Initial administrator created from environment configuration.")

            now_str = datetime.now(timezone.utc).isoformat()
            user_id = str(uuid.uuid4())
            conn.execute(
                """
                INSERT INTO users (id, username, email, password_hash, full_name, role, is_active, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?);
                """,
                (
                    user_id,
                    admin_user,
                    admin_email,
                    hash_password(admin_pass),
                    "System Administrator",
                    UserRole.ADMIN.value,
                    now_str,
                    now_str,
                )
            )
            conn.commit()


# ---------------------------------------------------------------------------
# Rate Limiting Logic
# ---------------------------------------------------------------------------
def check_rate_limit(conn: sqlite3.Connection, ip_address: str, identifier: str) -> bool:
    """Return True if rate limit is exceeded (too many failed attempts within window)."""
    now = time.time()
    window_start = now - settings.RATE_LIMIT_WINDOW_SECONDS
    cur = conn.execute(
        """
        SELECT COUNT(*) as failed_count FROM login_attempts
        WHERE (ip_address = ? OR identifier = ?)
          AND success = 0
          AND timestamp >= ?;
        """,
        (ip_address, identifier.lower(), window_start)
    )
    row = cur.fetchone()
    count = row["failed_count"] if row else 0
    return count >= settings.RATE_LIMIT_FAILED_LOGINS


def record_login_attempt(conn: sqlite3.Connection, ip_address: str, identifier: str, success: bool) -> None:
    """Record a login attempt for rate limiting and auditing."""
    now = time.time()
    conn.execute(
        """
        INSERT INTO login_attempts (ip_address, identifier, timestamp, success)
        VALUES (?, ?, ?, ?);
        """,
        (ip_address, identifier.lower(), now, 1 if success else 0)
    )
    if success:
        # Clear recent failed attempts for this identifier
        conn.execute(
            "DELETE FROM login_attempts WHERE identifier = ? AND success = 0;",
            (identifier.lower(),)
        )
    conn.commit()


# ---------------------------------------------------------------------------
# Session Operations
# ---------------------------------------------------------------------------
def create_user_session(conn: sqlite3.Connection, user_id: str, ip_address: str = "", user_agent: str = "") -> str:
    """Create a new session token, record last login, and return the session ID."""
    session_id = secrets.token_urlsafe(32)
    now = time.time()
    expires_at = now + settings.SESSION_MAX_AGE_SECONDS

    conn.execute(
        """
        INSERT INTO sessions (session_id, user_id, created_at, expires_at, ip_address, user_agent)
        VALUES (?, ?, ?, ?, ?, ?);
        """,
        (session_id, user_id, now, expires_at, ip_address, user_agent)
    )
    conn.execute(
        """
        UPDATE users SET last_login_at = ? WHERE id = ?;
        """,
        (datetime.now(timezone.utc).isoformat(), user_id)
    )
    conn.commit()
    return session_id


def get_user_for_session(conn: sqlite3.Connection, session_id: str) -> Optional[User]:
    """Retrieve active user matching a valid, unexpired session ID."""
    if not session_id:
        return None
    now = time.time()
    cur = conn.execute(
        """
        SELECT u.id, u.username, u.email, u.full_name, u.role, u.is_active, u.created_at, u.last_login_at
        FROM sessions s
        JOIN users u ON s.user_id = u.id
        WHERE s.session_id = ?
          AND s.expires_at > ?
          AND u.is_active = 1;
        """,
        (session_id, now)
    )
    row = cur.fetchone()
    if not row:
        return None
    return User(
        id=row["id"],
        username=row["username"],
        email=row["email"],
        full_name=row["full_name"],
        role=UserRole(row["role"]),
        is_active=bool(row["is_active"]),
        created_at=row["created_at"],
        last_login_at=row["last_login_at"],
    )


def delete_session(conn: sqlite3.Connection, session_id: str) -> None:
    """Delete a session upon logout."""
    if session_id:
        conn.execute("DELETE FROM sessions WHERE session_id = ?;", (session_id,))
        conn.commit()


def invalidate_user_sessions(conn: sqlite3.Connection, user_id: str) -> None:
    """Invalidate all active sessions for a user (e.g. upon deactivation or password reset)."""
    conn.execute("DELETE FROM sessions WHERE user_id = ?;", (user_id,))
    conn.commit()


# ---------------------------------------------------------------------------
# User Administration (Admin Only — No Public Registration)
# ---------------------------------------------------------------------------
def list_all_users(conn: sqlite3.Connection) -> list[User]:
    """List all accounts for administrator management. Never includes password hashes."""
    cur = conn.execute(
        """
        SELECT id, username, email, full_name, role, is_active, created_at, last_login_at
        FROM users
        ORDER BY role ASC, created_at DESC;
        """
    )
    return [
        User(
            id=row["id"],
            username=row["username"],
            email=row["email"],
            full_name=row["full_name"],
            role=UserRole(row["role"]),
            is_active=bool(row["is_active"]),
            created_at=row["created_at"],
            last_login_at=row["last_login_at"],
        )
        for row in cur.fetchall()
    ]


def create_teacher_account(conn: sqlite3.Connection, req: UserCreateRequest) -> User:
    """Admin-only account creation. Rejects duplicate username or email."""
    # Check duplicate username
    cur = conn.execute("SELECT id FROM users WHERE username = ?;", (req.username,))
    if cur.fetchone():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Username '{req.username}' is already in use."
        )

    # Check duplicate email
    cur = conn.execute("SELECT id FROM users WHERE email = ?;", (req.email,))
    if cur.fetchone():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Email '{req.email}' is already registered."
        )

    now_str = datetime.now(timezone.utc).isoformat()
    user_id = str(uuid.uuid4())
    pw_hash = hash_password(req.password)
    target_role = UserRole.TEACHER.value

    conn.execute(
        """
        INSERT INTO users (id, username, email, password_hash, full_name, role, is_active, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?);
        """,
        (user_id, req.username, req.email, pw_hash, req.full_name, target_role, now_str, now_str)
    )
    conn.commit()
    logger.info("Admin created teacher account: %s (%s)", req.username, req.email)

    return User(
        id=user_id,
        username=req.username,
        email=req.email,
        full_name=req.full_name,
        role=UserRole.TEACHER,
        is_active=True,
        created_at=now_str,
        last_login_at=None,
    )


def update_user_status(conn: sqlite3.Connection, user_id: str, is_active: bool, acting_admin_id: str) -> User:
    """Activate or deactivate a user account. Deactivation immediately invalidates all active sessions."""
    cur = conn.execute("SELECT id, username, email, full_name, role, is_active, created_at, last_login_at FROM users WHERE id = ?;", (user_id,))
    row = cur.fetchone()
    if not row:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found.")

    if row["id"] == acting_admin_id and not is_active:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Administrators cannot deactivate their own account.")

    now_str = datetime.now(timezone.utc).isoformat()
    conn.execute("UPDATE users SET is_active = ?, updated_at = ? WHERE id = ?;", (1 if is_active else 0, now_str, user_id))
    if not is_active:
        invalidate_user_sessions(conn, user_id)
    conn.commit()

    return User(
        id=row["id"],
        username=row["username"],
        email=row["email"],
        full_name=row["full_name"],
        role=UserRole(row["role"]),
        is_active=is_active,
        created_at=row["created_at"],
        last_login_at=row["last_login_at"],
    )


def reset_user_password(conn: sqlite3.Connection, user_id: str, new_password: str) -> None:
    """Reset a user's password and terminate their active sessions to force re-login."""
    cur = conn.execute("SELECT id FROM users WHERE id = ?;", (user_id,))
    if not cur.fetchone():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found.")

    now_str = datetime.now(timezone.utc).isoformat()
    pw_hash = hash_password(new_password)
    conn.execute("UPDATE users SET password_hash = ?, updated_at = ? WHERE id = ?;", (pw_hash, now_str, user_id))
    invalidate_user_sessions(conn, user_id)
    conn.commit()


# ---------------------------------------------------------------------------
# FastAPI Authentication Dependencies
# ---------------------------------------------------------------------------
def extract_session_id(request: Request) -> Optional[str]:
    """Extract session token from cookie, Authorization header, or X-Session-ID header."""
    # 1. Cookie (primary for browser)
    cookie_val = request.cookies.get(COOKIE_NAME)
    if cookie_val:
        return cookie_val

    # 2. Authorization header: Bearer <token>
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header[7:].strip()
        if token:
            return token

    # 3. Custom X-Session-ID header (API clients)
    x_sess = request.headers.get("X-Session-ID", "").strip()
    if x_sess:
        return x_sess

    return None


def get_current_user(request: Request) -> User:
    """
    Dependency verifying active session.
    If unauthenticated and request accepts text/html, raises AuthenticationRedirectException.
    Otherwise raises HTTP 401 Unauthorized.
    """
    session_id = extract_session_id(request)
    if not session_id:
        if "text/html" in request.headers.get("accept", ""):
            raise AuthenticationRedirectException()
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Please log in.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    with get_db() as conn:
        user = get_user_for_session(conn, session_id)
        if not user or not user.is_active:
            if "text/html" in request.headers.get("accept", ""):
                raise AuthenticationRedirectException()
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired session. Please log in again.",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return user


def require_admin(current_user: User = Depends(get_current_user)) -> User:
    """Dependency verifying that the authenticated user possesses admin privileges."""
    if current_user.role != UserRole.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: Administrator privileges required.",
        )
    return current_user


def require_teacher(current_user: User = Depends(get_current_user)) -> User:
    """Dependency verifying that the authenticated user possesses teacher privileges."""
    if current_user.role != UserRole.TEACHER:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: Question paper generation, retrieval, and PDF downloads are restricted to teacher accounts.",
        )
    return current_user
