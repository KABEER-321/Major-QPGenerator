# tests/conftest.py
# Pytest configuration and shared fixtures for QPGen test suites.

import pytest
from app.auth import get_current_user, require_teacher, require_admin, User, UserRole
from app.main import app


@pytest.fixture(autouse=True)
def default_auth_override():
    """
    By default, provide an authenticated teacher user for existing question-generation
    and dashboard test suites so existing tests continue to pass seamlessly.
    Auth-specific tests can clear this override to test unauthenticated rejection.
    """
    mock_teacher = User(
        id="test-teacher-id",
        username="teacher",
        email="teacher@qpgen.local",
        full_name="Test Teacher",
        role=UserRole.TEACHER,
        is_active=True,
        created_at="2026-01-01T00:00:00Z",
    )
    app.dependency_overrides[get_current_user] = lambda: mock_teacher
    app.dependency_overrides[require_teacher] = lambda: mock_teacher
    yield
    app.dependency_overrides.pop(get_current_user, None)
    app.dependency_overrides.pop(require_teacher, None)
