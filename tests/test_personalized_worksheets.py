# tests/test_personalized_worksheets.py
# Feature: AI-Powered Performance-Based Personalized Practice Worksheets in QPGen
# Verifies:
# 1. Performance classification thresholds (<40% Weak, 40-59% Needs Improvement, 60-79% Average, 80-100% Strong).
# 2. Simulated performance profile generation with distinct archetypes and DEMO labeling.
# 3. Largest Remainder question allocation maintaining exact question count without rounding drift.
# 4. Difficulty progression (Easy for Weak, Medium for Needs Improvement/Average, Hard for Strong).
# 5. Personalized worksheet generation targeting individual student weaknesses.
# 6. Student Practice Worksheet PDF vs Teacher Answer Key PDF and ZIP export.
# 7. Clean EvaluationModuleAdapter for future evaluation module integration.
# 8. FastAPI teacher endpoints for performance analysis and worksheet generation.

import io
import zipfile
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.performance import (
    EvaluationModuleAdapter,
    EvaluationResultRecord,
    PerformanceCategory,
    StudentPerformanceProfile,
    StudentTopicScore,
    classify_performance_percentage,
    generate_simulated_profiles,
)
from app.schemas import Difficulty, QuestionType
from app.worksheet_pdf import generate_worksheet_pdf, generate_worksheets_zip
from app.worksheets import (
    PersonalizedWorksheet,
    WorksheetGenerateRequest,
    allocate_worksheet_questions,
    generate_personalized_worksheet,
)

client = TestClient(app)


# ---------------------------------------------------------------------------
# 1. Classification & Scoring Tests
# ---------------------------------------------------------------------------

def test_classify_performance_percentage():
    """Verify performance classification matches specification tiers."""
    assert classify_performance_percentage(0.0) == PerformanceCategory.WEAK
    assert classify_performance_percentage(39.9) == PerformanceCategory.WEAK
    assert classify_performance_percentage(40.0) == PerformanceCategory.NEEDS_IMPROVEMENT
    assert classify_performance_percentage(59.9) == PerformanceCategory.NEEDS_IMPROVEMENT
    assert classify_performance_percentage(60.0) == PerformanceCategory.AVERAGE
    assert classify_performance_percentage(79.9) == PerformanceCategory.AVERAGE
    assert classify_performance_percentage(80.0) == PerformanceCategory.STRONG
    assert classify_performance_percentage(100.0) == PerformanceCategory.STRONG


def test_student_topic_score_creation():
    """Verify StudentTopicScore validates and clamps marks properly."""
    score = StudentTopicScore.create("Fractions", "Fractions", marks_obtained=7.0, max_marks=20.0)
    assert score.percentage == 35.0
    assert score.category == PerformanceCategory.WEAK

    # Clamping out-of-range marks
    clamped_high = StudentTopicScore.create("Geometry", "Geometry", marks_obtained=25.0, max_marks=20.0)
    assert clamped_high.marks_obtained == 20.0
    assert clamped_high.percentage == 100.0
    assert clamped_high.category == PerformanceCategory.STRONG

    clamped_low = StudentTopicScore.create("Algebra", "Algebra", marks_obtained=-5.0, max_marks=20.0)
    assert clamped_low.marks_obtained == 0.0
    assert clamped_low.percentage == 0.0
    assert clamped_low.category == PerformanceCategory.WEAK


def test_student_performance_profile_summaries():
    """Verify profile recomputation extracts strengths and areas for improvement."""
    scores = [
        StudentTopicScore.create("Fractions", "Fractions", 5.0, 20.0),      # 25% -> Weak
        StudentTopicScore.create("Geometry", "Geometry", 10.0, 20.0),      # 50% -> Needs Improvement
        StudentTopicScore.create("Decimals", "Decimals", 14.0, 20.0),      # 70% -> Average
        StudentTopicScore.create("Addition", "Addition", 19.0, 20.0),      # 95% -> Strong
    ]
    profile = StudentPerformanceProfile(
        student_name="Aarav Sharma",
        roll_number="101",
        grade="5",
        topic_scores=scores,
    )
    profile.recompute_summaries()

    assert "Addition" in profile.strengths
    assert "Fractions" in profile.areas_for_improvement
    assert "Geometry" in profile.areas_for_improvement
    assert "Addition" not in profile.areas_for_improvement
    assert profile.overall_percentage == pytest.approx(60.0, abs=0.5)


# ---------------------------------------------------------------------------
# 2. Simulation Mode Tests
# ---------------------------------------------------------------------------

def test_simulated_profiles_generation():
    """Verify simulated profile generator produces distinct student archetypes with DEMO label."""
    students = [
        {"student_name": "Student A", "roll_number": "1"},
        {"student_name": "Student B", "roll_number": "2"},
        {"student_name": "Student C", "roll_number": "3"},
        {"student_name": "Student D", "roll_number": "4"},
    ]
    chapters = ["Fractions", "Geometry", "Large Numbers", "Operations"]

    profiles = generate_simulated_profiles(
        grade="5",
        students=students,
        chapters=chapters,
    )

    assert len(profiles) == 4
    for prof in profiles:
        assert prof.is_simulated is True
        assert prof.simulation_label is not None
        assert len(prof.topic_scores) == 4
        # Every profile must have calculated strengths and improvement areas
        assert isinstance(prof.strengths, list)
        assert isinstance(prof.areas_for_improvement, list)

    # Student A (weak fractions archetype) vs Student B (weak geometry archetype)
    prof_a = profiles[0]
    prof_b = profiles[1]
    score_a_fractions = next(s for s in prof_a.topic_scores if s.topic == "Fractions")
    score_b_geometry = next(s for s in prof_b.topic_scores if s.topic == "Geometry")

    assert score_a_fractions.category in (PerformanceCategory.WEAK, PerformanceCategory.NEEDS_IMPROVEMENT)
    assert score_b_geometry.category in (PerformanceCategory.WEAK, PerformanceCategory.NEEDS_IMPROVEMENT)


# ---------------------------------------------------------------------------
# 3. Largest Remainder Question Allocation Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("requested_count", [5, 10, 15, 20, 25])
def test_largest_remainder_exact_count_invariant(requested_count):
    """Verify that Largest Remainder allocation guarantees exact total questions for all sizes."""
    scores = [
        StudentTopicScore.create("Fractions", "Fractions", 4.0, 20.0),      # Weak (50% target)
        StudentTopicScore.create("Geometry", "Geometry", 9.0, 20.0),       # Needs Improvement (30% target)
        StudentTopicScore.create("Measurement", "Measurement", 13.0, 20.0), # Average (15% target)
        StudentTopicScore.create("Addition", "Addition", 19.0, 20.0),       # Strong (5% target)
    ]
    profile = StudentPerformanceProfile(
        student_name="Test Student",
        grade="5",
        topic_scores=scores,
    )
    profile.recompute_summaries()

    slots = allocate_worksheet_questions(profile, total_questions=requested_count)
    assert len(slots) == requested_count

    # Verify Weak topics receive the largest allocation
    cat_counts = {}
    for s in slots:
        cat_counts[s.category] = cat_counts.get(s.category, 0) + 1

    assert cat_counts.get(PerformanceCategory.WEAK, 0) >= cat_counts.get(PerformanceCategory.STRONG, 0)


def test_allocation_with_no_weak_topics():
    """Verify allocation redistributes smoothly when student has no weak topics."""
    scores = [
        StudentTopicScore.create("Topic A", "Topic A", 15.0, 20.0),  # Average
        StudentTopicScore.create("Topic B", "Topic B", 18.0, 20.0),  # Strong
        StudentTopicScore.create("Topic C", "Topic C", 19.0, 20.0),  # Strong
    ]
    profile = StudentPerformanceProfile(
        student_name="Top Performer",
        grade="5",
        topic_scores=scores,
    )
    profile.recompute_summaries()

    slots = allocate_worksheet_questions(profile, total_questions=10)
    assert len(slots) == 10
    # Questions allocated between Average and Strong
    assert all(s.category in (PerformanceCategory.AVERAGE, PerformanceCategory.STRONG) for s in slots)


# ---------------------------------------------------------------------------
# 4. Difficulty Progression Tests
# ---------------------------------------------------------------------------

def test_difficulty_progression():
    """Verify that allocated slots match the specified difficulty adaptation rules."""
    scores = [
        StudentTopicScore.create("Fractions", "Fractions", 4.0, 20.0),  # Weak
        StudentTopicScore.create("Multiplication", "Multiplication", 19.0, 20.0),  # Strong
    ]
    profile = StudentPerformanceProfile(
        student_name="Student",
        grade="5",
        topic_scores=scores,
    )
    profile.recompute_summaries()

    slots = allocate_worksheet_questions(profile, total_questions=10)
    for s in slots:
        if s.category == PerformanceCategory.WEAK:
            assert s.difficulty == Difficulty.EASY
        elif s.category == PerformanceCategory.STRONG:
            assert s.difficulty == Difficulty.HARD


# ---------------------------------------------------------------------------
# 5. Worksheet Generation Pipeline Tests
# ---------------------------------------------------------------------------

def test_generate_personalized_worksheet_differentiates_students():
    """Verify that two students with different weaknesses get different personalized worksheets."""
    # Student A: Weak in Fractions, Strong in Addition
    prof_a = StudentPerformanceProfile(
        student_name="Student A",
        roll_number="A01",
        grade="5",
        topic_scores=[
            StudentTopicScore.create("Fractions", "Fractions", 5.0, 20.0),
            StudentTopicScore.create("Addition", "Addition", 19.0, 20.0),
        ],
    )
    prof_a.recompute_summaries()

    # Student B: Weak in Geometry, Strong in Multiplication
    prof_b = StudentPerformanceProfile(
        student_name="Student B",
        roll_number="B02",
        grade="5",
        topic_scores=[
            StudentTopicScore.create("Geometry", "Geometry", 4.0, 20.0),
            StudentTopicScore.create("Multiplication", "Multiplication", 19.0, 20.0),
        ],
    )
    prof_b.recompute_summaries()

    req_a = WorksheetGenerateRequest(grade="5", student_profile=prof_a, num_questions=10)
    ws_a = generate_personalized_worksheet(req_a)

    req_b = WorksheetGenerateRequest(grade="5", student_profile=prof_b, num_questions=10)
    ws_b = generate_personalized_worksheet(req_b)

    assert len(ws_a.questions) == 10
    assert len(ws_b.questions) == 10

    # Student A should have more Fractions questions
    assert ws_a.topic_allocation.get("Fractions", 0) > ws_a.topic_allocation.get("Addition", 0)
    # Student B should have more Geometry questions
    assert ws_b.topic_allocation.get("Geometry", 0) > ws_b.topic_allocation.get("Multiplication", 0)


# ---------------------------------------------------------------------------
# 6. PDF & ZIP Generation Tests
# ---------------------------------------------------------------------------

def test_worksheet_pdf_and_zip_generation():
    """Verify Student Practice Worksheet PDF, Teacher Solution Key PDF, and ZIP generation."""
    prof = StudentPerformanceProfile(
        student_name="Meera Sen",
        roll_number="25",
        grade="4",
        topic_scores=[
            StudentTopicScore.create("Fractions", "Fractions", 6.0, 20.0),
            StudentTopicScore.create("Division", "Division", 16.0, 20.0),
        ],
    )
    prof.recompute_summaries()

    req = WorksheetGenerateRequest(grade="4", student_profile=prof, num_questions=6)
    ws = generate_personalized_worksheet(req)

    # 1. Student PDF (ruled lines, no answers)
    student_pdf = generate_worksheet_pdf(ws, include_answers=False)
    assert isinstance(student_pdf, bytes)
    assert student_pdf.startswith(b"%PDF")

    # 2. Teacher PDF (solutions, answers)
    teacher_pdf = generate_worksheet_pdf(ws, include_answers=True)
    assert isinstance(teacher_pdf, bytes)
    assert teacher_pdf.startswith(b"%PDF")

    # 3. Batch ZIP
    zip_bytes = generate_worksheets_zip([ws], include_answers=False)
    assert isinstance(zip_bytes, bytes)
    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
        namelist = zf.namelist()
        assert "Worksheet_Master_Index.txt" in namelist
        assert any(n.endswith(".pdf") for n in namelist)


# ---------------------------------------------------------------------------
# 7. Future Evaluation Module Adapter Integration Tests
# ---------------------------------------------------------------------------

def test_evaluation_module_adapter():
    """Verify that the external evaluation adapter cleanly converts raw evaluation records."""
    raw_records = [
        EvaluationResultRecord(
            student_id="st_101",
            student_name="Rahul Varma",
            roll_number="01",
            examination_id="exam_term1_2026",
            examination_title="Term 1 Final Examination",
            grade="5",
            topic_or_chapter_id="Fractions",
            marks_obtained=6.0,
            max_marks=20.0,
        ),
        EvaluationResultRecord(
            student_id="st_101",
            student_name="Rahul Varma",
            roll_number="01",
            examination_id="exam_term1_2026",
            examination_title="Term 1 Final Examination",
            grade="5",
            topic_or_chapter_id="Multiplication",
            marks_obtained=18.0,
            max_marks=20.0,
        ),
    ]

    profiles = EvaluationModuleAdapter.ingest_evaluation_results(raw_records)
    assert len(profiles) == 1
    prof = profiles[0]
    assert prof.student_name == "Rahul Varma"
    assert prof.is_simulated is False  # Real evaluation data, not simulated!
    assert "Fractions" in prof.areas_for_improvement
    assert "Multiplication" in prof.strengths

    # Directly feed into worksheet generator
    req = WorksheetGenerateRequest(grade=prof.grade, student_profile=prof, num_questions=8)
    ws = generate_personalized_worksheet(req)
    assert len(ws.questions) == 8
    assert ws.topic_allocation.get("Fractions", 0) > ws.topic_allocation.get("Multiplication", 0)


# ---------------------------------------------------------------------------
# 8. API Endpoint Tests
# ---------------------------------------------------------------------------

def test_api_simulated_performance():
    """Verify GET /worksheets/simulated-performance returns simulated profiles."""
    resp = client.get("/worksheets/simulated-performance?grade=5&count=3")
    assert resp.status_code == 200
    profiles = resp.json()
    assert isinstance(profiles, list)
    assert len(profiles) == 3
    assert all(p["is_simulated"] is True for p in profiles)
    assert all(p.get("simulation_label") is not None for p in profiles)


def test_api_generate_and_get_worksheet():
    """Verify POST /worksheets/generate and GET /worksheets/{id} endpoints."""
    prof = StudentPerformanceProfile(
        student_name="Ananya Rao",
        roll_number="12",
        grade="5",
        topic_scores=[
            StudentTopicScore.create("Fractions", "Fractions", 5.0, 20.0),
            StudentTopicScore.create("Geometry", "Geometry", 18.0, 20.0),
        ],
    )
    prof.recompute_summaries()

    payload = {
        "grade": "5",
        "subject": "Mathematics",
        "student_profile": prof.model_dump(),
        "num_questions": 6,
        "include_figures": True,
    }
    resp = client.post("/worksheets/generate", json=payload)
    assert resp.status_code == 200
    ws_data = resp.json()
    ws_id = ws_data["worksheet_id"]
    assert len(ws_data["questions"]) == 6

    # Fetch by ID
    get_resp = client.get(f"/worksheets/{ws_id}")
    assert get_resp.status_code == 200
    assert get_resp.json()["worksheet_id"] == ws_id

    # Download Student PDF
    pdf_resp = client.get(f"/worksheets/{ws_id}/pdf?mode=student")
    assert pdf_resp.status_code == 200
    assert pdf_resp.headers["content-type"] == "application/pdf"
    assert pdf_resp.content.startswith(b"%PDF")
