# app/performance.py
# Student Performance Analysis, Simulation Engine, and Evaluation Integration Adapter.
# Supports:
# 1. Topic-wise performance calculation and classification (Weak, Needs Improvement, Average, Strong).
# 2. Simulated performance profile generator (DEMO / SIMULATED DATA) with distinct student archetypes.
# 3. Manual performance entry processing.
# 4. EvaluationModuleAdapter: clean, isolated interface for future external evaluation engine integration.
# 5. Local persistence of student performance records.

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from app.config import settings
from app.curriculum import get_chapters_for_class

logger = logging.getLogger(__name__)


class PerformanceCategory(str, Enum):
    """Four-tier performance classification according to academic mastery."""
    WEAK = "weak"                          # Below 40%
    NEEDS_IMPROVEMENT = "needs_improvement"  # 40% – 59%
    AVERAGE = "average"                    # 60% – 79%
    STRONG = "strong"                      # 80% – 100%


CATEGORY_DISPLAY = {
    PerformanceCategory.WEAK: {"label": "Weak (<40%)", "color": "#dc2626", "badge_class": "badge-danger"},
    PerformanceCategory.NEEDS_IMPROVEMENT: {"label": "Needs Improvement (40-59%)", "color": "#d97706", "badge_class": "badge-warning"},
    PerformanceCategory.AVERAGE: {"label": "Average (60-79%)", "color": "#2563eb", "badge_class": "badge-info"},
    PerformanceCategory.STRONG: {"label": "Strong (80-100%)", "color": "#16a34a", "badge_class": "badge-success"},
}


def classify_performance_percentage(percentage: float) -> PerformanceCategory:
    """Classify percentage score into one of the 4 defined mastery tiers."""
    pct = round(percentage, 2)
    if pct < 40.0:
        return PerformanceCategory.WEAK
    elif pct < 60.0:
        return PerformanceCategory.NEEDS_IMPROVEMENT
    elif pct < 80.0:
        return PerformanceCategory.AVERAGE
    else:
        return PerformanceCategory.STRONG


class StudentTopicScore(BaseModel):
    """Performance record for an individual mathematics topic/chapter."""
    topic: str = Field(..., description="Topic or chapter name")
    chapter: str = Field(..., description="Textbook chapter title")
    marks_obtained: float = Field(..., ge=0, description="Marks scored by student")
    max_marks: float = Field(..., gt=0, description="Maximum possible marks for this topic")
    percentage: float = Field(..., ge=0, le=100, description="Percentage score (0-100%)")
    category: PerformanceCategory = Field(..., description="Weak / Needs Improvement / Average / Strong")
    notes: Optional[str] = Field(default=None, description="Diagnostic notes or teacher observation")

    @classmethod
    def create(cls, topic: str, chapter: str, marks_obtained: float, max_marks: float, notes: Optional[str] = None) -> "StudentTopicScore":
        marks_obtained = max(0.0, min(float(marks_obtained), float(max_marks)))
        max_marks = max(1.0, float(max_marks))
        pct = round((marks_obtained / max_marks) * 100.0, 1)
        cat = classify_performance_percentage(pct)
        return cls(
            topic=topic,
            chapter=chapter,
            marks_obtained=round(marks_obtained, 1),
            max_marks=round(max_marks, 1),
            percentage=pct,
            category=cat,
            notes=notes,
        )


class StudentPerformanceProfile(BaseModel):
    """Comprehensive performance profile of a student across an assessment or syllabus."""
    profile_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    student_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:8])
    student_name: str = Field(..., min_length=1, description="Student full name")
    roll_number: Optional[str] = Field(default=None, description="Student roll number or registration ID")
    grade: str = Field(..., description="Class or grade (e.g. '5', 'Class 4')")
    assessment_id: Optional[str] = Field(default=None, description="Reference exam or assessment ID")
    assessment_title: Optional[str] = Field(default="Diagnostic Assessment", description="Assessment title")
    is_simulated: bool = Field(default=False, description="Whether this data is DEMO / SIMULATED")
    simulation_label: Optional[str] = Field(default=None, description="Demo profile archetype (e.g. 'weak_fractions')")
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    topic_scores: list[StudentTopicScore] = Field(default_factory=list)
    strengths: list[str] = Field(default_factory=list)
    areas_for_improvement: list[str] = Field(default_factory=list)
    overall_percentage: float = Field(default=0.0)

    def recompute_summaries(self) -> None:
        """Update strengths, areas for improvement, and overall percentage."""
        if not self.topic_scores:
            self.overall_percentage = 0.0
            self.strengths = []
            self.areas_for_improvement = []
            return

        total_obtained = sum(s.marks_obtained for s in self.topic_scores)
        total_max = sum(s.max_marks for s in self.topic_scores)
        self.overall_percentage = round((total_obtained / total_max) * 100.0, 1) if total_max > 0 else 0.0

        self.strengths = [s.topic for s in self.topic_scores if s.category == PerformanceCategory.STRONG]
        self.areas_for_improvement = [
            s.topic for s in self.topic_scores
            if s.category in (PerformanceCategory.WEAK, PerformanceCategory.NEEDS_IMPROVEMENT)
        ]


# ---------------------------------------------------------------------------
# Simulation Engine: Generates realistic mock profiles for demonstration
# ---------------------------------------------------------------------------

SIMULATION_ARCHETYPES = [
    {
        "id": "weak_fractions_shapes",
        "description": "Student A: Struggles with Fractions & Shapes, Strong in Arithmetic",
        "bias": {
            "fractions": 0.25,      # Weak
            "shapes": 0.35,         # Weak
            "geometry": 0.30,       # Weak
            "addition": 0.90,       # Strong
            "multiplication": 0.85, # Strong
            "subtraction": 0.88,    # Strong
            "digits": 0.92,         # Strong
            "default": 0.65,        # Average
        },
    },
    {
        "id": "weak_geometry_measurement",
        "description": "Student B: Struggles with Geometry & Measurement, Strong in Multiplications",
        "bias": {
            "perimeter": 0.28,      # Weak
            "area": 0.32,           # Weak
            "measurement": 0.36,    # Weak
            "geometry": 0.30,       # Weak
            "solids": 0.38,         # Weak
            "multiplication": 0.90, # Strong
            "division": 0.82,       # Strong
            "addition": 0.75,       # Average
            "default": 0.62,        # Average
        },
    },
    {
        "id": "balanced_average",
        "description": "Student C: Balanced moderate performance across all chapters",
        "bias": {
            "fractions": 0.52,      # Needs improvement
            "division": 0.55,       # Needs improvement
            "addition": 0.72,       # Average
            "subtraction": 0.70,    # Average
            "multiplication": 0.68, # Average
            "perimeter": 0.64,      # Average
            "default": 0.65,        # Average
        },
    },
    {
        "id": "high_performer",
        "description": "Student D: Strong across most topics, needs slight improvement in Word Problems",
        "bias": {
            "problem": 0.54,        # Needs improvement
            "word": 0.52,           # Needs improvement
            "default": 0.86,        # Strong
        },
    },
]


def _match_topic_bias(topic_name: str, bias_dict: dict[str, float]) -> float:
    """Find the target percentage bias matching a topic name."""
    norm = topic_name.lower()
    for key, val in bias_dict.items():
        if key != "default" and key in norm:
            return val
    return bias_dict.get("default", 0.65)


def generate_simulated_profiles(
    grade: str,
    students: list[dict],
    assessment_id: Optional[str] = None,
    assessment_title: Optional[str] = None,
    chapters: Optional[list[str]] = None,
) -> list[StudentPerformanceProfile]:
    """
    Generate realistic simulated performance records for demonstration.
    Labels all outputs explicitly with is_simulated=True and 'DEMO / SIMULATED DATA'.
    """
    # 1. Resolve chapters for this grade
    if not chapters:
        try:
            chapters = get_chapters_for_class(grade)
        except Exception:
            chapters = [
                f"Chapter {i+1}: Topic {chr(65+i)}" for i in range(5)
            ]

    if not chapters:
        chapters = ["Unit 1: Numbers & Operations", "Unit 2: Fractions", "Unit 3: Geometry & Perimeter", "Unit 4: Measurement"]

    profiles: list[StudentPerformanceProfile] = []

    for idx, st in enumerate(students):
        archetype = SIMULATION_ARCHETYPES[idx % len(SIMULATION_ARCHETYPES)]
        st_name = st.get("student_name") or f"Student {idx + 1}"
        st_roll = st.get("roll_number") or str(101 + idx)

        scores: list[StudentTopicScore] = []
        for ch_idx, ch in enumerate(chapters):
            bias_pct = _match_topic_bias(ch, archetype["bias"])
            # Add slight realistic deterministic jitter per student and chapter
            jitter = (((idx + 1) * 7 + (ch_idx + 1) * 13) % 11 - 5) / 100.0
            actual_pct = max(0.15, min(0.98, bias_pct + jitter))

            max_m = 20.0
            obtained_m = round(max_m * actual_pct, 1)

            scores.append(StudentTopicScore.create(
                topic=ch,
                chapter=ch,
                marks_obtained=obtained_m,
                max_marks=max_m,
                notes="Simulated performance metric for demonstration."
            ))

        prof = StudentPerformanceProfile(
            profile_id=f"sim_{uuid.uuid4().hex[:10]}",
            student_id=f"st_{uuid.uuid4().hex[:8]}",
            student_name=st_name,
            roll_number=st_roll,
            grade=str(grade),
            assessment_id=assessment_id or "demo_midterm_2026",
            assessment_title=assessment_title or f"Class {grade} Assessment (Demo Evaluation)",
            is_simulated=True,
            simulation_label=archetype["description"],
            topic_scores=scores,
        )
        prof.recompute_summaries()
        profiles.append(prof)

    return profiles


# ---------------------------------------------------------------------------
# Future Evaluation Integration Adapter
# ---------------------------------------------------------------------------

class EvaluationResultRecord(BaseModel):
    """Canonical data record from external evaluation engine."""
    student_id: str
    student_name: str
    roll_number: Optional[str] = None
    examination_id: str
    examination_title: Optional[str] = None
    grade: str
    topic_or_chapter_id: str
    marks_obtained: float
    max_marks: float
    comments: Optional[str] = None


class EvaluationModuleAdapter:
    """
    Isolated interface adapter for the future external evaluation module.
    When the evaluation team finishes their module, they feed raw evaluation records
    into this adapter to construct verified StudentPerformanceProfiles without touching
    the worksheet generation engine.
    """

    @classmethod
    def ingest_evaluation_results(
        cls,
        records: list[EvaluationResultRecord],
    ) -> list[StudentPerformanceProfile]:
        """Convert a list of raw evaluation records into structured performance profiles."""
        by_student: dict[str, list[EvaluationResultRecord]] = {}
        for r in records:
            by_student.setdefault(r.student_id, []).append(r)

        profiles: list[StudentPerformanceProfile] = []
        for s_id, s_records in by_student.items():
            first = s_records[0]

            # Aggregate marks per topic/chapter
            topic_totals: dict[str, dict[str, float]] = {}
            for r in s_records:
                tt = topic_totals.setdefault(r.topic_or_chapter_id, {"obtained": 0.0, "max": 0.0})
                tt["obtained"] += r.marks_obtained
                tt["max"] += r.max_marks

            topic_scores = [
                StudentTopicScore.create(
                    topic=t_name,
                    chapter=t_name,
                    marks_obtained=vals["obtained"],
                    max_marks=vals["max"],
                    notes="Ingested from actual examination evaluation module."
                )
                for t_name, vals in topic_totals.items()
            ]

            profile = StudentPerformanceProfile(
                profile_id=f"eval_{uuid.uuid4().hex[:10]}",
                student_id=s_id,
                student_name=first.student_name,
                roll_number=first.roll_number,
                grade=first.grade,
                assessment_id=first.examination_id,
                assessment_title=first.examination_title or "Official Assessment",
                is_simulated=False,
                topic_scores=topic_scores,
            )
            profile.recompute_summaries()
            profiles.append(profile)

        return profiles


# ---------------------------------------------------------------------------
# Storage & Retrieval
# ---------------------------------------------------------------------------

def save_performance_profile(profile: StudentPerformanceProfile) -> Path:
    """Save performance profile JSON locally."""
    d = settings.performance_dir
    f_path = d / f"{profile.profile_id}.json"
    with open(f_path, "w", encoding="utf-8") as f:
        f.write(profile.model_dump_json(indent=2))
    return f_path


def load_performance_profile(profile_id: str) -> Optional[StudentPerformanceProfile]:
    """Load a saved performance profile by ID."""
    f_path = settings.performance_dir / f"{profile_id}.json"
    if not f_path.exists():
        return None
    with open(f_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return StudentPerformanceProfile.model_validate(data)


def list_performance_profiles(grade: Optional[str] = None) -> list[StudentPerformanceProfile]:
    """List saved performance profiles, optionally filtered by grade."""
    profiles: list[StudentPerformanceProfile] = []
    d = settings.performance_dir
    for f in d.glob("*.json"):
        try:
            with open(f, "r", encoding="utf-8") as fp:
                p = StudentPerformanceProfile.model_validate(json.load(fp))
                if grade is None or p.grade == str(grade):
                    profiles.append(p)
        except Exception as e:
            logger.debug("Skipping unreadable performance profile %s: %s", f.name, e)
    profiles.sort(key=lambda p: p.created_at, reverse=True)
    return profiles
