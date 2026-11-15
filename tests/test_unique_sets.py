# tests/test_unique_sets.py
# Unique question paper sets (Set A, B, C...): disjoint questions, identical
# pattern/marks, per-set PDFs, ZIP bundle, and honest shortage reporting.

import io
import zipfile
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.duplicates import normalize_question_text
from app.main import app
from app.paper import assemble_paper
from app.schemas import (
    Difficulty,
    MathCheckStatus,
    MathVerificationResult,
    PaperGenerateRequest,
    Question,
    QuestionType,
)

client = TestClient(app)

TOPICS = ["rivers", "markets", "gardens", "libraries", "buses", "kites", "rooms", "farms", "ponds", "trains"]


def _pool(n_mcq: int, n_sa: int):
    out = []
    for i in range(n_mcq):
        out.append(Question(
            question_id=f"mcq_{i}",
            question_text=f"Which option best completes statement {i} about {TOPICS[i % 10]}?",
            question_type=QuestionType.MULTIPLE_CHOICE,
            marks=1,
            difficulty=Difficulty.MEDIUM,
            topic="Algebra",
            chapter="Algebra",
            answer="A",
            options=["A", "B", "C", "D"],
            correct_option="A",
            math_verification=MathVerificationResult(status=MathCheckStatus.VERIFIED),
        ))
    for i in range(n_sa):
        out.append(Question(
            question_id=f"sa_{i}",
            question_text=f"Explain scenario {i} involving {TOPICS[i % 10]} in your own words.",
            question_type=QuestionType.SHORT_ANSWER,
            marks=2,
            difficulty=Difficulty.MEDIUM,
            topic="Algebra",
            chapter="Algebra",
            answer="x",
            solution="y",
            math_verification=MathVerificationResult(status=MathCheckStatus.VERIFIED),
        ))
    return out


SECTIONS = [
    {"section_id": "A", "title": "Section A: MCQs", "question_type": "multiple_choice",
     "num_questions": 3, "section_marks": 3, "marks_per_question": 1},
    {"section_id": "B", "title": "Section B: Short Answer", "question_type": "short_answer",
     "num_questions": 2, "section_marks": 4, "marks_per_question": 2},
]


def _payload(**kw):
    p = {"document_id": "doc_sets", "grade": "5", "topic": "Algebra", "num_versions": 3, "sections": SECTIONS}
    p.update(kw)
    return p


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    with patch("app.main.retrieve_relevant_chunks", return_value=[]), \
         patch("app.main.generate_candidate_pool") as gen:
        yield gen


def test_three_sets_are_labelled_disjoint_and_identical_in_pattern(env):
    env.return_value = (_pool(9, 6), 0, 0)  # exactly enough for 3 sets
    r = client.post("/papers/generate-batch", json=_payload())
    assert r.status_code == 200, r.text
    data = r.json()

    assert data["set_labels"] == ["A", "B", "C"]
    assert data["warnings"] == []
    papers = data["papers"]
    assert [p["set_label"] for p in papers] == ["A", "B", "C"]
    assert [p["title"].endswith(f"(Set {l})") for p, l in zip(papers, "ABC")] == [True] * 3

    # no repeated questions across sets (by id and by normalized text)
    ids = [q["question_id"] for p in papers for q in p["questions"]]
    assert len(ids) == len(set(ids)) == 15
    texts = [normalize_question_text(q["question_text"]) for p in papers for q in p["questions"]]
    assert len(texts) == len(set(texts))

    # identical pattern: counts, marks, per-section structure, difficulty, chapters
    def shape(p):
        return (
            p["num_questions"], p["total_marks"], p["difficulty"],
            [(s["section_id"], s["question_type"], s["num_questions"], s["section_marks"],
              [q["marks"] for q in s["questions"]]) for s in p["sections"]],
            sorted(p["covered_chapters"] or []),
        )
    assert shape(papers[0]) == shape(papers[1]) == shape(papers[2])
    assert papers[0]["total_marks"] == 7 and papers[0]["num_questions"] == 5
    assert data["validation_summary"]["is_unique"] is True


def test_each_set_has_its_own_pdf_and_zip_bundle(env):
    env.return_value = (_pool(6, 4), 0, 0)
    data = client.post("/papers/generate-batch", json=_payload(num_versions=2)).json()
    pdfs = []
    for pid in data["paper_ids"]:
        res = client.get(f"/papers/{pid}/pdf?include_answers=false")
        assert res.status_code == 200 and res.content.startswith(b"%PDF")
        pdfs.append(res.content)
    assert pdfs[0] != pdfs[1]

    z = client.get(f"/papers/batch/{data['batch_id']}/pdfs?include_answers=true")
    assert z.status_code == 200 and z.headers["content-type"] == "application/zip"
    zf = zipfile.ZipFile(io.BytesIO(z.content))
    assert sorted(zf.namelist()) == ["Set_A_solutions.pdf", "Set_B_solutions.pdf"]
    assert all(zf.read(n).startswith(b"%PDF") for n in zf.namelist())

    assert client.get("/papers/batch/nope/pdfs").status_code == 404


def test_saved_set_is_listed_with_its_label(env):
    env.return_value = (_pool(6, 4), 0, 0)
    data = client.post("/papers/generate-batch", json=_payload(num_versions=2)).json()
    listed = {p["paper_id"]: p for p in client.get("/papers").json()}
    assert [listed[i]["set_label"] for i in data["paper_ids"]] == ["A", "B"]
    assert client.get(f"/papers/{data['paper_ids'][1]}").json()["set_label"] == "B"


def test_shortage_is_reported_not_silently_duplicated(env):
    env.return_value = (_pool(6, 4), 0, 0)  # enough for only 2 of 3 sets
    r = client.post("/papers/generate-batch", json=_payload(num_versions=3))
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert detail["error"] == "CandidatePoolShortageError"
    assert detail["needed"] > detail["available"]
    assert "3 sets" in detail["message"]


def test_allow_fewer_sets_returns_unique_sets_with_warning(env):
    env.return_value = (_pool(6, 4), 0, 0)
    r = client.post("/papers/generate-batch", json=_payload(num_versions=3, allow_fewer_sets=True))
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["num_versions_requested"] == 3 and data["num_versions_generated"] == 2
    assert data["set_labels"] == ["A", "B"]
    assert len(data["warnings"]) == 1 and "2 of the 3" in data["warnings"][0]
    ids = [q["question_id"] for p in data["papers"] for q in p["questions"]]
    assert len(ids) == len(set(ids))  # still no duplicates


def test_allow_fewer_sets_with_nothing_possible_is_still_422(env):
    env.return_value = (_pool(1, 1), 0, 0)
    r = client.post("/papers/generate-batch", json=_payload(num_versions=2, allow_fewer_sets=True))
    assert r.status_code == 422
    assert "even one paper" in r.json()["detail"]["message"]


def test_pdf_header_shows_set_label():
    req = PaperGenerateRequest(grade="5", topic="Algebra", num_questions=1, total_marks=2,
                               question_types=["short_answer"])
    q = _pool(0, 1)
    paper = assemble_paper(req, q, set_label="B", num_versions=2, version_number=2)
    assert paper.set_label == "B" and paper.title.endswith("(Set B)")
    from app.pdf_export import generate_pdf
    assert generate_pdf(paper, include_answers=False).startswith(b"%PDF")


def test_single_paper_title_unchanged_without_set_label():
    req = PaperGenerateRequest(grade="5", topic="Algebra", num_questions=1, total_marks=2,
                               question_types=["short_answer"])
    paper = assemble_paper(req, _pool(0, 1), version_number=2, num_versions=2)
    assert "Version 2" in paper.title and paper.set_label is None


def test_custom_syllabus_with_multiple_sets(env):
    """Both features together: sets generated strictly from the selected custom chapters."""
    from langchain_core.documents import Document
    with patch("app.custom_syllabus.build_faiss_index"):
        sid = client.post("/syllabi/custom", data={
            "name": "Combo", "grade": "6",
            "text": "Unit 1: Algebra\n- Variables\nUnit 2: Geometry\n- Triangles\n",
        }).json()["syllabus_id"]
    env.return_value = (_pool(6, 4), 0, 0)
    with patch("app.main.retrieve_relevant_chunks", return_value=[
        Document(page_content="a", metadata={"chapter": "Unit 1: Algebra"}),
        Document(page_content="g", metadata={"chapter": "Unit 2: Geometry"}),
    ]):
        r = client.post("/papers/generate-batch", json=_payload(
            grade="6", num_versions=2, document_id="",
            custom_syllabus_id=sid, selected_chapters=["Unit 1: Algebra"],
        ))
    assert r.status_code == 200, r.text
    data = r.json()
    kwargs = env.call_args.kwargs
    assert kwargs["permitted_units"] == ["Unit 1: Algebra"]
    assert {d.metadata["chapter"] for d in kwargs["passages"]} == {"Unit 1: Algebra"}
    assert kwargs["syllabus_scope"].startswith("Custom syllabus")
    assert [p["set_label"] for p in data["papers"]] == ["A", "B"]
    assert all(p["custom_syllabus_id"] == sid for p in data["papers"])
    assert all(p["permitted_units"] == ["Unit 1: Algebra"] for p in data["papers"])
