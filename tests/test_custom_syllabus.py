# tests/test_custom_syllabus.py
# Custom syllabus feature: parsing, persistence, upload (PDF/DOCX/text), selection,
# strict generation scope, and reuse. Embeddings / Gemini are mocked.

import io
import json
from unittest.mock import patch

import pytest
from docx import Document as DocxDocument
from fastapi.testclient import TestClient
from langchain_core.documents import Document

from app.auth import User, UserRole, get_current_user, require_teacher
from app.config import settings
from app.custom_syllabus import (
    SelectionError,
    build_chapter_chunks,
    build_scope_text,
    parse_syllabus,
    resolve_selection,
    to_chapters,
)
from app.main import app
from app.schemas import CustomSyllabus, CustomSyllabusChapter, Difficulty, Question, QuestionType

client = TestClient(app)

SYLLABUS_TEXT = """Term 1 Syllabus
Unit 1: Number System
- Natural numbers
- Whole numbers
1.1 Integers
Unit 2 - Algebra
Topics: variables, expressions; equations
Chapter 3
Geometry
• Lines and angles
• Triangles
"""


@pytest.fixture(autouse=True)
def isolated_data(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    with patch("app.custom_syllabus.build_faiss_index") as m:
        yield m


def _create(name="My Syllabus", text=SYLLABUS_TEXT, **extra):
    data = {"name": name, "text": text, "grade": "6", **extra}
    return client.post("/syllabi/custom", data=data)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

class TestParsing:
    def test_keyword_headings_topics_and_prefixes(self):
        chapters = parse_syllabus(SYLLABUS_TEXT)
        assert [c.title for c in chapters] == [
            "Unit 1: Number System",
            "Unit 2: Algebra",
            "Chapter 3 Geometry",
        ]
        assert chapters[0].topics == ["Natural numbers", "Whole numbers", "Integers"]
        assert chapters[1].topics == ["variables", "expressions", "equations"]
        assert chapters[2].topics == ["Lines and angles", "Triangles"]

    def test_numbered_headings(self):
        chapters = parse_syllabus("1. Fractions\n- halves\n- quarters\n2. Decimals\n- tenths\n")
        assert [c.title for c in chapters] == ["1. Fractions", "2. Decimals"]
        assert chapters[0].topics == ["halves", "quarters"]

    def test_flat_list_becomes_single_chapter(self):
        chapters = parse_syllabus("Addition\nSubtraction\nShapes")
        assert len(chapters) == 1
        assert chapters[0].topics == ["Addition", "Subtraction", "Shapes"]

    def test_duplicate_chapter_titles_are_made_unique(self):
        chapters = parse_syllabus("Unit 1: Numbers\n- a1\nUnit 2: Numbers\n- b1\n")
        titles = [c.title for c in chapters]
        assert len(set(t.lower() for t in titles)) == 2


# ---------------------------------------------------------------------------
# Chunking / retrieval metadata
# ---------------------------------------------------------------------------

def test_every_chunk_is_tagged_with_exact_chapter():
    chapters = to_chapters(parse_syllabus(SYLLABUS_TEXT))
    chunks = build_chapter_chunks(chapters, "custom_x", "My Syllabus", "6", "Mathematics")
    assert chunks
    titles = {c.title for c in chapters}
    assert {c["metadata"]["chapter"] for c in chunks} == titles
    assert len({c["metadata"]["chunk_id"] for c in chunks}) == len(chunks)


# ---------------------------------------------------------------------------
# Selection resolution
# ---------------------------------------------------------------------------

class TestSelection:
    def _syl(self):
        return CustomSyllabus(
            syllabus_id="custom_t",
            name="S",
            chapters=[
                CustomSyllabusChapter(title="Unit 1: A", topics=["a1", "a2"]),
                CustomSyllabusChapter(title="Unit 2: B", topics=["b1", "b2"]),
            ],
        )

    def test_requires_a_selection(self):
        with pytest.raises(SelectionError):
            resolve_selection(self._syl(), [], None)

    def test_unknown_chapter_and_topic_rejected(self):
        with pytest.raises(SelectionError):
            resolve_selection(self._syl(), ["Unit 9: Z"], None)
        with pytest.raises(SelectionError):
            resolve_selection(self._syl(), ["Unit 1: A"], {"Unit 1: A": ["nope"]})

    def test_topic_subset_and_full_topics(self):
        chapters, topics = resolve_selection(
            self._syl(), ["unit 2: b", "Unit 1: A"], {"Unit 1: A": ["a1"], "Unit 2: B": ["b1", "b2"]}
        )
        assert chapters == ["Unit 1: A", "Unit 2: B"]  # syllabus order
        assert topics == {"Unit 1: A": ["a1"], "Unit 2: B": []}  # all topics == unrestricted

    def test_scope_text_names_only_selected(self):
        text = build_scope_text("S", ["Unit 1: A"], {"Unit 1: A": ["a1"]})
        assert text.startswith("Custom syllabus")
        assert "Unit 1: A" in text and "a1" in text and "Unit 2" not in text


# ---------------------------------------------------------------------------
# CRUD API: text, DOCX and PDF input, save & reuse
# ---------------------------------------------------------------------------

class TestCustomSyllabusApi:
    def test_create_from_text_saves_and_indexes(self, isolated_data):
        r = _create()
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["syllabus_id"].startswith("custom_")
        assert [c["title"] for c in body["chapters"]][0] == "Unit 1: Number System"
        assert body["has_index"] is True
        assert isolated_data.call_count == 1
        saved = settings.custom_syllabi_dir / f"{body['syllabus_id']}.json"
        assert saved.exists()
        # source text of chapters is stored for re-indexing but not exposed to the UI
        assert "content" not in body["chapters"][0]

    def test_list_get_and_reuse(self):
        sid = _create(name="Reusable").json()["syllabus_id"]
        lst = client.get("/syllabi/custom").json()
        assert [s["syllabus_id"] for s in lst] == [sid]
        assert lst[0]["num_chapters"] == 3 and lst[0]["num_topics"] >= 7
        got = client.get(f"/syllabi/custom/{sid}")
        assert got.status_code == 200 and got.json()["name"] == "Reusable"

    def test_upload_docx(self, tmp_path):
        d = DocxDocument()
        for line in ["Unit 1: Fractions", "Halves", "Quarters", "Unit 2: Decimals", "Tenths"]:
            d.add_paragraph(line)
        buf = io.BytesIO()
        d.save(buf)
        buf.seek(0)
        r = client.post(
            "/syllabi/custom",
            data={"name": "Docx Syl"},
            files={"file": ("syl.docx", buf, "application/vnd.openxmlformats-officedocument.wordprocessingml.document")},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["source_type"] == "docx"
        assert [c["title"] for c in body["chapters"]] == ["Unit 1: Fractions", "Unit 2: Decimals"]
        assert body["chapters"][0]["topics"] == ["Halves", "Quarters"]

    def test_upload_pdf(self):
        import pymupdf
        doc = pymupdf.open()
        page = doc.new_page()
        page.insert_text((72, 72), "Unit 1: Shapes\n- Circles\n- Squares\nUnit 2: Measures\n- Length")
        data = doc.tobytes()
        doc.close()
        r = client.post(
            "/syllabi/custom",
            data={"name": "Pdf Syl"},
            files={"file": ("syl.pdf", io.BytesIO(data), "application/pdf")},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["source_type"] == "pdf"
        assert [c["title"] for c in body["chapters"]] == ["Unit 1: Shapes", "Unit 2: Measures"]

    def test_validation_errors(self):
        assert client.post("/syllabi/custom", data={"name": "x"}).status_code == 400
        bad = client.post(
            "/syllabi/custom",
            data={"name": "x"},
            files={"file": ("a.txt", io.BytesIO(b"hello"), "text/plain")},
        )
        assert bad.status_code == 400 and "Unsupported" in bad.json()["detail"]

    def test_embedding_failure_leaves_nothing_saved(self, isolated_data):
        isolated_data.side_effect = Exception("quota")
        r = _create()
        assert r.status_code == 502
        assert list(settings.custom_syllabi_dir.glob("*.json")) == []

    def test_update_structure_reindexes_and_keeps_content(self, isolated_data):
        sid = _create().json()["syllabus_id"]
        isolated_data.reset_mock()
        r = client.put(
            f"/syllabi/custom/{sid}",
            json={
                "name": "Renamed",
                "chapters": [
                    {"title": "Unit 1: Number System", "topics": ["Only topic"]},
                    {"title": "Brand New", "topics": ["x1"]},
                ],
            },
        )
        assert r.status_code == 200, r.text
        assert r.json()["name"] == "Renamed"
        assert [c["title"] for c in r.json()["chapters"]] == ["Unit 1: Number System", "Brand New"]
        assert isolated_data.call_count == 1

    def test_delete_removes_record_and_index(self):
        sid = _create().json()["syllabus_id"]
        idx = settings.indexes_dir / sid
        idx.mkdir(parents=True)
        (idx / "index.faiss").write_bytes(b"x")
        assert client.delete(f"/syllabi/custom/{sid}").status_code == 200
        assert not idx.exists()
        assert client.get(f"/syllabi/custom/{sid}").status_code == 404
        assert client.delete(f"/syllabi/custom/{sid}").status_code == 404

    def test_other_teacher_cannot_access(self):
        sid = _create().json()["syllabus_id"]
        other = User(
            id="someone-else", username="o", email="o@x.y", full_name="O",
            role=UserRole.TEACHER, is_active=True, created_at="2026-01-01T00:00:00Z",
        )
        app.dependency_overrides[get_current_user] = lambda: other
        app.dependency_overrides[require_teacher] = lambda: other
        assert client.get(f"/syllabi/custom/{sid}").status_code == 404
        assert client.get("/syllabi/custom").json() == []
        assert client.delete(f"/syllabi/custom/{sid}").status_code == 404


# ---------------------------------------------------------------------------
# Paper generation strictly from the selected custom syllabus
# ---------------------------------------------------------------------------

def _passages():
    def doc(ch):
        return Document(page_content=f"{ch} content", metadata={"chapter": ch, "source": f"S - {ch}"})
    # retrieval (substring matching) may leak neighbours; API must filter strictly
    return [doc("Unit 1: Number System"), doc("Unit 2: Algebra"), doc("Chapter 3 Geometry")]


def _qs(chapter, n=2):
    return [
        Question(
            question_id=f"q{abs(hash((chapter, i))) % 10**8}",
            question_text=f"Describe example {i} about {chapter}",
            question_type=QuestionType.SHORT_ANSWER,
            marks=2,
            difficulty=Difficulty.MEDIUM,
            topic=chapter,
            chapter=chapter,
            answer="x",
            solution="y",
        )
        for i in range(n)
    ]


class TestCustomGeneration:
    @patch("app.main.save_paper")
    @patch("app.main.generate_questions")
    @patch("app.main.retrieve_relevant_chunks")
    def test_generation_scoped_to_selection(self, mock_retrieve, mock_gen, mock_save):
        sid = _create().json()["syllabus_id"]
        mock_retrieve.return_value = _passages()
        mock_gen.return_value = _qs("Unit 1: Number System", 1) + _qs("Unit 2: Algebra", 1)

        payload = {
            "grade": "6",
            "num_questions": 2,
            "total_marks": 4,
            "question_types": ["short_answer"],
            "custom_syllabus_id": sid,
            "selected_chapters": ["Unit 1: Number System", "Unit 2: Algebra"],
            "selected_topics": {"Unit 2: Algebra": ["equations"]},
        }
        r = client.post("/papers/generate", json=payload)
        assert r.status_code == 200, r.text
        paper = r.json()

        kwargs = mock_gen.call_args.kwargs
        assert kwargs["permitted_units"] == ["Unit 1: Number System", "Unit 2: Algebra"]
        # unselected chapter's passages are never handed to the LLM
        chapters_seen = {d.metadata["chapter"] for d in kwargs["passages"]}
        assert chapters_seen == {"Unit 1: Number System", "Unit 2: Algebra"}
        assert "Geometry" not in kwargs["request"].syllabus_scope
        assert "equations" in kwargs["request"].syllabus_scope
        assert kwargs["request"].syllabus_scope.startswith("Custom syllabus")

        assert paper["custom_syllabus_id"] == sid
        assert paper["custom_syllabus_name"] == "My Syllabus"
        assert paper["permitted_units"] == ["Unit 1: Number System", "Unit 2: Algebra"]
        assert set(paper["covered_chapters"]) <= set(paper["permitted_units"])
        # retrieval was restricted to the saved custom index
        assert mock_retrieve.call_args.kwargs["document_id"] == sid

    @patch("app.main.generate_questions")
    @patch("app.main.retrieve_relevant_chunks")
    def test_selection_is_mandatory_and_validated(self, mock_retrieve, mock_gen):
        sid = _create().json()["syllabus_id"]
        base = {
            "grade": "6", "num_questions": 2, "total_marks": 4,
            "question_types": ["short_answer"], "custom_syllabus_id": sid,
        }
        r = client.post("/papers/generate", json=base)
        assert r.status_code == 422 and "at least one chapter" in r.json()["detail"]
        r = client.post("/papers/generate", json={**base, "selected_chapters": ["Unit 99"]})
        assert r.status_code == 422 and "not part of" in r.json()["detail"]
        mock_gen.assert_not_called()

    def test_unknown_syllabus_is_404(self):
        r = client.post("/papers/generate", json={
            "grade": "6", "num_questions": 2, "total_marks": 4,
            "question_types": ["short_answer"], "custom_syllabus_id": "custom_missing",
            "selected_chapters": ["Unit 1"],
        })
        assert r.status_code == 404

    @patch("app.main.generate_questions")
    @patch("app.main.retrieve_relevant_chunks")
    def test_no_passages_in_selected_chapters_is_rejected(self, mock_retrieve, mock_gen):
        sid = _create().json()["syllabus_id"]
        mock_retrieve.return_value = [Document(page_content="x", metadata={"chapter": "Chapter 3 Geometry"})]
        r = client.post("/papers/generate", json={
            "grade": "6", "num_questions": 2, "total_marks": 4,
            "question_types": ["short_answer"], "custom_syllabus_id": sid,
            "selected_chapters": ["Unit 1: Number System"],
        })
        assert r.status_code == 422
        mock_gen.assert_not_called()

    def test_dashboard_contains_custom_syllabus_and_sets_ui(self):
        html = client.get("/").text
        assert "Custom Syllabus" in html and "custom-syllabus-panel" in html
        assert "sets-select" in html and "/syllabi/custom" in html
