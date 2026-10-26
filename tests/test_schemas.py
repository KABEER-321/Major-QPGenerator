# tests/test_schemas.py
# Tests for Pydantic schema validation — no API key needed.

import pytest
from app.schemas import (
    Question, QuestionType, Difficulty, Paper, PaperSection,
    ValidationStatus, SourceReference,
    DocumentRecord, DocumentSummary, TextDocumentRequest,
    SectionBlueprint, ExamBlueprint, create_default_blueprint,
    PaperGenerateRequest, BatchPaperGenerateRequest,
    BatchValidationSummary, BatchPaperResponse,
)


# ---------------------------------------------------------------------------
# Valid question data helpers
# ---------------------------------------------------------------------------

def _short_answer_data(**overrides):
    """Return a valid short-answer question dict."""
    base = {
        "question_text": "What is 2 + 2?",
        "question_type": "short_answer",
        "marks": 2,
        "difficulty": "easy",
        "topic": "Arithmetic",
        "answer": "4",
        "solution": "2 + 2 = 4",
    }
    base.update(overrides)
    return base


def _mcq_data(**overrides):
    """Return a valid MCQ question dict."""
    base = {
        "question_text": "What is the square root of 16?",
        "question_type": "multiple_choice",
        "marks": 1,
        "difficulty": "easy",
        "topic": "Square Roots",
        "answer": "4",
        "solution": "√16 = 4",
        "options": ["2", "3", "4", "5"],
        "correct_option": "4",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Question validation tests
# ---------------------------------------------------------------------------

class TestQuestionValidation:
    def test_valid_short_answer(self):
        q = Question.model_validate(_short_answer_data())
        assert q.question_type == QuestionType.SHORT_ANSWER
        assert q.marks == 2

    def test_valid_mcq(self):
        q = Question.model_validate(_mcq_data())
        assert q.question_type == QuestionType.MULTIPLE_CHOICE
        assert q.correct_option == "4"
        assert "4" in q.options

    def test_mcq_missing_options_fails(self):
        with pytest.raises(Exception):
            Question.model_validate(_mcq_data(options=None))

    def test_mcq_too_few_options_fails(self):
        with pytest.raises(Exception):
            Question.model_validate(_mcq_data(options=["4"]))

    def test_mcq_missing_correct_option_fails(self):
        with pytest.raises(Exception):
            Question.model_validate(_mcq_data(correct_option=None))

    def test_mcq_correct_option_not_in_options_fails(self):
        with pytest.raises(Exception):
            Question.model_validate(_mcq_data(correct_option="99"))

    def test_marks_must_be_positive(self):
        with pytest.raises(Exception):
            Question.model_validate(_short_answer_data(marks=0))

    def test_marks_negative_fails(self):
        with pytest.raises(Exception):
            Question.model_validate(_short_answer_data(marks=-5))

    def test_empty_question_text_fails(self):
        with pytest.raises(Exception):
            Question.model_validate(_short_answer_data(question_text=""))

    def test_invalid_question_type_fails(self):
        with pytest.raises(Exception):
            Question.model_validate(_short_answer_data(question_type="essay"))

    def test_invalid_difficulty_fails(self):
        with pytest.raises(Exception):
            Question.model_validate(_short_answer_data(difficulty="impossible"))

    def test_long_answer_valid(self):
        q = Question.model_validate(_short_answer_data(
            question_type="long_answer",
            marks=5,
            question_text="Prove that the sum of angles in a triangle is 180°.",
            answer="Proof using parallel line...",
            solution="Draw a line through one vertex parallel to the opposite side...",
        ))
        assert q.question_type == QuestionType.LONG_ANSWER

    def test_non_mcq_options_ignored(self):
        """Non-MCQ questions can have options=None without error."""
        q = Question.model_validate(_short_answer_data(options=None, correct_option=None))
        assert q.options is None


# ---------------------------------------------------------------------------
# Phase 3: New field tests — question_id, grade, subject, chapter,
#           source_references, validation_status
# ---------------------------------------------------------------------------

class TestQuestionNewFields:
    """Tests for the enhanced Question schema fields added in Phase 3."""

    def test_question_id_auto_generated(self):
        """question_id should be auto-generated when not provided."""
        q = Question.model_validate(_short_answer_data())
        assert q.question_id  # non-empty
        assert len(q.question_id) == 10  # hex[:10]

    def test_question_id_unique(self):
        """Two questions should get different auto-generated IDs."""
        q1 = Question.model_validate(_short_answer_data())
        q2 = Question.model_validate(_short_answer_data())
        assert q1.question_id != q2.question_id

    def test_question_id_can_be_set_explicitly(self):
        """An explicit question_id should be preserved."""
        q = Question.model_validate(_short_answer_data(question_id="custom123"))
        assert q.question_id == "custom123"

    def test_grade_defaults_to_empty(self):
        """Grade should default to empty string when not provided."""
        q = Question.model_validate(_short_answer_data())
        assert q.grade == ""

    def test_grade_can_be_set(self):
        q = Question.model_validate(_short_answer_data(grade="10"))
        assert q.grade == "10"

    def test_subject_defaults_to_mathematics(self):
        """Subject should default to 'Mathematics' when not provided."""
        q = Question.model_validate(_short_answer_data())
        assert q.subject == "Mathematics"

    def test_subject_can_be_set(self):
        q = Question.model_validate(_short_answer_data(subject="Physics"))
        assert q.subject == "Physics"

    def test_chapter_defaults_to_empty(self):
        q = Question.model_validate(_short_answer_data())
        assert q.chapter == ""

    def test_chapter_can_be_set(self):
        q = Question.model_validate(_short_answer_data(chapter="Chapter 5: Algebra"))
        assert q.chapter == "Chapter 5: Algebra"

    def test_source_references_default_empty(self):
        """source_references should default to an empty list."""
        q = Question.model_validate(_short_answer_data())
        assert q.source_references == []

    def test_source_references_can_be_set(self):
        refs = [
            {"source_filename": "textbook.pdf", "chunk_index": 0, "passage_preview": "Some text..."},
            {"source_filename": "textbook.pdf", "chunk_index": 3, "passage_preview": "More text..."},
        ]
        q = Question.model_validate(_short_answer_data(source_references=refs))
        assert len(q.source_references) == 2
        assert q.source_references[0].source_filename == "textbook.pdf"
        assert q.source_references[1].chunk_index == 3

    def test_validation_status_default_pending(self):
        """validation_status should default to 'pending'."""
        q = Question.model_validate(_short_answer_data())
        assert q.validation_status == ValidationStatus.PENDING

    def test_validation_status_can_be_set(self):
        q = Question.model_validate(_short_answer_data(validation_status="valid"))
        assert q.validation_status == ValidationStatus.VALID

    def test_invalid_validation_status_fails(self):
        with pytest.raises(Exception):
            Question.model_validate(_short_answer_data(validation_status="unknown"))


class TestSourceReference:
    """Tests for the SourceReference model."""

    def test_valid_source_reference(self):
        ref = SourceReference(
            source_filename="chapter1.pdf",
            chunk_index=5,
            passage_preview="The quadratic formula is..."
        )
        assert ref.source_filename == "chapter1.pdf"
        assert ref.chunk_index == 5

    def test_source_reference_defaults(self):
        ref = SourceReference()
        assert ref.source_filename == ""
        assert ref.chunk_index is None
        assert ref.passage_preview == ""

    def test_source_reference_round_trip(self):
        """SourceReference should serialize/deserialize correctly."""
        ref = SourceReference(
            source_filename="syllabus.docx",
            chunk_index=2,
            passage_preview="Trigonometric identities..."
        )
        data = ref.model_dump()
        ref2 = SourceReference.model_validate(data)
        assert ref2.source_filename == ref.source_filename
        assert ref2.chunk_index == ref.chunk_index


class TestValidationStatusEnum:
    """Tests for the ValidationStatus enum values."""

    def test_all_values(self):
        assert ValidationStatus.PENDING == "pending"
        assert ValidationStatus.VALID == "valid"
        assert ValidationStatus.INVALID == "invalid"
        assert ValidationStatus.FIXED == "fixed"


# ---------------------------------------------------------------------------
# Paper validation tests
# ---------------------------------------------------------------------------

class TestPaperValidation:
    def _make_paper_data(self, questions, num_questions=None, total_marks=None):
        """Build a paper data dict from question dicts."""
        q_objects = [Question.model_validate(q) for q in questions]
        return {
            "paper_id": "test123",
            "title": "Test Paper",
            "grade": "10",
            "subject": "Mathematics",
            "topic": "Algebra",
            "difficulty": "medium",
            "questions": [q.model_dump() for q in q_objects],
            "num_questions": num_questions if num_questions is not None else len(q_objects),
            "total_marks": total_marks if total_marks is not None else sum(q.marks for q in q_objects),
        }

    def test_valid_paper(self):
        data = self._make_paper_data([
            _short_answer_data(marks=3),
            _mcq_data(marks=2),
        ])
        paper = Paper.model_validate(data)
        assert paper.num_questions == 2
        assert paper.total_marks == 5

    def test_wrong_question_count_fails(self):
        data = self._make_paper_data(
            [_short_answer_data(marks=3)],
            num_questions=5,
        )
        with pytest.raises(Exception, match="num_questions"):
            Paper.model_validate(data)

    def test_wrong_total_marks_fails(self):
        data = self._make_paper_data(
            [_short_answer_data(marks=3)],
            total_marks=100,
        )
        with pytest.raises(Exception, match="total_marks"):
            Paper.model_validate(data)

    def test_paper_with_single_question(self):
        data = self._make_paper_data([_short_answer_data(marks=5)])
        paper = Paper.model_validate(data)
        assert paper.num_questions == 1
        assert paper.total_marks == 5

    def test_paper_questions_have_ids(self):
        """Every question in a paper should have a unique question_id."""
        data = self._make_paper_data([
            _short_answer_data(marks=3),
            _short_answer_data(marks=2, question_text="What is 3+3?", answer="6"),
        ])
        paper = Paper.model_validate(data)
        ids = [q.question_id for q in paper.questions]
        assert len(ids) == 2
        assert ids[0] != ids[1]  # auto-generated IDs should be unique

    def test_paper_preserves_source_references(self):
        """Source references should survive paper serialization round-trip."""
        refs = [{"source_filename": "test.pdf", "chunk_index": 0, "passage_preview": "Preview"}]
        data = self._make_paper_data([
            _short_answer_data(marks=5, source_references=refs, validation_status="valid"),
        ])
        paper = Paper.model_validate(data)
        assert len(paper.questions[0].source_references) == 1
        assert paper.questions[0].source_references[0].source_filename == "test.pdf"
        assert paper.questions[0].validation_status == ValidationStatus.VALID


# ---------------------------------------------------------------------------
# Phase 4: DocumentRecord and DocumentSummary tests
# ---------------------------------------------------------------------------

class TestDocumentRecordValidation:
    """Tests for the DocumentRecord schema added in Phase 4."""

    def test_valid_document_record(self):
        rec = DocumentRecord(
            document_id="abc123",
            original_filename="textbook.pdf",
            safe_filename="abc123.pdf",
            num_chunks=10,
            num_pages=5,
            file_size_bytes=50000,
            has_index=True,
            text_preview="Chapter 1: Algebra...",
        )
        assert rec.document_id == "abc123"
        assert rec.num_chunks == 10
        assert rec.has_index is True

    def test_document_record_defaults(self):
        """Check that optional fields get sensible defaults."""
        rec = DocumentRecord(
            document_id="xyz",
            original_filename="file.docx",
            num_chunks=3,
        )
        assert rec.safe_filename == ""
        assert rec.num_pages is None
        assert rec.file_size_bytes == 0
        assert rec.has_index is False
        assert rec.text_preview == ""
        assert rec.uploaded_at  # auto-generated timestamp

    def test_document_record_negative_chunks_fails(self):
        with pytest.raises(Exception):
            DocumentRecord(
                document_id="bad",
                original_filename="file.pdf",
                num_chunks=-1,
            )

    def test_document_record_negative_file_size_fails(self):
        with pytest.raises(Exception):
            DocumentRecord(
                document_id="bad",
                original_filename="file.pdf",
                num_chunks=1,
                file_size_bytes=-100,
            )

    def test_document_record_round_trip(self):
        """DocumentRecord should serialize/deserialize correctly."""
        rec = DocumentRecord(
            document_id="roundtrip",
            original_filename="test.pdf",
            safe_filename="roundtrip.pdf",
            num_chunks=7,
            num_pages=3,
            file_size_bytes=12345,
            has_index=True,
            text_preview="Some math content...",
        )
        data = rec.model_dump()
        rec2 = DocumentRecord.model_validate(data)
        assert rec2.document_id == rec.document_id
        assert rec2.num_chunks == rec.num_chunks
        assert rec2.text_preview == rec.text_preview


class TestDocumentSummary:
    """Tests for the DocumentSummary schema."""

    def test_valid_summary(self):
        s = DocumentSummary(
            document_id="doc1",
            original_filename="chapter.pdf",
            num_chunks=8,
            file_size_bytes=20000,
            uploaded_at="2026-01-01T00:00:00",
            has_index=True,
        )
        assert s.document_id == "doc1"
        assert s.has_index is True

    def test_summary_from_record(self):
        """A summary should be constructable from a full record's data."""
        rec = DocumentRecord(
            document_id="fromrec",
            original_filename="syllabus.docx",
            num_chunks=4,
            file_size_bytes=8000,
            has_index=False,
        )
        s = DocumentSummary(
            document_id=rec.document_id,
            original_filename=rec.original_filename,
            num_chunks=rec.num_chunks,
            file_size_bytes=rec.file_size_bytes,
            uploaded_at=rec.uploaded_at,
            has_index=rec.has_index,
        )
        assert s.document_id == rec.document_id
        assert s.num_chunks == rec.num_chunks


class TestTextDocumentRequest:
    """Tests for the TextDocumentRequest schema."""

    def test_valid_request(self):
        req = TextDocumentRequest(
            title="Chapter 1 Real Numbers",
            content="Euclid's division lemma states that for given positive integers a and b...",
        )
        assert req.title == "Chapter 1 Real Numbers"
        assert len(req.content) > 10

    def test_short_content_fails(self):
        with pytest.raises(Exception):
            TextDocumentRequest(
                title="Title",
                content="Short",
            )

    def test_empty_title_fails(self):
        with pytest.raises(Exception):
            TextDocumentRequest(
                title="",
                content="Valid length content here for testing.",
            )


# ---------------------------------------------------------------------------
# SectionBlueprint & ExamBlueprint Tests
# ---------------------------------------------------------------------------

class TestSectionBlueprint:
    """Unit tests for SectionBlueprint validation."""

    def test_valid_uniform_marks(self):
        sb = SectionBlueprint(
            section_id="A",
            title="Section A: MCQs",
            question_type=QuestionType.MULTIPLE_CHOICE,
            num_questions=5,
            section_marks=5,
            marks_per_question=1,
        )
        assert sb.get_marks_distribution() == [1, 1, 1, 1, 1]

    def test_valid_explicit_distribution(self):
        sb = SectionBlueprint(
            section_id="B",
            title="Section B: Short Answer",
            question_type=QuestionType.SHORT_ANSWER,
            num_questions=3,
            section_marks=7,
            marks_distribution=[2, 2, 3],
        )
        assert sb.get_marks_distribution() == [2, 2, 3]

    def test_marks_distribution_length_mismatch_fails(self):
        with pytest.raises(ValueError, match="marks_distribution length"):
            SectionBlueprint(
                section_id="B",
                title="Section B",
                question_type=QuestionType.SHORT_ANSWER,
                num_questions=3,
                section_marks=6,
                marks_distribution=[2, 2],
            )

    def test_marks_distribution_sum_mismatch_fails(self):
        with pytest.raises(ValueError, match="sum of marks_distribution"):
            SectionBlueprint(
                section_id="B",
                title="Section B",
                question_type=QuestionType.SHORT_ANSWER,
                num_questions=3,
                section_marks=10,
                marks_distribution=[2, 2, 2],
            )

    def test_marks_distribution_nonpositive_fails(self):
        with pytest.raises(ValueError, match="positive integers"):
            SectionBlueprint(
                section_id="B",
                title="Section B",
                question_type=QuestionType.SHORT_ANSWER,
                num_questions=2,
                section_marks=2,
                marks_distribution=[2, 0],
            )

    def test_marks_per_question_mismatch_fails(self):
        with pytest.raises(ValueError, match="does not match section_marks"):
            SectionBlueprint(
                section_id="A",
                title="Section A",
                question_type=QuestionType.MULTIPLE_CHOICE,
                num_questions=5,
                section_marks=10,
                marks_per_question=1,
            )

    def test_section_marks_less_than_questions_fails(self):
        with pytest.raises(ValueError, match="cannot be less than num_questions"):
            SectionBlueprint(
                section_id="A",
                title="Section A",
                question_type=QuestionType.MULTIPLE_CHOICE,
                num_questions=5,
                section_marks=3,
            )


class TestExamBlueprint:
    """Unit tests for ExamBlueprint validation."""

    def test_valid_exam_blueprint(self):
        sec_a = SectionBlueprint(
            section_id="A",
            title="Section A: MCQs",
            question_type=QuestionType.MULTIPLE_CHOICE,
            num_questions=4,
            section_marks=4,
            marks_per_question=1,
        )
        sec_b = SectionBlueprint(
            section_id="B",
            title="Section B: Long Answer",
            question_type=QuestionType.LONG_ANSWER,
            num_questions=2,
            section_marks=10,
            marks_per_question=5,
        )
        bp = ExamBlueprint(
            title="Mid-term Blueprint",
            sections=[sec_a, sec_b],
            total_questions=6,
            total_marks=14,
        )
        assert len(bp.sections) == 2
        assert bp.total_questions == 6
        assert bp.total_marks == 14

    def test_duplicate_section_ids_fails(self):
        sec_a1 = SectionBlueprint(
            section_id="A",
            title="Section A1",
            question_type=QuestionType.MULTIPLE_CHOICE,
            num_questions=2,
            section_marks=2,
        )
        sec_a2 = SectionBlueprint(
            section_id="A",
            title="Section A2",
            question_type=QuestionType.SHORT_ANSWER,
            num_questions=2,
            section_marks=4,
        )
        with pytest.raises(ValueError, match="Duplicate section identifiers"):
            ExamBlueprint(
                sections=[sec_a1, sec_a2],
                total_questions=4,
                total_marks=6,
            )

    def test_total_questions_mismatch_fails(self):
        sec_a = SectionBlueprint(
            section_id="A",
            title="Section A",
            question_type=QuestionType.MULTIPLE_CHOICE,
            num_questions=3,
            section_marks=3,
        )
        with pytest.raises(ValueError, match="Sum of section question counts"):
            ExamBlueprint(
                sections=[sec_a],
                total_questions=5,
                total_marks=3,
            )

    def test_total_marks_mismatch_fails(self):
        sec_a = SectionBlueprint(
            section_id="A",
            title="Section A",
            question_type=QuestionType.MULTIPLE_CHOICE,
            num_questions=3,
            section_marks=3,
        )
        with pytest.raises(ValueError, match="Sum of section marks"):
            ExamBlueprint(
                sections=[sec_a],
                total_questions=3,
                total_marks=5,
            )

    def test_difficulty_distribution_validation(self):
        sec_a = SectionBlueprint(
            section_id="A",
            title="Section A",
            question_type=QuestionType.MULTIPLE_CHOICE,
            num_questions=4,
            section_marks=4,
        )
        # Valid distribution
        bp = ExamBlueprint(
            sections=[sec_a],
            total_questions=4,
            total_marks=4,
            difficulty_distribution={Difficulty.EASY: 2, Difficulty.MEDIUM: 2},
        )
        assert bp.difficulty_distribution[Difficulty.EASY] == 2

        # Invalid distribution sum
        with pytest.raises(ValueError, match="Sum of questions in difficulty_distribution"):
            ExamBlueprint(
                sections=[sec_a],
                total_questions=4,
                total_marks=4,
                difficulty_distribution={Difficulty.EASY: 1, Difficulty.MEDIUM: 1},
            )


class TestDefaultBlueprint:
    """Unit tests for create_default_blueprint."""

    def test_single_type_default_blueprint(self):
        bp = create_default_blueprint(
            num_questions=5,
            total_marks=15,
            question_types=[QuestionType.SHORT_ANSWER],
            topic="Polynomials",
        )
        assert len(bp.sections) == 1
        assert bp.sections[0].num_questions == 5
        assert bp.sections[0].section_marks == 15
        assert bp.total_questions == 5
        assert bp.total_marks == 15

    def test_three_types_default_blueprint(self):
        bp = create_default_blueprint(
            num_questions=10,
            total_marks=25,
            question_types=[QuestionType.MULTIPLE_CHOICE, QuestionType.SHORT_ANSWER, QuestionType.LONG_ANSWER],
            topic="Real Numbers",
        )
        assert sum(s.num_questions for s in bp.sections) == 10
        assert sum(s.section_marks for s in bp.sections) == 25
        assert len(bp.sections) == 3

    def test_total_marks_less_than_questions_fails(self):
        with pytest.raises(ValueError, match="must be at least num_questions"):
            create_default_blueprint(
                num_questions=10,
                total_marks=5,
                question_types=[QuestionType.MULTIPLE_CHOICE],
            )


class TestBatchPaperGenerateRequest:
    """Unit tests for BatchPaperGenerateRequest and request blueprint integration."""

    def test_valid_batch_request_with_defaults(self):
        req = BatchPaperGenerateRequest(
            document_id="doc123",
            grade="10",
            topic="Triangles",
            num_questions=6,
            total_marks=16,
            num_versions=3,
        )
        assert req.num_versions == 3
        bp = req.get_or_create_blueprint()
        assert bp.total_questions == 6
        assert bp.total_marks == 16

    def test_batch_request_num_versions_bounds(self):
        # 1 version is valid
        req1 = BatchPaperGenerateRequest(
            document_id="doc123",
            grade="10",
            topic="Triangles",
            num_questions=5,
            total_marks=10,
            num_versions=1,
        )
        assert req1.num_versions == 1

        # 10 versions is valid
        req10 = BatchPaperGenerateRequest(
            document_id="doc123",
            grade="10",
            topic="Triangles",
            num_questions=5,
            total_marks=10,
            num_versions=10,
        )
        assert req10.num_versions == 10

        # 0 versions fails
        with pytest.raises(Exception):
            BatchPaperGenerateRequest(
                document_id="doc123",
                grade="10",
                topic="Triangles",
                num_questions=5,
                total_marks=10,
                num_versions=0,
            )

        # 11 versions fails
        with pytest.raises(Exception):
            BatchPaperGenerateRequest(
                document_id="doc123",
                grade="10",
                topic="Triangles",
                num_questions=5,
                total_marks=10,
                num_versions=11,
            )

    def test_batch_request_with_explicit_blueprint(self):
        sec = SectionBlueprint(
            section_id="A",
            title="Section A",
            question_type=QuestionType.SHORT_ANSWER,
            num_questions=4,
            section_marks=8,
            marks_per_question=2,
        )
        bp = ExamBlueprint(
            title="Custom Blueprint",
            sections=[sec],
            total_questions=4,
            total_marks=8,
        )
        req = BatchPaperGenerateRequest(
            document_id="doc123",
            grade="10",
            topic="Circles",
            blueprint=bp,
            num_versions=2,
        )
        assert req.num_questions == 4
        assert req.total_marks == 8
        assert req.get_or_create_blueprint() == bp


class TestPaperSectionsAndMetadata:
    """Unit tests for Paper sections and version metadata."""

    def test_paper_with_valid_sections(self):
        q1 = Question.model_validate(_mcq_data(question_id="q1", marks=1))
        q2 = Question.model_validate(_mcq_data(question_id="q2", marks=1))
        q3 = Question.model_validate(_short_answer_data(question_id="q3", marks=2))

        sec_a = PaperSection(
            section_id="A",
            title="Section A",
            question_type=QuestionType.MULTIPLE_CHOICE,
            num_questions=2,
            section_marks=2,
            questions=[q1, q2],
        )
        sec_b = PaperSection(
            section_id="B",
            title="Section B",
            question_type=QuestionType.SHORT_ANSWER,
            num_questions=1,
            section_marks=2,
            questions=[q3],
        )

        paper = Paper(
            paper_id="paper123",
            title="Math Paper",
            grade="10",
            subject="Mathematics",
            topic="Algebra",
            difficulty=Difficulty.MEDIUM,
            questions=[q1, q2, q3],
            num_questions=3,
            total_marks=4,
            sections=[sec_a, sec_b],
            version_number=1,
            num_versions=3,
            batch_id="batch456",
        )
        assert len(paper.sections) == 2
        assert paper.version_number == 1
        assert paper.batch_id == "batch456"

    def test_paper_section_question_count_mismatch_fails(self):
        q1 = Question.model_validate(_mcq_data(question_id="q1", marks=1))
        sec_a = PaperSection(
            section_id="A",
            title="Section A",
            question_type=QuestionType.MULTIPLE_CHOICE,
            num_questions=2,  # expects 2, but has 1
            section_marks=1,
            questions=[q1],
        )
        with pytest.raises(ValueError, match="expected 2 questions, but has 1"):
            Paper(
                paper_id="paper123",
                title="Math Paper",
                grade="10",
                subject="Mathematics",
                topic="Algebra",
                difficulty=Difficulty.MEDIUM,
                questions=[q1],
                num_questions=1,
                total_marks=1,
                sections=[sec_a],
            )

    def test_paper_duplicate_question_id_fails(self):
        q1 = Question.model_validate(_mcq_data(question_id="dup_id", marks=1))
        q2 = Question.model_validate(_mcq_data(question_id="dup_id", marks=1))
        with pytest.raises(ValueError, match="duplicate question IDs"):
            Paper(
                paper_id="paper123",
                title="Math Paper",
                grade="10",
                subject="Mathematics",
                topic="Algebra",
                difficulty=Difficulty.MEDIUM,
                questions=[q1, q2],
                num_questions=2,
                total_marks=2,
            )


