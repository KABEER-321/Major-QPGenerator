# app/pdf_export.py
# Generates a readable PDF of a question paper using ReportLab.
# Includes paper metadata, numbered questions, marks, MCQ options,
# and a separate answer key / solutions section.

from io import BytesIO

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    PageBreak,
    Table,
    TableStyle,
)
from reportlab.lib import colors

from app.schemas import Paper, QuestionType, ExamType, format_grade_display


def _get_styles():
    """Build custom paragraph styles for the PDF."""
    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(
        name="PaperTitle",
        parent=styles["Title"],
        fontSize=18,
        spaceAfter=6,
        alignment=TA_CENTER,
    ))
    styles.add(ParagraphStyle(
        name="PaperMeta",
        parent=styles["Normal"],
        fontSize=11,
        spaceAfter=2,
        alignment=TA_CENTER,
        textColor=colors.grey,
    ))
    styles.add(ParagraphStyle(
        name="SectionTitle",
        parent=styles["Heading2"],
        fontSize=14,
        spaceBefore=16,
        spaceAfter=8,
    ))
    styles.add(ParagraphStyle(
        name="QuestionText",
        parent=styles["Normal"],
        fontSize=11,
        spaceAfter=4,
        leading=15,
        alignment=TA_JUSTIFY,
    ))
    styles.add(ParagraphStyle(
        name="OptionText",
        parent=styles["Normal"],
        fontSize=10,
        leftIndent=24,
        spaceAfter=2,
    ))
    styles.add(ParagraphStyle(
        name="AnswerText",
        parent=styles["Normal"],
        fontSize=10,
        spaceAfter=4,
        leading=13,
        textColor=colors.HexColor("#333333"),
    ))
    styles.add(ParagraphStyle(
        name="Instructions",
        parent=styles["Normal"],
        fontSize=10,
        spaceAfter=12,
        leading=14,
        textColor=colors.HexColor("#555555"),
    ))
    return styles


def generate_pdf(paper: Paper, include_answers: bool = True) -> bytes:
    """
    Generate a PDF for the given paper and return it as bytes.

    Args:
        paper: The Paper object to render.
        include_answers: If True, append an answer key and solutions section.
    """
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )

    styles = _get_styles()
    story = []

    # --- Header ---
    exam_type = getattr(paper, "exam_type", ExamType.FINAL) or ExamType.FINAL
    academic_class = getattr(paper, "academic_grade", None) or paper.grade
    curriculum_class = getattr(paper, "curriculum_grade", None) or paper.grade
    scope = getattr(paper, "syllabus_scope", None)

    story.append(Paragraph(paper.title, styles["PaperTitle"]))
    if paper.num_versions > 1:
        story.append(Paragraph(
            f"Paper Version: {paper.version_number} of {paper.num_versions} | Batch: {paper.batch_id or 'N/A'}",
            styles["PaperMeta"],
        ))

    if exam_type == ExamType.PREREQUISITE:
        story.append(Paragraph(
            f"<b>Academic Class:</b> {format_grade_display(academic_class)} &nbsp;|&nbsp; "
            f"<b>Curriculum Used:</b> {format_grade_display(curriculum_class)} {paper.subject}",
            styles["PaperMeta"],
        ))
    elif exam_type == ExamType.MIDTERM:
        story.append(Paragraph(
            f"<b>Academic Class:</b> {format_grade_display(academic_class)} &nbsp;|&nbsp; <b>Subject:</b> {paper.subject} &nbsp;|&nbsp; "
            f"<b>Syllabus Scope:</b> {scope or 'First Half of Syllabus'}",
            styles["PaperMeta"],
        ))
    else:
        story.append(Paragraph(
            f"<b>Academic Class:</b> {format_grade_display(academic_class)} &nbsp;|&nbsp; <b>Subject:</b> {paper.subject} &nbsp;|&nbsp; "
            f"<b>Syllabus Scope:</b> {scope or 'Full Syllabus'}",
            styles["PaperMeta"],
        ))

    story.append(Paragraph(
        f"Topic: {paper.topic} &nbsp;|&nbsp; Difficulty: {paper.difficulty.value.title()} &nbsp;|&nbsp; "
        f"Total Questions: {paper.num_questions} &nbsp;|&nbsp; Total Marks: {paper.total_marks}",
        styles["PaperMeta"],
    ))
    story.append(Spacer(1, 12))

    # --- Instructions ---
    story.append(Paragraph("General Instructions:", styles["SectionTitle"]))
    instructions = [
        "1. Attempt all questions.",
        "2. Marks for each question are indicated next to the question number.",
        "3. Show all working for full marks.",
        f"4. Total marks: {paper.total_marks}.",
    ]
    if paper.sections:
        instructions.append(f"5. The question paper consists of {len(paper.sections)} sections.")
    for instr in instructions:
        story.append(Paragraph(instr, styles["Instructions"]))
    story.append(Spacer(1, 8))

    # --- Questions ---
    if paper.sections:
        q_num = 1
        for sec in paper.sections:
            story.append(Paragraph(
                f"<b>{sec.title}</b> ({sec.num_questions} Questions | {sec.section_marks} Marks)",
                styles["SectionTitle"],
            ))
            for q in sec.questions:
                safe_text = (
                    q.question_text
                    .replace("&", "&amp;")
                    .replace("<", "&lt;")
                    .replace(">", "&gt;")
                )
                q_line = f"<b>Q{q_num}.</b> [{q.marks} mark{'s' if q.marks > 1 else ''}] {safe_text}"
                story.append(Paragraph(q_line, styles["QuestionText"]))

                if q.question_type == QuestionType.MULTIPLE_CHOICE and q.options:
                    option_labels = "ABCDEFGHIJ"
                    for j, opt in enumerate(q.options):
                        safe_opt = (
                            opt.replace("&", "&amp;")
                            .replace("<", "&lt;")
                            .replace(">", "&gt;")
                        )
                        label = option_labels[j] if j < len(option_labels) else str(j + 1)
                        story.append(Paragraph(f"({label}) {safe_opt}", styles["OptionText"]))

                story.append(Spacer(1, 6))
                q_num += 1
    else:
        story.append(Paragraph("Questions", styles["SectionTitle"]))
        for i, q in enumerate(paper.questions, 1):
            safe_text = (
                q.question_text
                .replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
            )
            q_line = f"<b>Q{i}.</b> [{q.marks} mark{'s' if q.marks > 1 else ''}] {safe_text}"
            story.append(Paragraph(q_line, styles["QuestionText"]))

            if q.question_type == QuestionType.MULTIPLE_CHOICE and q.options:
                option_labels = "ABCDEFGHIJ"
                for j, opt in enumerate(q.options):
                    safe_opt = (
                        opt.replace("&", "&amp;")
                        .replace("<", "&lt;")
                        .replace(">", "&gt;")
                    )
                    label = option_labels[j] if j < len(option_labels) else str(j + 1)
                    story.append(Paragraph(f"({label}) {safe_opt}", styles["OptionText"]))

            story.append(Spacer(1, 6))

    # --- Answer Key & Solutions ---
    if include_answers:
        story.append(PageBreak())
        story.append(Paragraph("Answer Key &amp; Solutions", styles["SectionTitle"]))

        q_ans_num = 1
        for q in paper.questions:
            safe_answer = (
                q.answer
                .replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
            )
            id_tag = f" <font size='8' color='#999999'>[{q.question_id}]</font>" if q.question_id else ""
            answer_line = f"<b>Q{q_ans_num}.</b>{id_tag} <b>Answer:</b> {safe_answer}"
            story.append(Paragraph(answer_line, styles["AnswerText"]))

            if q.question_type == QuestionType.MULTIPLE_CHOICE and q.correct_option:
                safe_correct = (
                    q.correct_option
                    .replace("&", "&amp;")
                    .replace("<", "&lt;")
                    .replace(">", "&gt;")
                )
                story.append(Paragraph(
                    f"&nbsp;&nbsp;&nbsp;Correct Option: {safe_correct}",
                    styles["AnswerText"],
                ))

            if q.solution:
                safe_solution = (
                    q.solution
                    .replace("&", "&amp;")
                    .replace("<", "&lt;")
                    .replace(">", "&gt;")
                )
                story.append(Paragraph(
                    f"&nbsp;&nbsp;&nbsp;<i>Solution:</i> {safe_solution}",
                    styles["AnswerText"],
                ))

            story.append(Spacer(1, 4))
            q_ans_num += 1

        # --- Source References section ---
        # Collect all unique source references from questions
        all_refs = []
        seen = set()
        for q in paper.questions:
            for ref in q.source_references:
                key = (ref.source_filename, ref.chunk_index)
                if key not in seen:
                    seen.add(key)
                    all_refs.append(ref)

        if all_refs:
            story.append(Spacer(1, 12))
            story.append(Paragraph("Source References", styles["SectionTitle"]))
            story.append(Paragraph(
                "The questions above were generated from the following "
                "passages in the uploaded syllabus/textbook:",
                styles["Instructions"],
            ))
            for ref in all_refs:
                safe_preview = (
                    ref.passage_preview
                    .replace("&", "&amp;")
                    .replace("<", "&lt;")
                    .replace(">", "&gt;")
                )
                chunk_label = f"Chunk {ref.chunk_index}" if ref.chunk_index is not None else "Chunk"
                story.append(Paragraph(
                    f"<b>{ref.source_filename}</b> — {chunk_label}: "
                    f"<i>{safe_preview}</i>",
                    styles["AnswerText"],
                ))
                story.append(Spacer(1, 2))

    # Build the PDF
    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes
