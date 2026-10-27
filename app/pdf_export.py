# app/pdf_export.py
# Generates a formal, curriculum-aligned examination paper PDF using ReportLab.
# Follows the structural design of the official Karnataka Board model paper (Class 5 QP.pdf):
# - Formal examination board and school header
# - Metadata grid (Class, Subject, Examination Type, Medium, Marks, Duration)
# - Student & room invigilator information box
# - General examination instructions
# - Numbered sections with section-specific instructions and marks allocation (e.g. 10 × 2 = 20)
# - Continuous question numbering across sections
# - MCQ options with student answer boxes [ ]
# - Ruled workspace for student working/answers in student copy
# - Comprehensive worked solutions and marking schemes in teacher copy

from __future__ import annotations

import html
from io import BytesIO

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    HRFlowable,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from app.schemas import ExamType, Paper, QuestionType, format_grade_display


def _calculate_exam_duration(total_marks: int) -> str:
    """Calculate realistic examination duration based on total marks."""
    if total_marks <= 25:
        return "1 Hour"
    elif total_marks <= 50:
        return "2 Hours"
    elif total_marks <= 75:
        return "2 Hours 30 Minutes"
    else:
        return "3 Hours"


def _escape(text: str | None) -> str:
    """Safely escape text for ReportLab Paragraphs."""
    if not text:
        return ""
    return html.escape(str(text))


def _make_ruled_lines(num_lines: int, width_cm: float = 18.0, height_cm: float = 0.58) -> Table:
    """Produce light ruled writing lines for student answer work, matching school exam papers."""
    data = [[""] for _ in range(num_lines)]
    table = Table(data, colWidths=[width_cm * cm], rowHeights=[height_cm * cm] * num_lines)
    table.setStyle(
        TableStyle([
            ("LINEBELOW", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
            ("TOPPADDING", (0, 0), (-1, -1), 0),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ])
    )
    return table


def _get_styles():
    """Build custom paragraph styles for the examination PDF."""
    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(
        name="BoardHeader",
        parent=styles["Normal"],
        fontSize=12,
        leading=15,
        fontName="Helvetica-Bold",
        alignment=TA_CENTER,
        textColor=colors.HexColor("#0F172A"),
        spaceAfter=2,
    ))

    styles.add(ParagraphStyle(
        name="AssessmentSubHeader",
        parent=styles["Normal"],
        fontSize=10.5,
        leading=14,
        fontName="Helvetica-Bold",
        alignment=TA_CENTER,
        textColor=colors.HexColor("#334155"),
        spaceAfter=6,
    ))

    styles.add(ParagraphStyle(
        name="MetaCell",
        parent=styles["Normal"],
        fontSize=9.5,
        leading=13,
        textColor=colors.HexColor("#0F172A"),
    ))

    styles.add(ParagraphStyle(
        name="BoxHeader",
        parent=styles["Normal"],
        fontSize=8.5,
        leading=11,
        fontName="Helvetica-Bold",
        textColor=colors.HexColor("#1E293B"),
    ))

    styles.add(ParagraphStyle(
        name="BoxText",
        parent=styles["Normal"],
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#334155"),
    ))

    styles.add(ParagraphStyle(
        name="InstructionsHeading",
        parent=styles["Normal"],
        fontSize=9.5,
        leading=13,
        fontName="Helvetica-Bold",
        textColor=colors.HexColor("#0F172A"),
        spaceBefore=6,
        spaceAfter=2,
    ))

    styles.add(ParagraphStyle(
        name="InstructionItem",
        parent=styles["Normal"],
        fontSize=8.5,
        leading=11.5,
        textColor=colors.HexColor("#475569"),
    ))

    styles.add(ParagraphStyle(
        name="SecTitleLeft",
        parent=styles["Normal"],
        fontSize=10,
        leading=13,
        fontName="Helvetica-Bold",
        textColor=colors.HexColor("#0F172A"),
    ))

    styles.add(ParagraphStyle(
        name="SecTitleRight",
        parent=styles["Normal"],
        fontSize=10,
        leading=13,
        fontName="Helvetica-Bold",
        alignment=TA_RIGHT,
        textColor=colors.HexColor("#0F172A"),
    ))

    styles.add(ParagraphStyle(
        name="SecInstruction",
        parent=styles["Normal"],
        fontSize=9,
        leading=12,
        fontName="Helvetica-Oblique",
        textColor=colors.HexColor("#475569"),
    ))

    styles.add(ParagraphStyle(
        name="QuestionNumAndText",
        parent=styles["Normal"],
        fontSize=10,
        leading=14,
        alignment=TA_JUSTIFY,
        textColor=colors.HexColor("#0F172A"),
    ))

    styles.add(ParagraphStyle(
        name="OptionText",
        parent=styles["Normal"],
        fontSize=9.5,
        leading=13,
        textColor=colors.HexColor("#1E293B"),
    ))

    styles.add(ParagraphStyle(
        name="AnswerStudentLine",
        parent=styles["Normal"],
        fontSize=9.5,
        leading=13,
        textColor=colors.HexColor("#334155"),
    ))

    styles.add(ParagraphStyle(
        name="TeacherAnsHeader",
        parent=styles["Normal"],
        fontSize=9,
        leading=12,
        fontName="Helvetica-Bold",
        textColor=colors.HexColor("#1E3A8A"),
    ))

    styles.add(ParagraphStyle(
        name="TeacherAnsText",
        parent=styles["Normal"],
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#1E293B"),
    ))

    styles.add(ParagraphStyle(
        name="SectionHeadingDoc",
        parent=styles["Heading2"],
        fontSize=12,
        leading=15,
        fontName="Helvetica-Bold",
        textColor=colors.HexColor("#0F172A"),
        spaceBefore=10,
        spaceAfter=6,
    ))

    return styles


def generate_pdf(paper: Paper, include_answers: bool = True) -> bytes:
    """
    Generate a formal PDF for the question paper.

    Args:
        paper: The Paper object to render.
        include_answers:
            False -> Student examination paper (with ruled workspaces & answer lines, no answers).
            True  -> Teacher master paper / evaluation copy (with answers, solutions, and marking steps).
    """
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=1.5 * cm,
        rightMargin=1.5 * cm,
        topMargin=1.4 * cm,
        bottomMargin=1.4 * cm,
    )

    styles = _get_styles()
    story = []

    # 1. Header Information
    exam_type = getattr(paper, "exam_type", ExamType.FINAL) or ExamType.FINAL
    academic_class = getattr(paper, "academic_grade", None) or paper.grade
    curriculum_class = getattr(paper, "curriculum_grade", None) or paper.grade
    scope = getattr(paper, "syllabus_scope", None)
    duration_str = _calculate_exam_duration(paper.total_marks)

    story.append(Paragraph("KARNATAKA SCHOOL EXAMINATION AND ASSESSMENT BOARD", styles["BoardHeader"]))
    story.append(Paragraph(
        f"QPGen Curriculum-Aligned Assessment — Model Question Paper ({exam_type.display_name})",
        styles["AssessmentSubHeader"],
    ))

    # Paper Metadata Table (Class, Subject, Marks, Time, Medium)
    meta_row_1 = [
        Paragraph(f"<b>Class:</b> {format_grade_display(academic_class)}", styles["MetaCell"]),
        Paragraph(f"<b>Subject:</b> {_escape(paper.subject)}", styles["MetaCell"]),
        Paragraph(f"<b>Marks:</b> {paper.total_marks}", styles["MetaCell"]),
    ]
    meta_row_2 = [
        Paragraph(f"<b>Examination:</b> {exam_type.display_name}", styles["MetaCell"]),
        Paragraph("<b>Medium:</b> English", styles["MetaCell"]),
        Paragraph(f"<b>Time:</b> {duration_str}", styles["MetaCell"]),
    ]

    meta_table_data = [meta_row_1, meta_row_2]
    if exam_type == ExamType.PREREQUISITE:
        meta_table_data.append([
            Paragraph(
                f"<b>Curriculum Basis:</b> Class {curriculum_class} Syllabus (Prerequisite Knowledge Assessment)",
                styles["MetaCell"],
            ),
            Paragraph("", styles["MetaCell"]),
            Paragraph("", styles["MetaCell"]),
        ])
    elif exam_type == ExamType.MIDTERM:
        meta_table_data.append([
            Paragraph(
                f"<b>Syllabus Scope:</b> {_escape(scope or 'First Half of Syllabus')}",
                styles["MetaCell"],
            ),
            Paragraph("", styles["MetaCell"]),
            Paragraph("", styles["MetaCell"]),
        ])

    meta_table = Table(meta_table_data, colWidths=[6.2 * cm, 6.2 * cm, 5.6 * cm])
    meta_table.setStyle(
        TableStyle([
            ("BOX", (0, 0), (-1, -1), 0.75, colors.HexColor("#0F172A")),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ])
    )
    story.append(meta_table)
    story.append(Spacer(1, 8))

    # 2. Student & Room Invigilator Information Box (Standard Formal Layout)
    student_box_data = [
        [Paragraph("<b>Information to be filled by the Student</b>", styles["BoxHeader"])],
        [
            Paragraph(
                "Name of the Student : _____________________________________________________________________________",
                styles["BoxText"],
            )
        ],
        [
            Paragraph(
                "Student SATS / Roll No : [ &nbsp; ][ &nbsp; ][ &nbsp; ][ &nbsp; ][ &nbsp; ][ &nbsp; ][ &nbsp; ][ &nbsp; ][ &nbsp; ][ &nbsp; ] "
                "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; "
                "Signature of Student : ____________________",
                styles["BoxText"],
            )
        ],
        [Paragraph("<b>Information to be filled by the Room Invigilator</b>", styles["BoxHeader"])],
        [
            Paragraph(
                "School Name : ___________________________________________________________________________________",
                styles["BoxText"],
            )
        ],
        [
            Paragraph(
                "Cluster : ________________ &nbsp;&nbsp;&nbsp;&nbsp; Block : ________________ &nbsp;&nbsp;&nbsp;&nbsp; "
                "District : ________________ &nbsp;&nbsp;&nbsp;&nbsp; Invigilator Signature : ____________________",
                styles["BoxText"],
            )
        ],
    ]
    student_box = Table(student_box_data, colWidths=[18.0 * cm])
    student_box.setStyle(
        TableStyle([
            ("BOX", (0, 0), (-1, -1), 0.75, colors.HexColor("#475569")),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F1F5F9")),
            ("BACKGROUND", (0, 3), (-1, 3), colors.HexColor("#F1F5F9")),
            ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.HexColor("#CBD5E1")),
            ("LINEBELOW", (0, 3), (-1, 3), 0.5, colors.HexColor("#CBD5E1")),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ])
    )
    story.append(student_box)
    story.append(Spacer(1, 8))

    # 3. General Instructions
    num_sections = len(paper.sections) if paper.sections else 1
    story.append(Paragraph("General Instructions:", styles["InstructionsHeading"]))
    instructions = [
        "1. All questions are compulsory. Show necessary calculation steps where required.",
        f"2. The question paper contains {paper.num_questions} questions divided into {num_sections} section{'s' if num_sections > 1 else ''}.",
        f"3. Maximum marks: {paper.total_marks}. Marks allocated to each section and question are shown clearly.",
        "4. For multiple choice questions, write the alphabet of the correct option in the box [ ] provided.",
        "5. Write all answers and solutions neatly in the space provided below each question.",
    ]
    for instr in instructions:
        story.append(Paragraph(instr, styles["InstructionItem"]))
    story.append(Spacer(1, 10))

    # 4. Sections & Questions
    # If paper.sections is populated, iterate over structured sections
    sections_to_render = paper.sections
    if not sections_to_render:
        # Fallback grouping if paper.sections was empty
        sections_to_render = []
        mcq_qs = [q for q in paper.questions if q.question_type == QuestionType.MULTIPLE_CHOICE]
        sa_qs = [q for q in paper.questions if q.question_type == QuestionType.SHORT_ANSWER]
        la_qs = [q for q in paper.questions if q.question_type == QuestionType.LONG_ANSWER]

        sec_idx = 0
        names = ["A", "B", "C"]
        for q_group, q_type, title, inst in [
            (mcq_qs, QuestionType.MULTIPLE_CHOICE, "Multiple Choice Questions", "Choose the correct answer from the four options."),
            (sa_qs, QuestionType.SHORT_ANSWER, "Short Answer Questions", "Answer the following questions with necessary steps."),
            (la_qs, QuestionType.LONG_ANSWER, "Long Answer / Problem Solving Questions", "Solve the following problems showing detailed steps."),
        ]:
            if q_group:
                from app.schemas import PaperSection
                sec_id = names[sec_idx] if sec_idx < len(names) else f"Sec-{sec_idx+1}"
                sections_to_render.append(
                    PaperSection(
                        section_id=sec_id,
                        title=f"Section {sec_id}: {title}",
                        question_type=q_type,
                        num_questions=len(q_group),
                        section_marks=sum(q.marks for q in q_group),
                        instructions=inst,
                        questions=q_group,
                    )
                )
                sec_idx += 1

    # Question counter across all sections
    global_q_num = 1

    for sec in sections_to_render:
        # Section Header Banner
        sec_num_q = len(sec.questions)
        sec_marks = sec.section_marks

        # Marks summary text: e.g. "10 × 2 = 20 Marks"
        if sec.marks_per_question:
            marks_formula = f"{sec_num_q} × {sec.marks_per_question} = {sec_marks} Marks"
        else:
            base_m = sec_marks // sec_num_q if sec_num_q > 0 else 1
            if sec_marks % sec_num_q == 0:
                marks_formula = f"{sec_num_q} × {base_m} = {sec_marks} Marks"
            else:
                marks_formula = f"{sec_num_q} Questions | Total: {sec_marks} Marks"

        sec_header_data = [
            [
                Paragraph(f"<b>{_escape(sec.title).upper()}</b>", styles["SecTitleLeft"]),
                Paragraph(f"<b>{marks_formula}</b>", styles["SecTitleRight"]),
            ]
        ]
        sec_header_table = Table(sec_header_data, colWidths=[11.5 * cm, 6.5 * cm])
        sec_header_table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#E2E8F0")),
                ("BOX", (0, 0), (-1, -1), 0.75, colors.HexColor("#94A3B8")),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ])
        )

        sec_flowables = [Spacer(1, 8), sec_header_table]
        if sec.instructions:
            sec_flowables.append(Spacer(1, 3))
            sec_flowables.append(Paragraph(f"<i>{_escape(sec.instructions)}</i>", styles["SecInstruction"]))
        sec_flowables.append(Spacer(1, 6))

        story.append(KeepTogether(sec_flowables))

        # Render questions in this section
        for q in sec.questions:
            q_flowables = []
            safe_text = _escape(q.question_text)
            marks_badge = f"<font color='#475569'><b>[{q.marks} Mark{'s' if q.marks > 1 else ''}]</b></font>"
            q_line = f"<b>{global_q_num}.</b> {safe_text} &nbsp;&nbsp; {marks_badge}"
            q_flowables.append(Paragraph(q_line, styles["QuestionNumAndText"]))
            q_flowables.append(Spacer(1, 4))

            # Multiple Choice Question Options
            if q.question_type == QuestionType.MULTIPLE_CHOICE and q.options:
                opt_labels = ["A", "B", "C", "D", "E", "F"]
                # 2-column layout for options
                opt_cells = []
                for idx, opt in enumerate(q.options):
                    lbl = opt_labels[idx] if idx < len(opt_labels) else str(idx + 1)
                    opt_cells.append(Paragraph(f"<b>({lbl})</b> {_escape(opt)}", styles["OptionText"]))

                # Pair into rows of 2
                opt_table_data = []
                for i in range(0, len(opt_cells), 2):
                    row = [opt_cells[i]]
                    if i + 1 < len(opt_cells):
                        row.append(opt_cells[i + 1])
                    else:
                        row.append(Paragraph("", styles["OptionText"]))
                    opt_table_data.append(row)

                opt_table = Table(opt_table_data, colWidths=[9.0 * cm, 9.0 * cm])
                opt_table.setStyle(
                    TableStyle([
                        ("TOPPADDING", (0, 0), (-1, -1), 2),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                        ("LEFTPADDING", (0, 0), (-1, -1), 16),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                    ])
                )
                q_flowables.append(opt_table)
                q_flowables.append(Spacer(1, 4))

                # Student answer box / line
                if not include_answers:
                    ans_box_line = (
                        "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<b>Answer : &nbsp;[ &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ]</b>"
                        " &nbsp;&nbsp;&nbsp;&nbsp; __________________________________________________"
                    )
                    q_flowables.append(Paragraph(ans_box_line, styles["AnswerStudentLine"]))
                    q_flowables.append(Spacer(1, 8))

            # Non-MCQ student workspace
            elif not include_answers:
                # Ruled lines for student writing space
                if q.question_type == QuestionType.SHORT_ANSWER:
                    num_lines = 4
                else:  # LONG_ANSWER
                    num_lines = 7
                q_flowables.append(Spacer(1, 2))
                q_flowables.append(_make_ruled_lines(num_lines=num_lines, width_cm=18.0))
                q_flowables.append(Spacer(1, 8))

            # Teacher evaluation copy details (Answers, Solutions, Marking Guidance)
            if include_answers:
                teacher_box = []
                if q.question_type == QuestionType.MULTIPLE_CHOICE and q.correct_option:
                    teacher_box.append(Paragraph(
                        f"<b>Correct Option:</b> {_escape(q.correct_option)} &nbsp;|&nbsp; <b>Key Answer:</b> {_escape(q.answer)}",
                        styles["TeacherAnsHeader"],
                    ))
                else:
                    teacher_box.append(Paragraph(
                        f"<b>Answer:</b> {_escape(q.answer)}",
                        styles["TeacherAnsHeader"],
                    ))

                if q.solution:
                    teacher_box.append(Paragraph(
                        f"<b>Worked Solution &amp; Marking Steps:</b> {_escape(q.solution)}",
                        styles["TeacherAnsText"],
                    ))

                t_table = Table([[item] for item in teacher_box], colWidths=[17.6 * cm])
                t_table.setStyle(
                    TableStyle([
                        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#EFF6FF")),
                        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#BFDBFE")),
                        ("TOPPADDING", (0, 0), (-1, -1), 3),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                        ("LEFTPADDING", (0, 0), (-1, -1), 6),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                    ])
                )
                q_flowables.append(Spacer(1, 3))
                q_flowables.append(t_table)
                q_flowables.append(Spacer(1, 6))

            story.append(KeepTogether(q_flowables))
            global_q_num += 1

    # 5. Teacher Copy: Curriculum Traceability & Reference Sources
    if include_answers:
        story.append(PageBreak())
        story.append(Paragraph("Curriculum Alignment &amp; Textbook Source References", styles["SectionHeadingDoc"]))
        story.append(Paragraph(
            f"The questions above were generated from the official {format_grade_display(curriculum_class)} "
            "Mathematics curriculum textbook using FAISS vector retrieval.",
            styles["InstructionItem"],
        ))
        story.append(Spacer(1, 8))

        # Summary of question mapping
        mapping_data = [
            [
                Paragraph("<b>Q. No.</b>", styles["BoxHeader"]),
                Paragraph("<b>Section</b>", styles["BoxHeader"]),
                Paragraph("<b>Question Type</b>", styles["BoxHeader"]),
                Paragraph("<b>Marks</b>", styles["BoxHeader"]),
                Paragraph("<b>Chapter / Topic</b>", styles["BoxHeader"]),
                Paragraph("<b>Source Textbook Chunk</b>", styles["BoxHeader"]),
            ]
        ]

        q_idx = 1
        for sec in sections_to_render:
            for q in sec.questions:
                chunk_info = "Textbook Chunk"
                ch_display = q.chapter or q.topic or "Mathematics"
                if q.source_references:
                    ref = q.source_references[0]
                    chunk_idx = ref.chunk_index if ref.chunk_index is not None else 0
                    if ref.source_filename and ref.source_filename not in ("unknown", "Textbook", "Mathematics Curriculum"):
                        chunk_info = f"{ref.source_filename} (Chunk {chunk_idx})"
                        if ch_display in ("Mathematics Curriculum", "Mathematics", "General Math"):
                            ch_display = ref.source_filename
                    else:
                        chunk_info = f"Chunk {chunk_idx}"

                mapping_data.append([
                    Paragraph(f"Q{q_idx}", styles["BoxText"]),
                    Paragraph(_escape(sec.section_id), styles["BoxText"]),
                    Paragraph(_escape(q.question_type.value.replace("_", " ").title()), styles["BoxText"]),
                    Paragraph(f"{q.marks}", styles["BoxText"]),
                    Paragraph(_escape(ch_display), styles["BoxText"]),
                    Paragraph(_escape(chunk_info), styles["BoxText"]),
                ])
                q_idx += 1

        mapping_table = Table(
            mapping_data,
            colWidths=[1.5 * cm, 2.0 * cm, 4.0 * cm, 1.8 * cm, 4.7 * cm, 4.0 * cm],
        )
        mapping_table.setStyle(
            TableStyle([
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#64748B")),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F1F5F9")),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ])
        )
        story.append(mapping_table)

    # Build the document
    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes
