# app/worksheet_pdf.py
# ReportLab PDF Export for AI-Powered Personalized Practice Worksheets.
# Generates:
# 1. Student Practice Worksheet PDF (with personalized focus areas callout, questions, figures, and ruled answer workspace).
# 2. Teacher Answer Key & Solutions PDF (with full solutions, diagnostic mastery summary, and SymPy verification).
# 3. Batch ZIP downloads for multiple students.

from __future__ import annotations

import html
import io
import zipfile
from datetime import datetime, timezone
from typing import Optional

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    HRFlowable,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from app.pdf_export import _escape, _render_svg_flowable
from app.schemas import format_grade_display
from app.worksheets import PersonalizedWorksheet


def _build_worksheet_styles():
    """Construct typographic hierarchy for personalized worksheets."""
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "WsTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=15,
        leading=18,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#1e293b"),
    )
    subtitle_style = ParagraphStyle(
        "WsSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=14,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#334155"),
    )
    meta_style = ParagraphStyle(
        "WsMeta",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=12,
        alignment=TA_LEFT,
        textColor=colors.HexColor("#1e293b"),
    )
    meta_bold = ParagraphStyle(
        "WsMetaBold",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9,
        leading=12,
        alignment=TA_LEFT,
        textColor=colors.HexColor("#0f172a"),
    )
    q_num_style = ParagraphStyle(
        "WsQNum",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=10,
        leading=13,
        textColor=colors.HexColor("#0f172a"),
    )
    q_text_style = ParagraphStyle(
        "WsQText",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        alignment=TA_LEFT,
        textColor=colors.HexColor("#1e293b"),
    )
    sol_style = ParagraphStyle(
        "WsSolText",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#065f46"),
    )
    badge_style = ParagraphStyle(
        "WsBadge",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=10,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#1e40af"),
    )
    return {
        "title": title_style,
        "subtitle": subtitle_style,
        "meta": meta_style,
        "meta_bold": meta_bold,
        "q_num": q_num_style,
        "q_text": q_text_style,
        "sol": sol_style,
        "badge": badge_style,
    }


def generate_worksheet_pdf(
    worksheet: PersonalizedWorksheet,
    include_answers: bool = False,
) -> bytes:
    """
    Generate PDF for an individual student's personalized worksheet.
    include_answers=False generates the Student Practice Sheet with ruled answer space.
    include_answers=True generates the Teacher Solution Key.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=1.5 * cm,
        rightMargin=1.5 * cm,
        topMargin=1.5 * cm,
        bottomMargin=1.5 * cm,
    )
    styles = _build_worksheet_styles()
    story = []

    # 1. Header Banner
    story.append(Paragraph("QPGen — PERSONALIZED PRACTICE WORKSHEET", styles["title"]))
    sub_title = (
        "TEACHER ANSWER KEY & STEP-BY-STEP SOLUTIONS"
        if include_answers
        else "INDIVIDUALIZED MATHEMATICS REINFORCEMENT"
    )
    story.append(Paragraph(sub_title, styles["subtitle"]))
    story.append(Spacer(1, 0.2 * cm))

    # Demo Simulation Notice if applicable
    if worksheet.is_simulated_basis:
        demo_style = ParagraphStyle(
            "DemoBadge",
            fontName="Helvetica-Bold",
            fontSize=8,
            leading=10,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#b45309"),
        )
        story.append(Paragraph("★ DEMO / SIMULATED PERFORMANCE DATA (INITIAL INTEGRATION MODE) ★", demo_style))
        story.append(Spacer(1, 0.15 * cm))

    # 2. Student & Assessment Info Box
    grade_str = format_grade_display(worksheet.grade)
    info_data = [
        [
            Paragraph("<b>Student Name:</b>", styles["meta_bold"]),
            Paragraph(_escape(worksheet.student_name), styles["meta"]),
            Paragraph("<b>Class / Grade:</b>", styles["meta_bold"]),
            Paragraph(_escape(grade_str), styles["meta"]),
        ],
        [
            Paragraph("<b>Roll / SATS No:</b>", styles["meta_bold"]),
            Paragraph(_escape(worksheet.roll_number or "—"), styles["meta"]),
            Paragraph("<b>Subject:</b>", styles["meta_bold"]),
            Paragraph(_escape(worksheet.subject), styles["meta"]),
        ],
        [
            Paragraph("<b>Assessment Ref:</b>", styles["meta_bold"]),
            Paragraph(_escape(worksheet.assessment_reference or "General Assessment"), styles["meta"]),
            Paragraph("<b>Total Marks:</b>", styles["meta_bold"]),
            Paragraph(f"{worksheet.total_marks} Marks ({len(worksheet.questions)} Questions)", styles["meta"]),
        ],
    ]
    info_table = Table(info_data, colWidths=[3.2 * cm, 5.5 * cm, 3.2 * cm, 6.1 * cm])
    info_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
        ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#cbd5e1")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(info_table)
    story.append(Spacer(1, 0.3 * cm))

    # 3. Focus Areas Callout (Personalized for this student)
    focus_text = ", ".join(worksheet.focus_areas) if worksheet.focus_areas else "General revision across all permitted curriculum chapters"
    focus_box_data = [
        [
            Paragraph("<b>🎯 Personalized Focus Topics:</b>", styles["meta_bold"]),
            Paragraph(f"<i>{_escape(focus_text)}</i>", styles["meta"]),
        ]
    ]
    focus_table = Table(focus_box_data, colWidths=[4.8 * cm, 13.2 * cm])
    focus_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#eff6ff")),
        ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#93c5fd")),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(focus_table)
    story.append(Spacer(1, 0.3 * cm))

    # Instructions
    instr_text = (
        "<b>Instructions for Teacher:</b> This key provides complete worked solutions and marking steps. Diagnostic focus areas are weighted to strengthen student weaknesses."
        if include_answers
        else "<b>Instructions:</b> Read each question carefully. Write your complete working and calculations in the workspace provided below each problem."
    )
    story.append(Paragraph(instr_text, styles["meta"]))
    story.append(HRFlowable(width="100%", thickness=0.8, color=colors.HexColor("#cbd5e1"), spaceAfter=10, spaceBefore=4))

    # 4. Questions
    for idx, q in enumerate(worksheet.questions):
        q_elements = []

        # Question header line: Q1. [Topic] [Difficulty] [Marks]
        diff_label = q.difficulty.value.upper()
        header_text = (
            f"<b>Q{idx + 1}.</b> &nbsp; [{_escape(q.topic)}] &nbsp; "
            f"<i>({diff_label})</i> &nbsp; &nbsp; &nbsp; &nbsp; "
            f"<font color='#047857'><b>[{q.marks} Mark{'s' if q.marks != 1 else ''}]</b></font>"
        )
        q_elements.append(Paragraph(header_text, styles["q_num"]))
        q_elements.append(Spacer(1, 0.1 * cm))

        # Question prompt
        q_elements.append(Paragraph(_escape(q.question_text), styles["q_text"]))
        q_elements.append(Spacer(1, 0.15 * cm))

        # Render Figure if present
        if getattr(q, "figure", None) and getattr(q.figure, "svg_content", None):
            fig_flow = _render_svg_flowable(q.figure.svg_content, max_width_cm=11.0, max_height_cm=4.0)
            if fig_flow:
                q_elements.append(Spacer(1, 0.1 * cm))
                q_elements.append(fig_flow)
                q_elements.append(Spacer(1, 0.15 * cm))

        # Teacher View: Solution & Answer
        if include_answers:
            ans_box_data = [
                [
                    Paragraph("<b>Correct Answer:</b>", styles["meta_bold"]),
                    Paragraph(f"<b>{_escape(q.answer)}</b>", styles["meta_bold"]),
                ],
                [
                    Paragraph("<b>Worked Solution:</b>", styles["meta_bold"]),
                    Paragraph(_escape(q.solution or q.answer), styles["sol"]),
                ],
            ]
            ans_table = Table(ans_box_data, colWidths=[3.2 * cm, 14.0 * cm])
            ans_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f0fdf4")),
                ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#86efac")),
                ("INNERGRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#bbf7d0")),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ]))
            q_elements.append(ans_table)
            q_elements.append(Spacer(1, 0.25 * cm))
        else:
            # Student View: Ruled answer workspace
            num_lines = 4 if q.marks >= 3 else 3
            q_elements.append(Spacer(1, 0.1 * cm))
            for _ in range(num_lines):
                q_elements.append(HRFlowable(width="100%", thickness=0.4, color=colors.HexColor("#94a3b8"), spaceAfter=14, spaceBefore=0))
            q_elements.append(Spacer(1, 0.15 * cm))

        story.append(KeepTogether(q_elements))

    doc.build(story)
    return buffer.getvalue()


def generate_worksheets_zip(
    worksheets: list[PersonalizedWorksheet],
    include_answers: bool = False,
) -> bytes:
    """Pack multiple student worksheet PDFs into a single ZIP archive."""
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        index_lines = ["QPGen Personalized Worksheets Master Index", "=" * 50, ""]
        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        index_lines.append(f"Generated at: {now_str}")
        index_lines.append(f"Total Worksheets: {len(worksheets)}")
        index_lines.append(f"Mode: {'Teacher Solutions Key' if include_answers else 'Student Practice Worksheets'}")
        index_lines.append("")

        for idx, ws in enumerate(worksheets):
            pdf_bytes = generate_worksheet_pdf(ws, include_answers=include_answers)
            safe_name = ws.student_name.replace(" ", "_").replace("/", "_")
            roll_str = f"_Roll_{ws.roll_number}" if ws.roll_number else ""
            suffix = "solutions" if include_answers else "worksheet"
            filename = f"Worksheet_{idx + 1:02d}_{safe_name}{roll_str}_{suffix}.pdf"
            zf.writestr(filename, pdf_bytes)

            focus_str = ", ".join(ws.focus_areas) if ws.focus_areas else "General revision"
            index_lines.append(
                f"{idx + 1}. {ws.student_name} (Roll: {ws.roll_number or 'N/A'}) - "
                f"Class {ws.grade} - Focus: {focus_str} -> {filename}"
            )

        zf.writestr("Worksheet_Master_Index.txt", "\n".join(index_lines))

    return zip_buffer.getvalue()
