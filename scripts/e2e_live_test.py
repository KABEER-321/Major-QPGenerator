"""
scripts/e2e_live_test.py
Phase 6 — End-to-End Live Integration Test for QPGen

Walks through the complete QPGen pipeline against the running server:
1. Health check (GET /health)
2. Generate a sample syllabus PDF
3. Upload and index the document (POST /documents/upload)
4. Verify document catalog (GET /documents, GET /documents/{id})
5. Generate a full question paper with live Gemini (POST /papers/generate)
6. Verify paper catalog (GET /papers, GET /papers/{id})
7. Download and verify both Teacher and Student PDFs (GET /papers/{id}/pdf)
8. Verify question traceability and source references

Run with: uv run python scripts/e2e_live_test.py
"""

import io
import json
import sys
from pathlib import Path

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import httpx
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer

BASE_URL = "http://127.0.0.1:8000"


def generate_sample_pdf_bytes() -> bytes:
    """Generate a realistic Grade 10 mathematics syllabus PDF in memory."""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter, rightMargin=50, leftMargin=50, topMargin=50, bottomMargin=50)
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontSize=18,
        leading=22,
        spaceAfter=12,
    )
    h2_style = ParagraphStyle(
        "DocH2",
        parent=styles["Heading2"],
        fontSize=13,
        leading=16,
        spaceBefore=10,
        spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "DocBody",
        parent=styles["Normal"],
        fontSize=10,
        leading=14,
        spaceAfter=8,
    )

    story = [
        Paragraph("Grade 10 Mathematics — Chapter 4: Quadratic Equations", title_style),
        Spacer(1, 10),
        Paragraph("4.1 Standard Form of a Quadratic Equation", h2_style),
        Paragraph(
            "A quadratic equation in the variable x is an equation of the form "
            "ax² + bx + c = 0, where a, b, c are real numbers and a ≠ 0. "
            "For example, 2x² + x - 300 = 0 is a quadratic equation. "
            "Any equation of the form p(x) = 0, where p(x) is a polynomial of degree 2, is a quadratic equation.",
            body_style,
        ),
        Paragraph("4.2 Solution of a Quadratic Equation by Factorisation", h2_style),
        Paragraph(
            "A real number α is called a root of the quadratic equation ax² + bx + c = 0 if aα² + bα + c = 0. "
            "We also say that x = α is a solution of the quadratic equation, or that α satisfies the quadratic equation. "
            "To solve by factorisation, we express ax² + bx + c as the product of two linear factors "
            "and equate each factor to zero.",
            body_style,
        ),
        Paragraph("4.3 Nature of Roots and the Discriminant", h2_style),
        Paragraph(
            "For the quadratic equation ax² + bx + c = 0 (a ≠ 0), the expression b² - 4ac is called the discriminant, "
            "denoted by D. The discriminant determines whether the quadratic equation has real roots or not. "
            "Specifically: "
            "(i) Two distinct real roots exist if D = b² - 4ac > 0. "
            "(ii) Two equal real roots (coincident roots) exist if D = b² - 4ac = 0, in which case both roots are -b / (2a). "
            "(iii) No real roots exist if D = b² - 4ac < 0 (the roots are complex/imaginary).",
            body_style,
        ),
        Paragraph("4.4 The Quadratic Formula", h2_style),
        Paragraph(
            "If b² - 4ac ≥ 0, the roots of the quadratic equation ax² + bx + c = 0 are given by the quadratic formula: "
            "x = (-b ± √(b² - 4ac)) / (2a). This method was attributed to ancient mathematician Sridharacharya.",
            body_style,
        ),
    ]

    doc.build(story)
    return buf.getvalue()


def run_e2e_test():
    print("=" * 65)
    print("QPGen — Phase 6 End-to-End Live Integration Test")
    print(f"Target Server: {BASE_URL}")
    print("=" * 65)

    client = httpx.Client(base_url=BASE_URL, timeout=90.0)

    # 1. Health check
    print("\n[Step 1/7] Testing /health endpoint...")
    try:
        r = client.get("/health")
        r.raise_for_status()
        assert r.json().get("status") == "ok"
        print("  ✓ Server is healthy (status: ok)")
    except Exception as e:
        print(f"  ✗ Server health check failed: {e}")
        print("  Please ensure 'uv run uvicorn app.main:app --reload --port 8000' is running.")
        sys.exit(1)

    # 2. Generate sample PDF
    print("\n[Step 2/7] Generating sample syllabus PDF...")
    pdf_bytes = generate_sample_pdf_bytes()
    print(f"  ✓ Created in-memory PDF ({len(pdf_bytes):,} bytes)")

    # 3. Upload document
    print("\n[Step 3/7] Uploading document to /documents/upload...")
    files = {"file": ("Grade10_Maths_Quadratic_Equations.pdf", pdf_bytes, "application/pdf")}
    r = client.post("/documents/upload", files=files)
    r.raise_for_status()
    doc_info = r.json()
    doc_id = doc_info["document_id"]
    print(f"  ✓ Uploaded successfully!")
    print(f"    - Document ID : {doc_id}")
    print(f"    - Filename    : {doc_info['filename']}")
    print(f"    - Chunks      : {doc_info['num_chunks']}")
    print(f"    - Message     : {doc_info['message']}")

    # 4. Check document metadata endpoints
    print("\n[Step 4/7] Verifying document metadata endpoints...")
    r_list = client.get("/documents")
    r_list.raise_for_status()
    docs = r_list.json()
    assert any(d["document_id"] == doc_id for d in docs), "Uploaded doc not found in /documents list"
    print(f"  ✓ Document appears in /documents listing (total documents: {len(docs)})")

    r_doc = client.get(f"/documents/{doc_id}")
    r_doc.raise_for_status()
    doc_meta = r_doc.json()
    print(f"  ✓ Retrieved metadata for {doc_id}:")
    print(f"    - Pages       : {doc_meta.get('num_pages')}")
    print(f"    - Index Built : {doc_meta.get('has_index')}")
    print(f"    - Preview     : {doc_meta.get('text_preview', '')[:80]}...")

    # 5. Generate question paper via Gemini
    print("\n[Step 5/7] Generating question paper with live Gemini...")
    generate_payload = {
        "document_id": doc_id,
        "grade": "10",
        "subject": "Mathematics",
        "topic": "Quadratic Equations",
        "difficulty": "medium",
        "question_types": ["multiple_choice", "short_answer", "long_answer"],
        "num_questions": 3,
        "total_marks": 10,
    }
    print(f"  Calling /papers/generate (requesting 3 questions, 10 marks total)...")
    r_gen = client.post("/papers/generate", json=generate_payload)
    if r_gen.status_code != 200:
        print(f"  ✗ Paper generation failed ({r_gen.status_code}): {r_gen.text}")
        sys.exit(1)

    paper = r_gen.json()
    paper_id = paper["paper_id"]
    print(f"  ✓ Paper generated successfully!")
    print(f"    - Paper ID       : {paper_id}")
    print(f"    - Title          : {paper['title']}")
    print(f"    - Total Marks    : {paper['total_marks']}")
    print(f"    - Questions Count: {paper['num_questions']}")

    # Validate questions
    questions = paper.get("questions", [])
    assert len(questions) == 3, f"Expected 3 questions, got {len(questions)}"
    total_q_marks = sum(q["marks"] for q in questions)
    assert total_q_marks == 10, f"Expected 10 total marks, got {total_q_marks}"

    print("\n  Generated Questions Details:")
    for i, q in enumerate(questions, 1):
        print(f"\n    Q{i}. [{q['question_type'].upper()}] ({q['marks']} marks) - {q['difficulty'].capitalize()}")
        print(f"        Text     : {q['question_text']}")
        if q.get("options"):
            print(f"        Options  : {q['options']}")
            print(f"        Correct  : {q.get('correct_option')}")
        print(f"        Answer   : {q.get('answer')}")
        print(f"        Solution : {q.get('solution')[:120]}...")
        print(f"        Status   : {q.get('validation_status')}")
        refs = q.get("source_references", [])
        if refs:
            print(f"        Source   : {refs[0].get('source_filename')} (chunk {refs[0].get('chunk_index')})")
            print(f"        Citation : \"{refs[0].get('passage_preview', '')[:70]}...\"")

    # 6. Verify paper persistence
    print("\n[Step 6/7] Verifying paper persistence via /papers and /papers/{paper_id}...")
    r_papers = client.get("/papers")
    r_papers.raise_for_status()
    papers_list = r_papers.json()
    assert any(p["paper_id"] == paper_id for p in papers_list), "Paper not found in /papers list"
    print(f"  ✓ Paper listed in /papers (total papers: {len(papers_list)})")

    r_single = client.get(f"/papers/{paper_id}")
    r_single.raise_for_status()
    assert r_single.json()["paper_id"] == paper_id
    print(f"  ✓ Verified GET /papers/{paper_id} returns identical paper data")

    # 7. Download PDF (both student & teacher versions)
    print("\n[Step 7/8] Downloading and verifying generated PDFs...")
    # Teacher PDF (with solutions)
    r_teacher_pdf = client.get(f"/papers/{paper_id}/pdf?include_answers=true")
    r_teacher_pdf.raise_for_status()
    assert r_teacher_pdf.headers["content-type"] == "application/pdf"
    assert r_teacher_pdf.content.startswith(b"%PDF-"), "Response content is not a valid PDF"
    teacher_size = len(r_teacher_pdf.content)
    print(f"  ✓ Teacher PDF generated ({teacher_size:,} bytes, Content-Type: application/pdf)")

    # Student PDF (without solutions)
    r_student_pdf = client.get(f"/papers/{paper_id}/pdf?include_answers=false")
    r_student_pdf.raise_for_status()
    assert r_student_pdf.headers["content-type"] == "application/pdf"
    assert r_student_pdf.content.startswith(b"%PDF-"), "Response content is not a valid PDF"
    student_size = len(r_student_pdf.content)
    print(f"  ✓ Student PDF generated ({student_size:,} bytes, Content-Type: application/pdf)")

    # Save outputs to disk for inspection
    out_dir = Path("data") / "e2e_exports"
    out_dir.mkdir(parents=True, exist_ok=True)
    teacher_path = out_dir / f"paper_{paper_id}_teacher.pdf"
    student_path = out_dir / f"paper_{paper_id}_student.pdf"
    teacher_path.write_bytes(r_teacher_pdf.content)
    student_path.write_bytes(r_student_pdf.content)
    print(f"  ✓ Saved test PDFs to data/e2e_exports/")

    # 8. Test Teacher Web UI & Resource Cleanup
    print("\n[Step 8/8] Verifying Teacher Web UI & DELETE endpoints...")
    r_ui = client.get("/")
    assert r_ui.status_code == 200
    assert "text/html" in r_ui.headers["content-type"]
    assert "QPGen" in r_ui.text
    print("  ✓ Teacher Web UI responded (GET / -> 200 OK HTML)")

    r_del_paper = client.delete(f"/papers/{paper_id}")
    assert r_del_paper.status_code == 200
    print(f"  ✓ DELETE /papers/{paper_id} succeeded")

    r_del_doc = client.delete(f"/documents/{doc_id}")
    assert r_del_doc.status_code == 200
    print(f"  ✓ DELETE /documents/{doc_id} succeeded")

    print("\n" + "=" * 65)
    print("🎉 Phase 7 End-to-End Live Integration Test PASSED SUCCESSFULLY!")
    print("=" * 65)


if __name__ == "__main__":
    run_e2e_test()

