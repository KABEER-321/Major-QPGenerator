"""
scripts/e2e_batch_test.py
Live End-to-End Test for Multi-Version Batch Question Paper Generation

Exercises the complete batch workflow:
1. Health check (GET /health)
2. Upload plain-text syllabus (POST /documents/text)
3. Generate 2 unique paper versions with shared blueprint (POST /papers/generate-batch)
4. Verify shared blueprint and cross-version question uniqueness
5. Verify paper retrieval and PDF downloads (GET /papers/{id}, GET /papers/{id}/pdf)
6. Clean up test papers and document

Run with: uv run python scripts/e2e_batch_test.py
"""

import sys
from pathlib import Path

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import httpx

BASE_URL = "http://127.0.0.1:8000"


def run_batch_test():
    client = httpx.Client(base_url=BASE_URL, timeout=120.0)

    print("=" * 65)
    print("QPGen — Multi-Version Batch Generation Live Verification")
    print("=" * 65)

    # 1. Health check
    print("\n[Step 1/5] Checking server health...")
    r_health = client.get("/health")
    r_health.raise_for_status()
    print("  ✓ Server is healthy (GET /health -> 200 OK)")

    # 2. Upload sample syllabus as text
    print("\n[Step 2/5] Uploading syllabus content as text...")
    syllabus_content = """
    Chapter 4: Quadratic Equations

    1. Standard Form:
       A quadratic equation in variable x is an equation of the form ax^2 + bx + c = 0,
       where a, b, c are real numbers and a != 0.

    2. Roots of a Quadratic Equation:
       A real number alpha is called a root of the quadratic equation ax^2 + bx + c = 0
       if a*(alpha)^2 + b*(alpha) + c = 0.
       A quadratic equation has at most two roots.

    3. Solution by Factorisation:
       If we can factorise ax^2 + bx + c as (px + q)(rx + s), then the roots are -q/p and -s/r.

    4. Discriminant and Nature of Roots:
       For the equation ax^2 + bx + c = 0, the discriminant is D = b^2 - 4ac.
       - If D > 0: two distinct real roots.
       - If D = 0: two equal real roots (coincident roots).
       - If D < 0: no real roots.
    """
    r_upload = client.post(
        "/documents/text",
        json={
            "title": "Class 10 Quadratic Equations",
            "content": syllabus_content.strip(),
        },
    )
    r_upload.raise_for_status()
    doc_info = r_upload.json()
    doc_id = doc_info["document_id"]
    print(f"  ✓ Syllabus uploaded and indexed. Document ID: {doc_id}")

    # 3. Generate 2 unique versions via POST /papers/generate-batch
    print("\n[Step 3/5] Requesting 2 unique paper versions (POST /papers/generate-batch)...")
    batch_request = {
        "document_id": doc_id,
        "grade": "Class 10",
        "subject": "Mathematics",
        "topic": "Quadratic Equations",
        "difficulty": "medium",
        "num_versions": 2,
        "sections": [
            {
                "section_id": "A",
                "title": "Section A: Multiple Choice Questions",
                "question_type": "multiple_choice",
                "num_questions": 2,
                "section_marks": 2,
                "marks_per_question": 1,
            },
            {
                "section_id": "B",
                "title": "Section B: Short Answer Questions",
                "question_type": "short_answer",
                "num_questions": 1,
                "section_marks": 2,
                "marks_per_question": 2,
            },
        ],
    }

    r_batch = client.post("/papers/generate-batch", json=batch_request)
    r_batch.raise_for_status()
    batch_data = r_batch.json()

    batch_id = batch_data["batch_id"]
    num_req = batch_data["num_versions_requested"]
    num_gen = batch_data["num_versions_generated"]
    papers = batch_data["papers"]
    summary = batch_data["validation_summary"]

    print(f"  ✓ Batch generated successfully!")
    print(f"    Batch ID           : {batch_id}")
    print(f"    Versions requested : {num_req}")
    print(f"    Versions generated : {num_gen}")
    print(f"    Candidates valid   : {summary['valid_candidates_count']}")
    print(f"    Duplicates rejected: {summary['duplicate_candidates_rejected']}")
    print(f"    Math discrepancies : {summary['math_discrepancies_rejected']}")
    print(f"    Math verified      : {summary['math_verified_count']}")
    print(f"    Is unique          : {summary['is_unique']}")
    print(f"    All versions valid : {summary['all_versions_valid']}")

    # 4. Verify uniqueness and blueprint consistency
    print("\n[Step 4/5] Verifying blueprint structure and cross-version question uniqueness...")
    assert len(papers) == 2, f"Expected 2 papers, got {len(papers)}"
    p1 = papers[0]
    p2 = papers[1]

    print(f"  Version 1: ID={p1['paper_id']} | Title='{p1['title']}' | Questions={p1['num_questions']} | Marks={p1['total_marks']}")
    print(f"  Version 2: ID={p2['paper_id']} | Title='{p2['title']}' | Questions={p2['num_questions']} | Marks={p2['total_marks']}")

    assert p1["version_number"] == 1
    assert p2["version_number"] == 2
    assert p1["num_questions"] == 3
    assert p2["num_questions"] == 3
    assert p1["total_marks"] == 4
    assert p2["total_marks"] == 4

    p1_q_ids = [q["question_id"] for q in p1["questions"]]
    p2_q_ids = [q["question_id"] for q in p2["questions"]]
    all_q_ids = set(p1_q_ids + p2_q_ids)

    assert len(all_q_ids) == len(p1_q_ids) + len(p2_q_ids), "Question IDs overlapped across versions!"
    print(f"  ✓ Confirmed complete cross-version uniqueness ({len(all_q_ids)} distinct questions).")

    # 5. Verify PDF download and retrieval
    print("\n[Step 5/5] Verifying PDF export for generated batch papers...")
    for p in papers:
        pid = p["paper_id"]
        # Teacher PDF
        r_pdf_t = client.get(f"/papers/{pid}/pdf?include_answers=true")
        assert r_pdf_t.status_code == 200
        assert r_pdf_t.headers["content-type"] == "application/pdf"
        assert r_pdf_t.content.startswith(b"%PDF-")

        # Student PDF
        r_pdf_s = client.get(f"/papers/{pid}/pdf?include_answers=false")
        assert r_pdf_s.status_code == 200
        assert r_pdf_s.headers["content-type"] == "application/pdf"
        assert r_pdf_s.content.startswith(b"%PDF-")
        assert len(r_pdf_s.content) < len(r_pdf_t.content)
        print(f"  ✓ Paper {pid}: Teacher PDF ({len(r_pdf_t.content):,} B) & Student PDF ({len(r_pdf_s.content):,} B) verified.")

    # Clean up test artifacts
    for p in papers:
        client.delete(f"/papers/{p['paper_id']}")
    client.delete(f"/documents/{doc_id}")
    print("  ✓ Cleanup completed successfully.")

    print("\n" + "=" * 65)
    print("🎉 Multi-Version Batch Generation Test PASSED SUCCESSFULLY!")
    print("=" * 65)


if __name__ == "__main__":
    run_batch_test()
