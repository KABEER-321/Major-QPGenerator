# QPGen — AI-Based Curriculum-Aligned Mathematics Question Paper Generator

QPGen uses Google Gemini and retrieval-augmented generation (RAG) to create
mathematics question papers grounded in your uploaded syllabus or textbook.

## What It Does

1. You **upload** a syllabus or textbook chapter (PDF or DOCX).
2. The app extracts text and builds a searchable vector index (FAISS).
3. You specify grade, topic, difficulty, question types, count, and marks.
4. The app retrieves the most relevant passages from your document.
5. Google Gemini generates curriculum-aligned questions based on those passages.
6. Questions are validated with Pydantic, assembled into a paper, and saved as JSON.
7. You can view the paper and **download it as a PDF** (with or without answers).

## Project Structure

```
QPGen-Model/
├── app/
│   ├── main.py              # FastAPI routes and application entry point
│   ├── config.py            # Environment variables and file paths
│   ├── model.py             # Gemini LLM and embedding model setup
│   ├── schemas.py           # Pydantic models (Blueprints, Questions, Papers, etc.)
│   ├── documents.py         # PDF/DOCX text extraction and chunking
│   ├── documents_store.py   # Document metadata persistence (save/load/list/delete)
│   ├── retrieval.py         # FAISS index creation, saving, and retrieval
│   ├── generator.py         # Candidate pool generation and Gemini prompting
│   ├── duplicates.py        # Text normalization, duplicate detection, and version selection
│   ├── math_verify.py       # SymPy mathematical expression checking
│   ├── paper.py             # Paper assembly, JSON persistence, and deletion
│   ├── pdf_export.py        # PDF generation with ReportLab
│   └── templates/
│       └── index.html       # Interactive Teacher Portal single-page web UI
├── data/
│   ├── uploads/             # Uploaded syllabus/textbook files + metadata JSON
│   ├── indexes/             # Saved FAISS vector indexes
│   └── papers/              # Saved paper JSON files
├── tests/
│   ├── test_schemas.py      # Blueprint and Pydantic schema validation tests
│   ├── test_duplicates.py   # Duplicate detection and candidate pool selection tests
│   ├── test_paper.py        # Paper assembly, sectioning, PDF export, and persistence tests
│   ├── test_documents_store.py  # Document metadata persistence tests
│   ├── test_math_verify.py  # SymPy mathematical verification tests
│   └── test_api.py          # FastAPI endpoint (single and batch) tests
├── scripts/
│   ├── verify_gemini.py     # Manual smoke test for Gemini API key
│   ├── e2e_live_test.py     # End-to-end live integration test (single paper)
│   └── e2e_batch_test.py    # End-to-end live integration test (batch multi-version)
├── .env.example             # Template for environment variables
├── .gitignore
├── pyproject.toml           # Project config and dependencies
├── requirements.txt         # Compatibility list (mirrors pyproject.toml)
└── README.md                # This file
```

### What each module does

| Module | Responsibility |
|--------|---------------|
| `config.py` | Loads API key, model names, and paths from `.env` using Pydantic Settings |
| `model.py` | Creates LangChain wrappers for the Gemini LLM and embedding model |
| `schemas.py` | Pydantic models: `SectionBlueprint`, `ExamBlueprint`, `Question`, `Paper`, `BatchPaperResponse` |
| `documents.py` | Extracts text from PDF (PyMuPDF) and DOCX (python-docx), splits into chunks |
| `documents_store.py` | Saves/loads/lists/deletes document metadata and assets for traceability |
| `retrieval.py` | Builds FAISS indexes from text chunks, retrieves relevant passages by topic |
| `generator.py` | Generates candidate questions with Gemini, enriches metadata, rejects math discrepancies |
| `duplicates.py` | Normalizes question text, calculates similarity, detects duplicates, selects version questions without replacement |
| `math_verify.py` | Evaluates algebraic roots, discriminants, and arithmetic with SymPy |
| `paper.py` | Assembles questions into versioned papers, enforces blueprint consistency, safe atomic JSON persistence |
| `pdf_export.py` | Renders a paper as a formatted PDF with ReportLab (teacher & student views, section support) |
| `main.py` | FastAPI application with single (`/papers/generate`) and batch (`/papers/generate-batch`) endpoints |

## Prerequisites

- **Python 3.12+**
- **uv** (Python package manager) — install from https://docs.astral.sh/uv/
- **Google AI Studio API key** — get one free at https://aistudio.google.com/apikey

## Setup & Installation (Windows PowerShell)

```powershell
# 1. Navigate to the project
cd QPGen-Model

# 2. Install uv if you don't have it
pip install uv

# 3. Create a virtual environment and install all dependencies
uv sync --extra dev

# 4. Configure your API key
Copy-Item .env.example .env
# Open .env and replace 'replace_with_your_google_ai_studio_key' with your actual key
notepad .env
```

### Cross-Platform Alternative (Linux/Mac)

```bash
cd QPGen-Model
pip install uv
uv sync --extra dev
cp .env.example .env
# Edit .env with your API key
nano .env
```

## Configuring the Gemini API Key

1. Go to https://aistudio.google.com/apikey and create an API key.
2. Open the `.env` file in the project root.
3. Replace `replace_with_your_google_ai_studio_key` with your actual key.
4. **Never share this key** or commit it to version control.

The `.env` file is listed in `.gitignore` so it won't be tracked by git.
The `.env.example` file (without a real key) is safe to commit.

To verify your key and Gemini connection work:

```powershell
uv run python scripts/verify_gemini.py
```

## Running the API

```powershell
uv run uvicorn app.main:app --reload --port 8000
```

The API will be running at `http://127.0.0.1:8000`.

### Open Interactive API Docs

Visit **http://127.0.0.1:8000/** in your browser for the interactive Teacher Portal,
or **http://127.0.0.1:8000/docs** for FastAPI's interactive Swagger UI documentation.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Interactive Teacher Portal web UI |
| `GET` | `/health` | Basic health check (`{"status":"ok"}`) |
| `POST` | `/documents/upload` | Upload a PDF/DOCX, extract text, build FAISS index |
| `POST` | `/documents/text` | Submit syllabus/textbook content directly as plain text |
| `GET` | `/documents` | List all uploaded documents |
| `GET` | `/documents/{document_id}` | Retrieve metadata for one uploaded document |
| `DELETE` | `/documents/{document_id}` | Delete a document, its record, and its FAISS index |
| `POST` | `/papers/generate` | Generate a question paper from an uploaded document |
| `GET` | `/papers` | List all saved papers |
| `GET` | `/papers/{paper_id}` | Retrieve a saved paper's full JSON |
| `GET` | `/papers/{paper_id}/pdf` | Download a paper as a PDF (teacher or student version) |
| `DELETE` | `/papers/{paper_id}` | Delete a saved question paper |

## How to Use

### Step 1: Upload a File or Enter Syllabus Text

You can provide reference syllabus content in two ways:

1. **Upload File (`POST /documents/upload`)**: Upload a PDF or DOCX syllabus/textbook (max 20 MB).
2. **Submit Plain Text (`POST /documents/text`)**: Paste textbook chapters, notes, or curriculum definitions directly:
   ```json
   {
     "title": "Grade 10 - Quadratic Equations",
     "content": "A quadratic equation in the variable x is an equation of the form ax^2 + bx + c = 0..."
   }
   ```

Both methods automatically extract/split the content into overlapping chunks, create embeddings with `gemini-embedding-001`, build a FAISS index, and return a `document_id`.

### Step 2: Browse Uploaded Documents (optional)

```
GET /documents                    # list all uploaded documents
GET /documents/{document_id}     # view details of one document
```

### Step 3: Generate a Question Paper

```
POST /papers/generate
```

Send a JSON body:
```json
{
  "document_id": "your_document_id_here",
  "grade": "10",
  "subject": "Mathematics",
  "topic": "Quadratic Equations",
  "difficulty": "medium",
  "question_types": ["short_answer", "long_answer", "multiple_choice"],
  "num_questions": 10,
  "total_marks": 50
}
```

The API retrieves relevant passages from the uploaded document, asks Gemini
to generate questions grounded in those passages, validates the output with
Pydantic, and saves the paper as JSON.

### Step 4: View or List Papers

```
GET /papers                     # list all saved papers
GET /papers/{paper_id}          # view a specific paper's JSON
```

### Step 5: Download as PDF

```
GET /papers/{paper_id}/pdf                        # teacher version (with answers)
GET /papers/{paper_id}/pdf?include_answers=false   # student version (no answers)
```

The PDF includes paper metadata, numbered questions with marks, MCQ options,
page numbers, and a separate answer key with solutions section.

## Where Data Is Stored

All data is stored locally as files — no database is used.

| Data | Location | Format |
|------|----------|--------|
| Uploaded documents | `data/uploads/` | Original PDF/DOCX + metadata JSON |
| FAISS vector indexes | `data/indexes/` | Binary FAISS index files |
| Saved papers | `data/papers/` | JSON files |

These directories are created automatically on first use and are listed
in `.gitignore` to keep generated data out of version control.

## Running Tests

Tests are designed to run **without** a real Gemini API key. All model and
embedding calls are mocked.

```powershell
# Run all tests with verbose output
uv run pytest -v

# Run a specific test file
uv run pytest tests/test_schemas.py -v

# Run tests quietly (just pass/fail counts)
uv run pytest -q
```

Current test count: **140 tests**, covering:
- Blueprint validation: Section blueprints, exam blueprints, marks distribution, total marks & question count sums
- Multi-version generation: Candidate pool selection without replacement, cross-version uniqueness, shortage errors
- Duplicate & near-duplicate detection: Normalized text, token Jaccard, character 3-grams, MCQ option matching
- Pydantic schema validation: questions, papers, sections, documents, batch response, traceability
- Paper assembly, sectioning, JSON persistence, and legacy backward-compatible loading
- Document metadata persistence and asset cleanup
- SymPy automated mathematical verification and candidate discrepancy rejection
- PDF export: formatted teacher & student views with section headers and version metadata
- All FastAPI endpoints (single generation, batch multi-version generation, documents, papers, PDF download, deletions)
- Edge cases: candidate shortage, invalid blueprints, malformed JSON, path traversal, oversized files

## Configuration

Environment variables (set in `.env`):

| Variable | Default | Description |
|----------|---------|-------------|
| `GOOGLE_API_KEY` | *(required)* | Your Google AI Studio API key |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Gemini model for question generation |
| `GEMINI_EMBEDDING_MODEL` | `gemini-embedding-001` | Gemini model for text embeddings |

Additional settings are configurable in `app/config.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `MAX_UPLOAD_SIZE_BYTES` | 20 MB | Maximum upload file size |
| `CHUNK_SIZE` | 1000 | Characters per text chunk |
| `CHUNK_OVERLAP` | 200 | Overlap between consecutive chunks |
| `RETRIEVAL_TOP_K` | 6 | Number of relevant chunks to retrieve |
| `MAX_RETRIES` | 2 | Retry attempts for Gemini calls |

## Common Errors & Fixes

| Error | Fix |
|-------|-----|
| `GOOGLE_API_KEY is not set` | Copy `.env.example` to `.env` and add your key |
| `Unsupported file type` | Only `.pdf` and `.docx` files are supported |
| `File too large` | Maximum upload size is 20 MB by default |
| `No readable text could be extracted` | Scanned PDFs need OCR, which is not supported yet |
| `No FAISS index found for document` | Upload and process the document first via `/documents/upload` |
| `Failed to create embeddings` | Check your API key and internet connection |
| Model returns invalid JSON | The system retries automatically (up to 2 retries); try a more specific topic |
| `ModuleNotFoundError` | Run `uv sync --extra dev` to install all dependencies |

## Current Limitations & Design Boundaries

- **No authentication** — the API is open. Only use in local or trusted private environments.
  Do not deploy this as a public production service without adding authentication.
- **No OCR** — scanned PDFs with image-only pages cannot be processed.
- **Mathematical validation boundaries** — SymPy provides automated, optional verification
  for supported algebraic equations, quadratic discriminants, and arithmetic expressions.
  As stated in the specification, it does not claim to validate every question, geometry proof,
  or complex textual derivation. Always review generated papers before classroom use.
- **No student accounts or grading** — this is an exam generation tool only.
- **Web Interface** — An interactive Teacher Portal is available at `http://127.0.0.1:8000/`,
  and full API docs are at `/docs`.
- **Local file storage only** — JSON files in `data/` are used for simple, transparent persistence.

