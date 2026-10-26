# app/config.py
# Loads configuration from environment variables (.env file).
# All settings are centralized here so no secrets are hard-coded anywhere else.

import os
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import ConfigDict
from dotenv import load_dotenv

# Load .env from the project root (one level above app/)
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Google Gemini API key — required, no default so it fails loudly if missing
    GOOGLE_API_KEY: str = ""

    # Which Gemini model to use for question generation
    GEMINI_MODEL: str = "gemini-2.5-flash"

    # Which Gemini model to use for text embeddings
    GEMINI_EMBEDDING_MODEL: str = "gemini-embedding-001"

    # Maximum upload file size in bytes (default 20 MB)
    MAX_UPLOAD_SIZE_BYTES: int = 20 * 1024 * 1024

    # Chunk size for splitting document text (in characters)
    CHUNK_SIZE: int = 1000

    # Overlap between consecutive chunks (in characters)
    CHUNK_OVERLAP: int = 200

    # Number of relevant chunks to retrieve for question generation
    RETRIEVAL_TOP_K: int = 6

    # Maximum retries for model calls
    MAX_RETRIES: int = 2

    # Base directory for all data files (uploads, indexes, papers)
    DATA_DIR: Path = Path(__file__).resolve().parent.parent / "data"

    @property
    def curriculum_dir(self) -> Path:
        """Directory for permanently stored curriculum textbooks."""
        d = self.DATA_DIR / "curriculum"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def uploads_dir(self) -> Path:
        """Directory for uploaded syllabus/textbook files."""
        d = self.DATA_DIR / "uploads"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def indexes_dir(self) -> Path:
        """Directory for saved FAISS indexes."""
        d = self.DATA_DIR / "indexes"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def papers_dir(self) -> Path:
        """Directory for saved paper JSON files and PDFs."""
        d = self.DATA_DIR / "papers"
        d.mkdir(parents=True, exist_ok=True)
        return d

    model_config = ConfigDict(env_file=".env", extra="ignore")


# Singleton settings instance used throughout the application
settings = Settings()
