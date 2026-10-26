# app/model.py
# Sets up the Google Gemini LLM and embedding model via LangChain.
# These objects are reused across the application.

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from app.config import settings


def get_llm() -> ChatGoogleGenerativeAI:
    """
    Create and return the Gemini chat model.
    Raises a clear error if the API key is missing.
    """
    if not settings.GOOGLE_API_KEY:
        raise RuntimeError(
            "GOOGLE_API_KEY is not set. "
            "Copy .env.example to .env and add your Google AI Studio key. "
            "Get one at https://aistudio.google.com/apikey"
        )
    return ChatGoogleGenerativeAI(
        model=settings.GEMINI_MODEL,
        google_api_key=settings.GOOGLE_API_KEY,
        temperature=0.4,
        max_retries=settings.MAX_RETRIES,
    )


def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    """
    Create and return the Gemini embedding model.
    Used to convert text chunks into vectors for FAISS similarity search.
    """
    if not settings.GOOGLE_API_KEY:
        raise RuntimeError(
            "GOOGLE_API_KEY is not set. "
            "Copy .env.example to .env and add your Google AI Studio key."
        )
    return GoogleGenerativeAIEmbeddings(
        model=f"models/{settings.GEMINI_EMBEDDING_MODEL}",
        google_api_key=settings.GOOGLE_API_KEY,
    )
