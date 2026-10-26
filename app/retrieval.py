# app/retrieval.py
# Handles embeddings, FAISS index creation, saving, loading, and retrieval.
#
# How it works:
# 1. Text chunks from an uploaded document are converted into numerical vectors
#    (embeddings) using the configured Google embedding model.
# 2. These vectors are stored in a FAISS index, which allows fast similarity search.
# 3. When the user requests questions on a topic, the topic text is also embedded
#    and FAISS finds the most similar document chunks.
# 4. Those chunks become the "retrieved passages" fed to Gemini as reference material.
#
# FAISS performs vector similarity search — it finds text that is semantically
# close, not an exact keyword match. The embedding model is what converts
# human-readable text into vectors that FAISS can compare.

from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from app.config import settings
from app.model import get_embeddings


def build_faiss_index(
    chunks: list[dict],
    document_id: str,
) -> FAISS:
    """
    Build a FAISS index from text chunks and save it to disk.

    Args:
        chunks: List of dicts with 'content' and 'metadata' keys.
        document_id: Unique ID used as the folder name for saving.

    Returns:
        The FAISS vector store object.
    """
    # Convert chunk dicts into LangChain Document objects
    documents = [
        Document(page_content=c["content"], metadata=c["metadata"])
        for c in chunks
    ]

    embeddings = get_embeddings()
    # FAISS.from_documents embeds all texts and builds the index
    vectorstore = FAISS.from_documents(documents, embeddings)

    # Save the index so it can be reloaded later without re-embedding
    save_path = settings.indexes_dir / document_id
    vectorstore.save_local(str(save_path))
    return vectorstore


def load_faiss_index(document_id: str) -> FAISS:
    """
    Load a previously saved FAISS index from disk.
    Only loads indexes created by this application (stored under indexes_dir).

    Raises FileNotFoundError if the index doesn't exist.
    """
    save_path = settings.indexes_dir / document_id
    if not save_path.exists():
        raise FileNotFoundError(
            f"No FAISS index found for document '{document_id}'. "
            "Upload and process the document first."
        )

    embeddings = get_embeddings()
    # allow_dangerous_deserialization is safe here because we only load
    # indexes that this application created and saved under our data directory.
    return FAISS.load_local(
        str(save_path),
        embeddings,
        allow_dangerous_deserialization=True,
    )


def retrieve_relevant_chunks(
    document_id: str,
    query: str,
    top_k: int | None = None,
) -> list[Document]:
    """
    Retrieve the most relevant text chunks for a given query.

    Args:
        document_id: ID of the document whose index to search.
        query: The topic or question to search for.
        top_k: Number of results to return (defaults to settings.RETRIEVAL_TOP_K).

    Returns:
        List of LangChain Document objects ranked by relevance.
    """
    if top_k is None:
        top_k = settings.RETRIEVAL_TOP_K

    vectorstore = load_faiss_index(document_id)
    results = vectorstore.similarity_search(query, k=top_k)

    if not results:
        raise ValueError(
            f"No relevant content found for topic '{query}' in document '{document_id}'. "
            "The paper cannot be grounded in the curriculum without matching passages."
        )
    return results
