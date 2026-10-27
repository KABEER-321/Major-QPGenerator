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

import logging
import math
import re
import time
from pathlib import Path
from typing import Optional, Union

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from app.config import settings
from app.model import get_embeddings
from app.schemas import extract_chapter_number

logger = logging.getLogger(__name__)


def build_faiss_index(
    chunks: list[dict | Document],
    document_id: str,
    batch_size: int = 20,
) -> FAISS:
    """
    Build a FAISS index from text chunks and save it to disk.
    Embeds in batches with automatic rate-limit (429) backoff.

    Args:
        chunks: List of dicts with 'content' and 'metadata' keys, or Document objects.
        document_id: Unique ID used as the folder name for saving.
        batch_size: Number of documents to embed per request batch.

    Returns:
        The FAISS vector store object.
    """
    # Convert chunk dicts or Documents into LangChain Document objects
    documents = [
        c if isinstance(c, Document) else Document(page_content=c["content"], metadata=c.get("metadata", {}))
        for c in chunks
    ]
    if not documents:
        raise ValueError(f"No documents provided to build index for {document_id}")

    embeddings = get_embeddings()

    vectorstore = None
    total_docs = len(documents)

    for i in range(0, total_docs, batch_size):
        batch = documents[i : i + batch_size]
        max_attempts = 5
        for attempt in range(1, max_attempts + 1):
            try:
                if vectorstore is None:
                    vectorstore = FAISS.from_documents(batch, embeddings)
                else:
                    vectorstore.add_documents(batch)
                break
            except Exception as e:
                err_str = str(e)
                if ("429" in err_str or "RESOURCE_EXHAUSTED" in err_str) and attempt < max_attempts:
                    wait_time = 20 * attempt
                    logger.warning(
                        "Rate limit 429 during embedding for %s (batch %d/%d). Sleeping %ds...",
                        document_id,
                        i // batch_size + 1,
                        (total_docs + batch_size - 1) // batch_size,
                        wait_time,
                    )
                    time.sleep(wait_time)
                else:
                    raise

        if i + batch_size < total_docs:
            time.sleep(1.0)

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


def is_chunk_in_permitted_scope(
    doc: Document,
    permitted_units: list[str] | None = None,
    first_half_only: bool = False,
) -> bool:
    """
    Check if a Document chunk belongs to the permitted examination scope.
    Filters by chapter/unit metadata, numerical chapter ordering, or explicit is_first_half flag.
    Strictly excludes chunks with is_first_half=False or matching forbidden chapters.
    """
    meta = doc.metadata or {}

    # Explicitly flagged as second half
    if first_half_only and meta.get("is_first_half") is False:
        return False

    # If permitted_units are provided, match by chapter/unit name or chapter number
    if permitted_units:
        doc_ch = (meta.get("chapter") or meta.get("unit") or "").strip().lower()
        if not doc_ch:
            # Fallback for chunks without chapter metadata
            if first_half_only:
                return meta.get("is_first_half") is True
            return False

        doc_num, _ = extract_chapter_number(doc_ch)
        for pu in permitted_units:
            pu_clean = pu.strip().lower()
            if not pu_clean:
                continue
            if doc_ch == pu_clean or pu_clean in doc_ch or doc_ch in pu_clean:
                return True
            pu_num, _ = extract_chapter_number(pu_clean)
            if doc_num != 9999 and pu_num != 9999 and doc_num == pu_num:
                return True
        return False

    # If first_half_only without specific permitted_units
    if first_half_only:
        if meta.get("is_first_half") is True:
            return True
        if "chunk_index" in meta and "total_chunks" in meta:
            return meta["chunk_index"] < math.ceil(meta["total_chunks"] / 2.0)
        return False

    return True


def retrieve_relevant_chunks(
    document_id: str,
    query: str,
    top_k: int | None = None,
    first_half_only: bool = False,
    permitted_units: list[str] | None = None,
) -> list[Document]:
    """
    Retrieve the most relevant text chunks for a given query.
    Optionally enforces mid-semester scope restrictions (first half of document or permitted units).
    Guarantees that when mid-semester or permitted units are specified, NO chunk from
    outside the permitted scope is returned.

    Args:
        document_id: ID of the document whose index to search.
        query: The topic or question to search for.
        top_k: Number of results to return (defaults to settings.RETRIEVAL_TOP_K).
        first_half_only: If True, restricts retrieval to the first half of the syllabus.
        permitted_units: Optional list of unit names to filter/prioritize.

    Returns:
        List of LangChain Document objects ranked by relevance.
    """
    if top_k is None:
        top_k = settings.RETRIEVAL_TOP_K

    vectorstore = load_faiss_index(document_id)
    all_stored_docs = list(vectorstore.docstore._dict.values())

    needs_filter = first_half_only or (permitted_units is not None and len(permitted_units) > 0)

    if needs_filter:
        # Pre-filter all allowed docs in the vector store
        allowed_docs = [
            d for d in all_stored_docs
            if is_chunk_in_permitted_scope(d, permitted_units, first_half_only)
        ]
        if not allowed_docs:
            raise ValueError(
                f"No textbook chunks found matching the permitted examination scope in document '{document_id}'."
            )

        filter_fn = lambda m: is_chunk_in_permitted_scope(
            Document(page_content="", metadata=m), permitted_units, first_half_only
        )
        try:
            raw_results = vectorstore.similarity_search(
                query,
                k=min(top_k * 4, len(allowed_docs)),
                filter=filter_fn,
                fetch_k=len(all_stored_docs),
            )
        except Exception as e:
            logger.warning("FAISS filter search failed: %s. Falling back to candidate filtering.", e)
            raw_results = vectorstore.similarity_search(query, k=min(len(all_stored_docs), 50))

        # Strictly enforce allowed docs only
        results = [d for d in raw_results if is_chunk_in_permitted_scope(d, permitted_units, first_half_only)]
        if not results:
            results = allowed_docs[:top_k]
        elif len(results) < top_k:
            # Supplement from allowed_docs to give Gemini sufficient context
            seen_chunk_ids = {d.metadata.get("chunk_id", id(d)) for d in results}
            for ad in allowed_docs:
                if len(results) >= top_k:
                    break
                cid = ad.metadata.get("chunk_id", id(ad))
                if cid not in seen_chunk_ids:
                    results.append(ad)
                    seen_chunk_ids.add(cid)
    else:
        results = vectorstore.similarity_search(query, k=top_k)

    if not results:
        raise ValueError(
            f"No relevant content found for topic '{query}' in document '{document_id}'. "
            "The paper cannot be grounded in the curriculum without matching passages."
        )

    return results[:top_k]
