"""
Phase 2 — Gemini Integration Smoke Test
Verifies that the API key, LLM, and embedding model work with a real call.
NOT part of the default test suite (requires live credentials).
Run manually: uv run python scripts/verify_gemini.py
"""

import sys
from pathlib import Path

# Ensure the project root is on the path so 'app' can be imported
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from app.config import settings
from app.model import get_llm, get_embeddings


def verify_api_key():
    """Check the API key is loaded (without printing it)."""
    if not settings.GOOGLE_API_KEY:
        print("❌ GOOGLE_API_KEY is not set. Add it to .env")
        return False
    # Show only that a key exists and its length, never the actual value
    print(f"✅ GOOGLE_API_KEY is loaded ({len(settings.GOOGLE_API_KEY)} chars)")
    print(f"   Model: {settings.GEMINI_MODEL}")
    print(f"   Embedding model: {settings.GEMINI_EMBEDDING_MODEL}")
    return True


def verify_llm():
    """Send a minimal prompt to Gemini and check the response."""
    print("\n--- LLM Test ---")
    try:
        llm = get_llm()
        response = llm.invoke("What is 2 + 3? Reply with just the number.")
        content = response.content
        # Some models return content as a list of parts/dicts instead of a string
        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, dict) and "text" in part:
                    parts.append(str(part["text"]))
                elif hasattr(part, "text"):
                    parts.append(str(getattr(part, "text")))
                else:
                    parts.append(str(part))
            answer = " ".join(parts).strip()
        else:
            answer = str(content).strip()
        print(f"✅ Gemini responded: '{answer}'")
        return True
    except Exception as e:
        print(f"❌ LLM call failed: {e}")
        return False


def verify_embeddings():
    """Embed a short text and check the vector dimensions."""
    print("\n--- Embedding Test ---")
    try:
        embeddings = get_embeddings()
        vector = embeddings.embed_query("quadratic equations")
        print(f"✅ Embedding returned a vector of {len(vector)} dimensions")
        return True
    except Exception as e:
        print(f"❌ Embedding call failed: {e}")
        return False


if __name__ == "__main__":
    print("=" * 50)
    print("QPGen — Gemini Integration Verification")
    print("=" * 50)

    key_ok = verify_api_key()
    if not key_ok:
        sys.exit(1)

    llm_ok = verify_llm()
    emb_ok = verify_embeddings()

    print("\n" + "=" * 50)
    if llm_ok and emb_ok:
        print("✅ All checks passed. Gemini integration is working.")
    else:
        failed = []
        if not llm_ok:
            failed.append("LLM")
        if not emb_ok:
            failed.append("Embeddings")
        print(f"❌ Failed: {', '.join(failed)}")
        sys.exit(1)
