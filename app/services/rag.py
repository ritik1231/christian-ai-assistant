"""
RAG service: embed query → retrieve top-K KJV verses from ChromaDB.
Denomination-aware: Catholic queries include Deuterocanonical books,
others are filtered to the 66-book Protestant canon.
"""
import sys
import warnings
warnings.filterwarnings("ignore", message=".*capture.*")

from functools import lru_cache
from sentence_transformers import SentenceTransformer
import chromadb

sys.path.insert(0, ".")
from app.config import CHROMA_DIR, RAG_TOP_K, DEUTEROCANONICAL_BOOKS

COLLECTION_NAME = "bible_verses"


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    return SentenceTransformer("all-MiniLM-L6-v2")


@lru_cache(maxsize=1)
def _get_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        return client.get_collection(COLLECTION_NAME)
    except Exception as exc:
        raise RuntimeError(
            "ChromaDB collection not found. Run `python scripts/build_index.py` first."
        ) from exc


def _denomination_filter(denomination: str) -> dict | None:
    """
    Non-Catholic denominations exclude Deuterocanonical books.
    Returns a ChromaDB `where` filter or None (no filter).
    """
    if denomination.lower() == "catholic":
        return None  # include everything

    # Exclude Deuterocanonical books for all other denominations
    excluded = list(DEUTEROCANONICAL_BOOKS)
    if not excluded:
        return None
    return {"book": {"$nin": excluded}}


def retrieve_verses(
    query: str,
    denomination: str = "Non-denominational (default)",
    top_k: int = RAG_TOP_K,
) -> list[dict]:
    """
    Returns a list of dicts: {ref, text, book, chapter, verse, score}
    sorted by relevance (highest first).
    """
    model = _get_model()
    collection = _get_collection()

    query_embedding = model.encode(query).tolist()
    where_filter = _denomination_filter(denomination)

    kwargs = dict(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    if where_filter:
        kwargs["where"] = where_filter

    results = collection.query(**kwargs)

    verses = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        verses.append({
            "ref":     meta["ref"],
            "text":    doc,
            "book":    meta["book"],
            "chapter": meta["chapter"],
            "verse":   meta["verse"],
            "score":   round(1 - dist, 4),   # cosine similarity (higher = more relevant)
        })

    return verses


def format_context_block(verses: list[dict]) -> str:
    """Format retrieved verses into an LLM-ready context block."""
    if not verses:
        return "No directly relevant scripture found for this query."

    lines = ["RETRIEVED SCRIPTURE (KJV):"]
    for v in verses:
        lines.append(f"  [{v['ref']}] {v['text']}")
    return "\n".join(lines)
