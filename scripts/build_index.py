"""
One-time script: embed all KJV verses and store in ChromaDB.
Run once before starting the app: venv/bin/python scripts/build_index.py
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import KJV_PATH, CHROMA_DIR
from sentence_transformers import SentenceTransformer
import chromadb

COLLECTION_NAME = "bible_verses"
BATCH_SIZE = 512


def flatten_corpus(kjv_data: list) -> list[dict]:
    """Convert nested KJV JSON into flat list of verse dicts."""
    verses = []
    for book in kjv_data:
        book_name = book["name"]
        for ch_idx, chapter in enumerate(book["chapters"], start=1):
            for v_idx, text in enumerate(chapter, start=1):
                verses.append({
                    "id": f"{book_name}.{ch_idx}.{v_idx}",
                    "ref": f"{book_name} {ch_idx}:{v_idx}",
                    "book": book_name,
                    "chapter": ch_idx,
                    "verse": v_idx,
                    "text": text.strip(),
                })
    return verses


def build():
    print("Loading KJV corpus...")
    with open(KJV_PATH, encoding="utf-8") as f:
        kjv_data = json.load(f)

    verses = flatten_corpus(kjv_data)
    print(f"  {len(verses):,} verses loaded.")

    print("Loading embedding model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Connecting to ChromaDB...")
    CHROMA_DIR.mkdir(exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # Wipe existing collection on re-run
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    print(f"Embedding and indexing in batches of {BATCH_SIZE}...")
    total = len(verses)
    for i in range(0, total, BATCH_SIZE):
        batch = verses[i: i + BATCH_SIZE]
        texts = [v["text"] for v in batch]
        embeddings = model.encode(texts, show_progress_bar=False).tolist()
        collection.add(
            ids=[v["id"] for v in batch],
            documents=texts,
            embeddings=embeddings,
            metadatas=[{
                "ref":     v["ref"],
                "book":    v["book"],
                "chapter": v["chapter"],
                "verse":   v["verse"],
            } for v in batch],
        )
        done = min(i + BATCH_SIZE, total)
        print(f"  [{done:>5}/{total}] indexed", end="\r")

    print(f"\nDone. {collection.count():,} verses in ChromaDB at {CHROMA_DIR}")


def build_if_needed() -> bool:
    """
    Build the index only if the collection is missing or empty.
    Returns True if a build was performed, False if it already existed.
    """
    CHROMA_DIR.mkdir(exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        col = client.get_collection(COLLECTION_NAME)
        if col.count() > 0:
            return False
    except Exception:
        pass
    build()
    return True


if __name__ == "__main__":
    build()
