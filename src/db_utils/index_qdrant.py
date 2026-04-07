import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List

from tqdm import tqdm

from src.config import settings
from src.db_utils.qdrant_store import QdrantStore


def iter_chunks_jsonl(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def build_argparser() -> argparse.ArgumentParser:
    cfg = settings()
    p = argparse.ArgumentParser(description="Embed Moodle chunks and index them into local Qdrant.")
    p.add_argument(
        "--chunks-jsonl",
        type=Path,
        default=cfg.data_dir / "processed" / "moodle_chunks.jsonl",
    )
    p.add_argument("--batch-size", type=int, default=cfg.embedding_batch_size)
    p.add_argument("--recreate-collection", action="store_true", help="Drop and recreate collection.")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    cfg = settings()

    if not args.chunks_jsonl.exists():
        raise FileNotFoundError(f"Chunks file not found: {args.chunks_jsonl}")

    # Local embeddings (no external services).
    from src.rag.embeddings import SentenceTransformerEmbedder

    embedder = SentenceTransformerEmbedder(model_name=cfg.embed_model)
    store = QdrantStore()

    if args.recreate_collection:
        # Optional collection reset for clean re-index.
        store.client.delete_collection(store.collection_name)

    store.ensure_collection(vector_size=embedder.embedding_dimension)

    batch_texts: List[str] = []
    batch_payloads: List[Dict] = []
    batch_ids: List[str] = []

    def flush_batch() -> None:
        nonlocal batch_texts, batch_payloads, batch_ids
        if not batch_texts:
            return

        vectors = embedder.embed_texts(batch_texts)
        store.upsert_embeddings(
            vectors=vectors,
            payloads=batch_payloads,
            ids=batch_ids,
        )
        batch_texts = []
        batch_payloads = []
        batch_ids = []

    for chunk in tqdm(iter_chunks_jsonl(args.chunks_jsonl), desc="index chunks"):
        chunk_id = chunk["chunk_id"]
        batch_ids.append(chunk_id)
        batch_texts.append(chunk["text"])
        batch_payloads.append(
            {
                "chunk_id": chunk_id,
                "page_title": chunk.get("page_title", ""),
                "url": chunk.get("url", ""),
                "chunk_index": chunk.get("chunk_index", 0),
                "text": chunk["text"],
            }
        )
        if len(batch_texts) >= args.batch_size:
            flush_batch()

    flush_batch()


if __name__ == "__main__":
    main()

