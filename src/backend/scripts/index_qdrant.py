import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List

from tqdm import tqdm

from src.backend.infrastructure.vector_store.qdrant_store import QdrantStore
from src.config import settings


def iter_chunks_jsonl(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def index_chunks(*, chunks_jsonl: Path, batch_size: int, recreate_collection: bool) -> int:
    cfg = settings()
    if not chunks_jsonl.exists():
        raise FileNotFoundError(f"Chunks file not found: {chunks_jsonl}")

    from src.rag.embeddings import SentenceTransformerEmbedder

    embedder = SentenceTransformerEmbedder(model_name=cfg.embed_model)
    store = QdrantStore()

    if recreate_collection:
        store.client.delete_collection(store.collection_name)

    store.ensure_collection(vector_size=embedder.embedding_dimension)

    batch_texts: List[str] = []
    batch_payloads: List[Dict] = []
    batch_ids: List[str] = []
    indexed = 0

    def flush_batch() -> None:
        nonlocal batch_texts, batch_payloads, batch_ids, indexed
        if not batch_texts:
            return

        vectors = embedder.embed_texts(batch_texts)
        store.upsert_embeddings(
            vectors=vectors,
            payloads=batch_payloads,
            ids=batch_ids,
        )
        indexed += len(batch_texts)
        batch_texts = []
        batch_payloads = []
        batch_ids = []

    for chunk in tqdm(iter_chunks_jsonl(chunks_jsonl), desc="index chunks"):
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
        if len(batch_texts) >= batch_size:
            flush_batch()

    flush_batch()
    return indexed


def build_argparser() -> argparse.ArgumentParser:
    cfg = settings()
    parser = argparse.ArgumentParser(
        description="Embed Moodle chunks and index them into local Qdrant."
    )
    parser.add_argument(
        "--chunks-jsonl",
        type=Path,
        default=cfg.data_dir / "processed" / "moodle_chunks.jsonl",
    )
    parser.add_argument("--batch-size", type=int, default=cfg.embedding_batch_size)
    parser.add_argument(
        "--recreate-collection", action="store_true", help="Drop and recreate collection."
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    index_chunks(
        chunks_jsonl=args.chunks_jsonl,
        batch_size=args.batch_size,
        recreate_collection=args.recreate_collection,
    )


if __name__ == "__main__":
    main()
