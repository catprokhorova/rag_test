import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from langchain_core.documents import Document
from tqdm import tqdm

from src.backend.infrastructure.vector_store.qdrant_store import (
    delete_collection_if_exists,
    ensure_collection,
    get_vector_store,
    to_qdrant_point_id,
)
from src.config import settings
from src.rag.embeddings import get_embeddings


def iter_chunks_jsonl(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def index_chunks(
    *,
    chunks_jsonl: Path,
    batch_size: int,
    recreate_collection: bool,
    embeddings=None,
) -> int:
    if not chunks_jsonl.exists():
        raise FileNotFoundError(f"Chunks file not found: {chunks_jsonl}")

    embeddings = embeddings or get_embeddings()

    if recreate_collection:
        delete_collection_if_exists()

    ensure_collection(vector_size=512)
    store = get_vector_store(embeddings)

    batch_docs: List[Document] = []
    batch_ids: List[str | int] = []
    indexed = 0

    def flush_batch() -> None:
        nonlocal batch_docs, batch_ids, indexed
        if not batch_docs:
            return

        store.add_documents(documents=batch_docs, ids=[str(point_id) for point_id in batch_ids])
        indexed += len(batch_docs)
        batch_docs = []
        batch_ids = []

    for chunk in tqdm(iter_chunks_jsonl(chunks_jsonl), desc="index chunks"):
        chunk_id = chunk["chunk_id"]
        batch_ids.append(to_qdrant_point_id(chunk_id))
        batch_docs.append(
            Document(
                page_content=chunk["text"],
                metadata={
                "chunk_id": chunk_id,
                "page_title": chunk.get("page_title", ""),
                "url": chunk.get("url", ""),
                "chunk_index": chunk.get("chunk_index", 0),
                },
            )
        )
        if len(batch_docs) >= batch_size:
            flush_batch()

    flush_batch()
    return indexed


def build_argparser() -> argparse.ArgumentParser:
    cfg = settings()
    parser = argparse.ArgumentParser(
        description="Embed docs chunks and index them into local Qdrant."
    )
    parser.add_argument(
        "--chunks-jsonl",
        type=Path,
        default=cfg.data_dir / "processed" / "docs_chunks.jsonl",
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
