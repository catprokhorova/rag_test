"""Parse local LangChain PDFs and write chunked JSONL for Qdrant indexing."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Set

from langchain_community.document_loaders.pdf import PagedPDFSplitter
from tqdm import tqdm

from src.config import settings
from src.prep.chunking import chunk_text, make_chunk_id

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IngestOutcome:
    pdfs_processed: int
    chunks_appended: int

    @property
    def pages_fetched(self) -> int:
        """Alias kept for admin ingest API compatibility."""
        return self.pdfs_processed


def _state_path(output_jsonl: Path) -> Path:
    return output_jsonl.parent / "ingest_state.json"


def _load_processed_pdfs(state_path: Path) -> Set[str]:
    if not state_path.exists():
        return set()
    data = json.loads(state_path.read_text(encoding="utf-8"))
    return set(data.get("processed_pdfs", []))


def _save_processed_pdfs(state_path: Path, processed: Set[str]) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(
        json.dumps({"processed_pdfs": sorted(processed)}, indent=2),
        encoding="utf-8",
    )


def _iter_pdf_paths(pdf_dir: Path) -> List[Path]:
    return sorted(pdf_dir.glob("*.pdf"))


def _pdf_source_url(pdf_path: Path) -> str:
    return pdf_path.resolve().as_uri()


def _chunk_records_for_pdf(
    pdf_path: Path,
    *,
    max_chars: int,
    overlap_chars: int,
) -> List[dict]:
    loader = PagedPDFSplitter(str(pdf_path))
    page_docs = loader.load()
    page_title = pdf_path.stem
    source_url = _pdf_source_url(pdf_path)
    records: List[dict] = []

    for page_doc in page_docs:
        page_num = int(page_doc.metadata.get("page", 0))
        page_chunks = chunk_text(
            page_doc.page_content,
            max_chars=max_chars,
            overlap_chars=overlap_chars,
        )
        for chunk_index, text in enumerate(page_chunks):
            if not text.strip():
                continue
            records.append(
                {
                    "chunk_id": make_chunk_id(
                        f"{pdf_path.name}#p{page_num}", chunk_index
                    ),
                    "text": text,
                    "page_title": page_title,
                    "url": source_url,
                    "chunk_index": chunk_index,
                    "source_file": pdf_path.name,
                    "page": page_num,
                }
            )
    return records


def ingest(
    *,
    pdf_dir: Path | None = None,
    max_pdfs: int | None = None,
    output_jsonl: Path | None = None,
    resume: bool = False,
) -> IngestOutcome:
    cfg = settings()
    pdf_dir = pdf_dir or cfg.pdf_dir
    output_jsonl = output_jsonl or (cfg.data_dir / "processed" / "docs_chunks.jsonl")

    if not pdf_dir.is_dir():
        raise FileNotFoundError(f"PDF directory not found: {pdf_dir}")

    pdf_paths = _iter_pdf_paths(pdf_dir)
    if max_pdfs is not None:
        pdf_paths = pdf_paths[:max_pdfs]

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    state_path = _state_path(output_jsonl)

    if not resume:
        output_jsonl.write_text("", encoding="utf-8")
        processed: Set[str] = set()
    else:
        processed = _load_processed_pdfs(state_path)

    chunks_appended = 0
    pdfs_processed = 0

    with output_jsonl.open("a", encoding="utf-8") as out:
        for pdf_path in tqdm(pdf_paths, desc="ingest pdfs"):
            if pdf_path.name in processed:
                continue
            try:
                records = _chunk_records_for_pdf(
                    pdf_path,
                    max_chars=cfg.chunk_max_chars,
                    overlap_chars=cfg.chunk_overlap_chars,
                )
            except Exception:
                logger.exception("Failed to parse PDF: %s", pdf_path)
                raise

            for record in records:
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
            chunks_appended += len(records)
            processed.add(pdf_path.name)
            pdfs_processed += 1
            _save_processed_pdfs(state_path, processed)

    return IngestOutcome(pdfs_processed=pdfs_processed, chunks_appended=chunks_appended)


def build_argparser() -> argparse.ArgumentParser:
    cfg = settings()
    parser = argparse.ArgumentParser(
        description="Parse local LangChain PDFs into docs_chunks.jsonl."
    )
    parser.add_argument(
        "--pdf-dir",
        type=Path,
        default=cfg.pdf_dir,
        help="Directory containing downloaded PDF files.",
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=cfg.data_dir / "processed" / "docs_chunks.jsonl",
    )
    parser.add_argument(
        "--max-pdfs",
        type=int,
        default=None,
        help="Limit number of PDFs (for quick tests).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip PDFs already recorded in ingest_state.json.",
    )
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = build_argparser().parse_args()
    outcome = ingest(
        pdf_dir=args.pdf_dir,
        max_pdfs=args.max_pdfs,
        output_jsonl=args.output_jsonl,
        resume=args.resume,
    )
    logger.info(
        "Ingest complete (pdfs_processed=%d, chunks_appended=%d, output=%s)",
        outcome.pdfs_processed,
        outcome.chunks_appended,
        args.output_jsonl,
    )


if __name__ == "__main__":
    main()
