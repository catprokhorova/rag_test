import math
from typing import Dict, Iterable, List, Tuple


def _split_into_paragraphs(text: str) -> List[str]:
    # Keep paragraph-like blocks, filter too-short noise.
    parts = [p.strip() for p in text.split("\n\n")]
    return [p for p in parts if len(p) >= 40]


def chunk_text(
    text: str,
    *,
    max_chars: int = 3000,
    overlap_chars: int = 400,
) -> List[str]:
    """
    Simple paragraph-based chunking with overlap.

    We chunk by character budget (token-aware chunking requires tokenizer).
    """
    text = text.strip()
    if not text:
        return []

    paras = _split_into_paragraphs(text)
    if not paras:
        return [text]

    chunks: List[str] = []
    current: List[str] = []
    current_len = 0

    def flush() -> None:
        nonlocal chunks, current, current_len
        if not current:
            return
        chunk = "\n\n".join(current).strip()
        if chunk:
            chunks.append(chunk)
        current = []
        current_len = 0

    for p in paras:
        if current_len + len(p) + 2 > max_chars and current:
            flush()
            # overlap: take tail paragraphs until overlap_chars budget
            if overlap_chars > 0:
                tail: List[str] = []
                tail_len = 0
                for t in reversed(chunks[-1].split("\n\n")):
                    if tail_len + len(t) + 2 > overlap_chars:
                        break
                    tail.append(t)
                    tail_len += len(t) + 2
                current = list(reversed(tail))
                current_len = sum(len(x) for x in current) + max(0, len(current) - 1) * 2

        current.append(p)
        current_len += len(p) + 2

    flush()
    return chunks


def make_chunk_id(page_title: str, chunk_index: int) -> str:
    from src.utils import stable_hash

    return stable_hash(f"{page_title}#{chunk_index}")

