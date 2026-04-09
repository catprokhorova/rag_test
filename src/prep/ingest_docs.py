import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set
from urllib.parse import urldefrag, urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

from src.config import settings
from src.prep.chunking import chunk_text, make_chunk_id

DEFAULT_START_URLS = (
    "https://docs.langchain.com/oss/python/langchain/overview",
    "https://docs.langchain.com/oss/python/langgraph/overview",
)
DEFAULT_ALLOWED_PREFIXES = (
    "https://docs.langchain.com/oss/python/langchain/",
    "https://docs.langchain.com/oss/python/langgraph/",
)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _load_state(state_path: Path) -> Dict[str, bool]:
    if not state_path.exists():
        return {}
    return json.loads(state_path.read_text(encoding="utf-8"))


def _save_state(state_path: Path, state: Dict[str, bool]) -> None:
    _ensure_dir(state_path.parent)
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def _normalize_url(url: str) -> str:
    clean, _ = urldefrag(url.strip())
    return clean.rstrip("/")


def _is_allowed_url(url: str, allowed_prefixes: Sequence[str]) -> bool:
    return any(url.startswith(prefix.rstrip("/")) for prefix in allowed_prefixes)


def _title_from_url(url: str, soup: BeautifulSoup) -> str:
    if soup.title and soup.title.string:
        return soup.title.string.strip()
    return urlparse(url).path.strip("/") or url


def _clean_html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript", "svg"]):
        tag.decompose()

    main = soup.find("main") or soup.body or soup
    text = main.get_text("\n")
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _extract_internal_links(
    *,
    base_url: str,
    soup: BeautifulSoup,
    allowed_prefixes: Sequence[str],
) -> List[str]:
    links: List[str] = []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if not href:
            continue
        full = _normalize_url(urljoin(base_url, href))
        parsed = urlparse(full)
        if parsed.scheme not in ("http", "https"):
            continue
        if not _is_allowed_url(full, allowed_prefixes):
            continue
        links.append(full)
    return links


def _iter_docs_pages(
    *,
    start_urls: Sequence[str],
    allowed_prefixes: Sequence[str],
    max_pages: Optional[int],
    timeout_s: int = 20,
) -> Iterable[Dict[str, str]]:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "rag-test-ingester/1.0 (+local)",
            "Accept": "text/html,application/xhtml+xml",
        }
    )

    queue: List[str] = [_normalize_url(url) for url in start_urls]
    queued: Set[str] = set(queue)
    seen: Set[str] = set()

    while queue:
        if max_pages is not None and len(seen) >= max_pages:
            break
        url = queue.pop(0)
        if url in seen:
            continue
        seen.add(url)

        try:
            resp = session.get(url, timeout=timeout_s)
            resp.raise_for_status()
        except Exception:
            continue

        ctype = resp.headers.get("Content-Type", "")
        if "text/html" not in ctype:
            continue

        html = resp.text
        soup = BeautifulSoup(html, "lxml")
        title = _title_from_url(url, soup)
        text = _clean_html_to_text(html)
        yield {"title": title, "url": url, "text": text}

        for link in _extract_internal_links(base_url=url, soup=soup, allowed_prefixes=allowed_prefixes):
            if link not in queued and link not in seen:
                queue.append(link)
                queued.add(link)


def ingest(
    *,
    start_urls: Sequence[str],
    allowed_prefixes: Sequence[str],
    max_pages: Optional[int],
    output_jsonl: Path,
    resume: bool,
):
    cfg = settings()
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    state_path = output_jsonl.parent / "ingest_state.json"
    state: Dict[str, bool] = _load_state(state_path) if resume else {}

    out_f = output_jsonl.open("a" if resume else "w", encoding="utf-8")

    def write_chunk(page_title: str, page_url: str, chunk_text_value: str, chunk_index: int) -> None:
        chunk_id = make_chunk_id(page_url, chunk_index)
        payload = {
            "chunk_id": chunk_id,
            "page_title": page_title,
            "url": page_url,
            "chunk_index": chunk_index,
            "text": chunk_text_value,
        }
        out_f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    pages = _iter_docs_pages(
        start_urls=start_urls,
        allowed_prefixes=allowed_prefixes,
        max_pages=max_pages,
    )
    for page in tqdm(pages, desc="fetch+chunk docs"):
        page_url = page["url"]
        if resume and state.get(page_url):
            continue
        cleaned = page["text"]
        if not cleaned:
            state[page_url] = True
            _save_state(state_path, state)
            continue
        chunks = chunk_text(
            cleaned,
            max_chars=cfg.chunk_max_chars,
            overlap_chars=cfg.chunk_overlap_chars,
        )
        for i, c in enumerate(chunks):
            write_chunk(page["title"], page_url, c, i)
        state[page_url] = True
        _save_state(state_path, state)

    out_f.close()


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Ingestion for LangChain/LangGraph docs (HTML fetch + chunking).")
    p.add_argument(
        "--start-url",
        action="append",
        dest="start_urls",
        default=None,
        help="Seed URL to crawl. Can be specified multiple times.",
    )
    p.add_argument(
        "--allowed-prefix",
        action="append",
        dest="allowed_prefixes",
        default=None,
        help="Allowed URL prefix for crawling. Can be specified multiple times.",
    )
    p.add_argument("--max-pages", type=int, default=None, help="Limit pages for quick runs.")
    p.add_argument(
        "--output-jsonl",
        type=Path,
        default=settings().data_dir / "processed" / "docs_chunks.jsonl",
    )
    p.add_argument("--resume", action="store_true", help="Resume using ingest_state.json.")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    start_urls = args.start_urls or list(DEFAULT_START_URLS)
    allowed_prefixes = args.allowed_prefixes or list(DEFAULT_ALLOWED_PREFIXES)
    ingest(
        start_urls=start_urls,
        allowed_prefixes=allowed_prefixes,
        max_pages=args.max_pages,
        output_jsonl=args.output_jsonl,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
