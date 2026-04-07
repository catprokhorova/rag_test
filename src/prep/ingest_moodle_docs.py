import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from tqdm import tqdm

from src.config import settings
from src.prep.chunking import chunk_text, make_chunk_id
from src.prep.text_clean import clean_page_text
from src.prep.wiki_fetcher import MoodleWikiAPI


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _load_state(state_path: Path) -> Dict[str, bool]:
    if not state_path.exists():
        return {}
    return json.loads(state_path.read_text(encoding="utf-8"))


def _save_state(state_path: Path, state: Dict[str, bool]) -> None:
    _ensure_dir(state_path.parent)
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def iter_local_pages_json(local_pages_dir: Path) -> Iterable[Dict]:
    # Expect files containing {"title":..., "wikitext":..., "html":...} (best effort).
    for p in sorted(local_pages_dir.glob("*.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(data, dict) and data.get("title"):
                yield data
        except Exception:
            continue


def ingest(
    *,
    toc_title: str,
    max_pages: Optional[int],
    output_jsonl: Path,
    cache_dir: Path,
    from_local_dir: Optional[Path],
    resume: bool,
):
    cfg = settings()
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Persist "which page titles were already processed".
    state_path = output_jsonl.parent / "ingest_state.json"
    state: Dict[str, bool] = _load_state(state_path) if resume else {}

    api = MoodleWikiAPI(cache_dir=cache_dir)

    # Output format: JSON Lines where each line is a chunk payload.
    out_f = output_jsonl.open("a" if resume else "w", encoding="utf-8")

    def write_chunk(page_title: str, page_url: str, text: str, chunk_index: int) -> None:
        chunk_id = make_chunk_id(page_title, chunk_index)
        payload = {
            "chunk_id": chunk_id,
            "page_title": page_title,
            "url": page_url,
            "chunk_index": chunk_index,
            "text": text,
        }
        out_f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    if from_local_dir:
        local_pages = list(iter_local_pages_json(from_local_dir))
        iterable_titles: List[str] = [p["title"] for p in local_pages]
        for page_data in tqdm(local_pages, desc="chunk local pages"):
            title = page_data["title"]
            if resume and state.get(title):
                continue
            from src.utils import title_to_page_url

            page_url = page_data.get("url") or title_to_page_url(cfg.moodle_docs_root, title)
            cleaned = clean_page_text(page_data.get("wikitext"), page_data.get("html"))
            chunks = chunk_text(
                cleaned,
                max_chars=cfg.chunk_max_chars,
                overlap_chars=cfg.chunk_overlap_chars,
            )
            for i, c in enumerate(chunks):
                write_chunk(title, page_url, c, i)
            state[title] = True
            _save_state(state_path, state)
        out_f.close()
        return

    titles = api.fetch_titles(toc_title, max_pages=max_pages)
    for title in tqdm(titles, desc="fetch+chunk pages"):
        if resume and state.get(title):
            continue
        page = api.fetch_page(title)
        cleaned = clean_page_text(page.wikitext, page.html)
        if not cleaned.strip():
            state[title] = True
            _save_state(state_path, state)
            continue
        chunks = chunk_text(
            cleaned,
            max_chars=cfg.chunk_max_chars,
            overlap_chars=cfg.chunk_overlap_chars,
        )
        for i, c in enumerate(chunks):
            write_chunk(title, page.url, c, i)
        state[title] = True
        _save_state(state_path, state)

    out_f.close()


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Offline ingestion for Moodle docs (MediaWiki API + chunking).")
    p.add_argument("--toc-title", default="Table_of_Contents")
    p.add_argument("--max-pages", type=int, default=None, help="Limit pages for quick runs.")
    p.add_argument(
        "--output-jsonl",
        type=Path,
        default=settings().data_dir / "processed" / "moodle_chunks.jsonl",
    )
    p.add_argument(
        "--cache-dir",
        type=Path,
        default=settings().data_dir / "cache" / "wiki_pages",
    )
    p.add_argument(
        "--from-local-dir",
        type=Path,
        default=None,
        help="If set, load pages from local JSON files instead of fetching via API.",
    )
    p.add_argument("--resume", action="store_true", help="Resume using ingest_state.json.")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    ingest(
        toc_title=args.toc_title,
        max_pages=args.max_pages,
        output_jsonl=args.output_jsonl,
        cache_dir=args.cache_dir,
        from_local_dir=args.from_local_dir,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()

