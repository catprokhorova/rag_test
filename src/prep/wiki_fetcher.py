import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.config import settings


USER_AGENT = (
    "docs-rag-bot/0.1 (local RAG; requests via MediaWiki API; "
    "contact: none; user-agent for reproducibility)"
)


@dataclass(frozen=True)
class WikiPage:
    title: str
    url: str
    wikitext: Optional[str]
    html: Optional[str]


class DocsWikiAPI:
    """
    Minimal MediaWiki API client for documentation pages.

    Key idea: use MediaWiki API rather than HTML scraping to avoid many
    anti-bot measures and make ingestion resumable.
    """

    def __init__(self, root_url: Optional[str] = None, cache_dir: Optional[Path] = None):
        cfg = settings()
        self.root_url = (root_url or cfg.docs_root).rstrip("/")
        self.api_url = f"{self.root_url}/api.php"
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path_for_title(self, title: str) -> Optional[Path]:
        if not self.cache_dir:
            return None
        safe = title.replace("/", "__").replace(":", "_")
        return self.cache_dir / f"{safe}.json"

    @retry(
        retry=retry_if_exception_type((requests.RequestException,)),
        stop=stop_after_attempt(6),
        wait=wait_exponential(min=1, max=30),
    )
    def _get_json(self, params: Dict) -> Dict:
        resp = self.session.get(self.api_url, params=params, timeout=60)
        # If the upstream returns a human verification page, we get non-JSON content.
        ctype = resp.headers.get("content-type", "")
        if "text/html" in ctype.lower():
            lowered = (resp.text or "").lower()
            if "captcha" in lowered or "verify" in lowered or "human" in lowered:
                raise RuntimeError(
                    "CAPTCHA/human verification detected while calling MediaWiki API. "
                    "Use offline ingestion with locally downloaded pages "
                    "or retry later with a browser-obtained session/cookies."
                )
        resp.raise_for_status()
        return resp.json()

    def get_all_links_from_page(self, title: str, *, max_pages: Optional[int] = None) -> List[str]:
        """
        Return all linked page titles from a given MediaWiki page, using continuation.
        """
        titles: List[str] = []
        seen = set()
        params = {
            "action": "query",
            "format": "json",
            "prop": "links",
            "titles": title,
            "pllimit": "max",
        }
        while True:
            payload = dict(params)
            if titles and "continue" in payload:
                payload.pop("continue")
            data = self._get_json(payload)

            pages = data.get("query", {}).get("pages", {})
            for _, page in pages.items():
                for link in page.get("links", []) or []:
                    lt = link.get("title")
                    if not lt or lt in seen:
                        continue
                    # MediaWiki titles often include namespaces (e.g. Category:, Help:).
                    # Keep only main-namespace-ish pages to reduce noise.
                    if ":" in lt:
                        continue
                    seen.add(lt)
                    titles.append(lt)
                    if max_pages and len(titles) >= max_pages:
                        return titles

            cont = data.get("continue")
            if not cont:
                break
            # continuation params are passed back as-is (plcontinue, etc.)
            params.update(cont)
        return titles

    def _fetch_page_wikitext(self, title: str) -> Optional[str]:
        data = self._get_json(
            {
                "action": "query",
                "format": "json",
                "titles": title,
                "prop": "revisions",
                "rvprop": "content",
                "rvslots": "main",
                "rvlimit": 1,
            }
        )
        pages = data.get("query", {}).get("pages", {}) or {}
        for _, page in pages.items():
            revisions = page.get("revisions") or []
            if not revisions:
                return None
            # rvslots.main has wikitext content under the "*" key
            slots = revisions[0].get("slots") or {}
            main = slots.get("main") or {}
            return main.get("*")
        return None

    def _fetch_page_html(self, title: str) -> Optional[str]:
        # prop=text returns HTML-ish page text.
        data = self._get_json(
            {
                "action": "parse",
                "format": "json",
                "page": title,
                "prop": "text",
                "disablelimitreport": 1,
            }
        )
        content = data.get("parse", {}).get("text", {}).get("*")
        return content

    def fetch_page(self, title: str, *, prefer_wikitext: bool = True) -> WikiPage:
        """
        Fetch a page content with best effort:
        - try wikitext (MediaWiki API revisions)
        - fallback to html parse (MediaWiki API parse)
        - optionally cache the raw API response
        """
        from src.utils import title_to_page_url

        url = title_to_page_url(self.root_url, title)
        cache_path = self._cache_path_for_title(title)
        if cache_path and cache_path.exists():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            return WikiPage(
                title=title,
                url=url,
                wikitext=cached.get("wikitext"),
                html=cached.get("html"),
            )

        wikitext: Optional[str] = None
        html: Optional[str] = None
        if prefer_wikitext:
            wikitext = self._fetch_page_wikitext(title)
            if not wikitext:
                html = self._fetch_page_html(title)
        else:
            html = self._fetch_page_html(title)
            if not html:
                wikitext = self._fetch_page_wikitext(title)

        if cache_path:
            cache_path.write_text(
                json.dumps({"wikitext": wikitext, "html": html}, ensure_ascii=False),
                encoding="utf-8",
            )

        return WikiPage(title=title, url=url, wikitext=wikitext, html=html)

    def fetch_titles(
        self,
        toc_title: str,
        *,
        max_pages: Optional[int] = None,
        extra_titles: Optional[Iterable[str]] = None,
    ) -> List[str]:
        titles = self.get_all_links_from_page(toc_title, max_pages=max_pages)
        if extra_titles:
            for t in extra_titles:
                if t not in titles and ":" not in t:
                    titles.append(t)
        return titles

