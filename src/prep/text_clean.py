import re
from typing import Optional

from bs4 import BeautifulSoup


_WS_RE = re.compile(r"[ \t]+")
_MULTI_NL_RE = re.compile(r"\n{3,}")
_RU_CHUNK_RE = re.compile(r"\s+")


def _normalize_whitespace(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = _WS_RE.sub(" ", text)
    text = _MULTI_NL_RE.sub("\n\n", text)
    return text.strip()


def clean_wikitext(wikitext: str) -> str:
    """
    Convert MediaWiki wikitext -> readable plain text (best effort).

    For ingestion quality: we don't need perfect rendering; we need stable
    textual content suitable for embedding.
    """
    if not wikitext:
        return ""

    try:
        import mwparserfromhell  # type: ignore

        wikicode = mwparserfromhell.parse(wikitext)
        # Remove templates and collapse markup where possible.
        wikicode = wikicode.strip_code(normalize=True, collapse=True)
        text = str(wikicode)
        return _normalize_whitespace(text)
    except Exception:
        # Fallback: naive cleanup (still better than raw markup).
        text = re.sub(r"\{\{.*?\}\}", " ", wikitext, flags=re.S)
        # [[A|B]] -> B; [[A]] -> A
        text = re.sub(r"\[\[([^|\]]+)\|([^\]]+)\]\]", r"\2", text)
        text = re.sub(r"\[\[([^\]]+)\]\]", r"\1", text)
        # Remove remaining wiki brackets
        text = text.replace("[", " ").replace("]", " ")
        return _normalize_whitespace(text)


def clean_html(html: str) -> str:
    if not html:
        return ""

    soup = BeautifulSoup(html, "lxml")
    # Remove noisy nodes.
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text("\n")
    text = _normalize_whitespace(text)
    return text


def clean_page_text(wikitext: Optional[str], html: Optional[str]) -> str:
    if wikitext:
        cleaned = clean_wikitext(wikitext)
        if cleaned:
            return cleaned
    if html:
        return clean_html(html)
    return ""

