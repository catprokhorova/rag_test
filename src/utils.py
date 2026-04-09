import hashlib
import re

def stable_hash(value: str) -> str:
    """Deterministic short hash for identifiers."""
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()
    return digest[:16]

_RU_RE = re.compile(r"[\u0400-\u04FF]")

def detect_language(text: str) -> str:
    """Very small heuristic for RU vs EN."""
    if _RU_RE.search(text or ""):
        return "ru"
    return "en"

def title_to_page_url(root: str, title: str) -> str:
    """
    MediaWiki title -> docs URL (best-effort).

    Example:
      root: https://docs.example.com
      title: "Course settings"
      => https://docs.example.com/en/Course_settings
    """
    safe_title = title.replace(" ", "_")
    # Preserve slashes in titles while escaping other characters.
    safe_title = safe_title.replace("%", "%25")
    return f"{root}/en/{safe_title}"

