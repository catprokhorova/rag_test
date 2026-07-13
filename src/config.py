import os
from dataclasses import dataclass
from pathlib import Path


def _csv_env(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw = os.getenv(name)
    if raw is None:
        return default
    items = tuple(item.strip() for item in raw.split(",") if item.strip())
    return items or default


def _optional_float_env(name: str, default: float | None) -> float | None:
    raw = os.getenv(name)
    if raw is None:
        return default
    stripped = raw.strip().lower()
    if stripped in {"", "none", "off", "disabled"}:
        return None
    return float(stripped)


@dataclass(frozen=True)
class Settings:
    # Local storage (PDFs live at the root of data_dir; processed/ holds JSONL)
    data_dir: Path = Path(os.getenv("DATA_DIR", "data"))
    # Override only if PDFs are not under data_dir
    pdf_dir: Path = Path(os.getenv("PDF_DIR") or os.getenv("DATA_DIR", "data"))

    # Qdrant
    qdrant_host: str = os.getenv("QDRANT_HOST", "localhost")
    qdrant_port: int = int(os.getenv("QDRANT_PORT", "6333"))
    qdrant_collection: str = os.getenv("QDRANT_COLLECTION", "docs_support")

    # Embeddings (multilingual)
    embed_model: str = os.getenv(
        "EMBED_MODEL", "intfloat/multilingual-e5-small"
    )
    embedding_batch_size: int = int(os.getenv("EMBED_BATCH_SIZE", "32"))
    # HF access/cache controls for embedding model loading.
    hf_token: str | None = os.getenv("HF_TOKEN")
    hf_home: str | None = os.getenv("HF_HOME")
    embed_local_files_only: bool = os.getenv("EMBED_LOCAL_FILES_ONLY", "1").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    # LLM generation (Yandex Cloud AI Studio via OpenAI-compatible Responses API)
    yandex_cloud_folder: str = os.getenv(
        "YANDEX_CLOUD_FOLDER", "b1g4o1j024hb54bou17c"
    )
    yandex_cloud_api_key: str | None = os.getenv("YANDEX_CLOUD_API_KEY") or None
    yandex_cloud_model: str = os.getenv(
        "YANDEX_CLOUD_MODEL", "aliceai-llm-flash/latest"
    )
    yandex_cloud_base_url: str = os.getenv(
        "YANDEX_CLOUD_BASE_URL", "https://ai.api.cloud.yandex.net/v1"
    )
    llm_request_timeout_s: float = float(os.getenv("LLM_REQUEST_TIMEOUT_S", "120"))
    llm_max_new_tokens: int = int(os.getenv("LLM_MAX_NEW_TOKENS", "256"))
    llm_max_input_tokens: int = int(os.getenv("LLM_MAX_INPUT_TOKENS", "2048"))
    llm_temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.3"))

    # Retrieval
    retrieve_top_k: int = int(os.getenv("RETRIEVE_TOP_K", "8"))
    retrieve_score_threshold: float | None = _optional_float_env(
        "RETRIEVE_SCORE_THRESHOLD", 0.75
    )

    # Chunking
    chunk_max_chars: int = int(os.getenv("CHUNK_MAX_CHARS", "3000"))
    chunk_overlap_chars: int = int(os.getenv("CHUNK_OVERLAP_CHARS", "400"))


def settings() -> Settings:
    return Settings()

