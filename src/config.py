import os
from dataclasses import dataclass
from pathlib import Path


def _csv_env(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw = os.getenv(name)
    if raw is None:
        return default
    items = tuple(item.strip() for item in raw.split(",") if item.strip())
    return items or default


@dataclass(frozen=True)
class Settings:
    # Source documentation
    docs_root: str = os.getenv(
        "DOCS_ROOT",
        "https://docs.langchain.com/oss/python/langchain/overview",
    )
    docs_start_urls: tuple[str, ...] = _csv_env(
        "DOCS_START_URLS",
        (
            "https://docs.langchain.com/oss/python/langchain/overview",
            "https://docs.langchain.com/oss/python/langgraph/overview",
        ),
    )
    docs_allowed_prefixes: tuple[str, ...] = _csv_env(
        "DOCS_ALLOWED_PREFIXES",
        (
            "https://docs.langchain.com/oss/python/langchain/",
            "https://docs.langchain.com/oss/python/langgraph/",
        ),
    )

    # Local storage
    data_dir: Path = Path(os.getenv("DATA_DIR", "data"))

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

    # LLM generation (must run locally)
    llm_backend: str = os.getenv("LLM_BACKEND", "llama_cpp")
    # llama.cpp backend: path to local GGUF file
    llm_gguf_path: str = os.getenv("LLM_GGUF_PATH")
    llm_model: str = os.getenv("LLM_MODEL", "qwen2.5-3b-instruct-q4_k_m")
    llm_max_new_tokens: int = int(os.getenv("LLM_MAX_NEW_TOKENS", "256"))
    llm_max_input_tokens: int = int(os.getenv("LLM_MAX_INPUT_TOKENS", "1024"))
    llm_temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.1"))
    llm_repetition_penalty: float = float(os.getenv("LLM_REPETITION_PENALTY", "1.1"))
    llm_ctx: int = int(os.getenv("LLM_CTX", "8192"))
    llm_threads: int = int(os.getenv("LLM_THREADS", "4"))

    # Retrieval
    retrieve_top_k: int = int(os.getenv("RETRIEVE_TOP_K", "8"))

    # Chunking
    chunk_max_chars: int = int(os.getenv("CHUNK_MAX_CHARS", "3000"))
    chunk_overlap_chars: int = int(os.getenv("CHUNK_OVERLAP_CHARS", "400"))


def settings() -> Settings:
    return Settings()

