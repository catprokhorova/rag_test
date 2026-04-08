import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    # Source documentation
    moodle_docs_root: str = os.getenv("MOODLE_DOCS_ROOT", "https://docs.moodle.org/501")

    # Local storage
    data_dir: Path = Path(os.getenv("DATA_DIR", "data"))

    # Qdrant
    qdrant_host: str = os.getenv("QDRANT_HOST", "localhost")
    qdrant_port: int = int(os.getenv("QDRANT_PORT", "6333"))
    qdrant_collection: str = os.getenv("QDRANT_COLLECTION", "moodle_docs")

    # Embeddings (multilingual)
    embed_model: str = os.getenv(
        "EMBED_MODEL", "intfloat/multilingual-e5-small"
    )
    embedding_batch_size: int = int(os.getenv("EMBED_BATCH_SIZE", "32"))

    # LLM generation (must run locally)
    llm_backend: str = os.getenv("LLM_BACKEND", "llama_cpp")
    # llama.cpp backend: path to local GGUF file
    llm_gguf_path: str = os.getenv("LLM_GGUF_PATH", "models/qwen2.5-3b-instruct-q4_k_m.gguf")

    llm_max_new_tokens: int = int(os.getenv("LLM_MAX_NEW_TOKENS", "256"))
    llm_max_input_tokens: int = int(os.getenv("LLM_MAX_INPUT_TOKENS", "1024"))
    llm_temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.1"))
    llm_repetition_penalty: float = float(os.getenv("LLM_REPETITION_PENALTY", "1.1"))
    llm_ctx: int = int(os.getenv("LLM_CTX", "2048"))
    llm_threads: int = int(os.getenv("LLM_THREADS", "4"))

    # Retrieval
    retrieve_top_k: int = int(os.getenv("RETRIEVE_TOP_K", "8"))

    # Chunking
    chunk_max_chars: int = int(os.getenv("CHUNK_MAX_CHARS", "3000"))
    chunk_overlap_chars: int = int(os.getenv("CHUNK_OVERLAP_CHARS", "400"))


def settings() -> Settings:
    return Settings()

