from dataclasses import dataclass
import os
from threading import Lock
from typing import Dict, Iterable, List, Sequence

from sentence_transformers import SentenceTransformer

from src.config import settings

_MODEL_CACHE: Dict[str, SentenceTransformer] = {}
_MODEL_CACHE_LOCK = Lock()


@dataclass(frozen=True)
class EmbeddingResult:
    vectors: List[List[float]]
    dim: int


class SentenceTransformerEmbedder:
    """
    Local multilingual embeddings via `sentence-transformers`.
    """

    def __init__(self, *, model_name: str):
        cfg = settings()
        self.model_name = model_name
        if cfg.hf_home:
            # Keep model cache in a persistent location in containers.
            os.environ.setdefault("HF_HOME", cfg.hf_home)
        with _MODEL_CACHE_LOCK:
            model = _MODEL_CACHE.get(model_name)
            if model is None:
                try:
                    model = SentenceTransformer(
                        model_name,
                        local_files_only=cfg.embed_local_files_only,
                        token=cfg.hf_token,
                    )
                except Exception as exc:
                    mode = "local-only" if cfg.embed_local_files_only else "online"
                    raise RuntimeError(
                        f"Failed to load embedding model '{model_name}' ({mode} mode). "
                        "Set HF_TOKEN for authenticated Hugging Face access, or pre-download "
                        "the model into cache and keep EMBED_LOCAL_FILES_ONLY=1."
                    ) from exc
                _MODEL_CACHE[model_name] = model
        self.model = model
        self.embedding_dimension = int(self.model.get_sentence_embedding_dimension())
        self.batch_size = cfg.embedding_batch_size

    def embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        vectors = self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        # Convert numpy arrays -> python lists for Qdrant client.
        return [v.tolist() for v in vectors]

    def embed_query(self, query: str) -> List[float]:
        return self.embed_texts([query])[0]

