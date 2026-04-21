import os
from functools import lru_cache
from typing import List

from langchain_community.embeddings import HuggingFaceEmbeddings
from src.config import settings


class E5HuggingFaceEmbeddings(HuggingFaceEmbeddings):
    """
    LangChain embeddings adapter with optional E5 query/passage prefixes.
    """

    def __init__(self, *, model_name: str):
        cfg = settings()
        if cfg.hf_home:
            os.environ.setdefault("HF_HOME", cfg.hf_home)
        super().__init__(
            model_name=model_name,
            model_kwargs={
                "local_files_only": cfg.embed_local_files_only,
                "token": cfg.hf_token,
            },
            encode_kwargs={
                "batch_size": cfg.embedding_batch_size,
                "normalize_embeddings": True,
            },
        )

    def _prefix_passages(self, texts: List[str]) -> List[str]:
        return [f"passage: {text}" for text in texts]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return super().embed_documents(self._prefix_passages(texts))

    def embed_query(self, text: str) -> List[float]:
        return super().embed_query(query)


@lru_cache(maxsize=1)
def get_embeddings() -> E5HuggingFaceEmbeddings:
    cfg = settings()
    return E5HuggingFaceEmbeddings(
        model_name=cfg.embed_model
    )

