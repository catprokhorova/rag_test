import os
from functools import lru_cache
from typing import List

from langchain_community.embeddings import HuggingFaceEmbeddings
from src.config import settings


class E5HuggingFaceEmbeddings(HuggingFaceEmbeddings):
    """
    LangChain embeddings adapter with optional E5 query/passage prefixes.
    """

    def __init__(self, *, model_name: str, use_e5_prefixes: bool = False):
        cfg = settings()
        if cfg.hf_home:
            os.environ.setdefault("HF_HOME", cfg.hf_home)
        self.use_e5_prefixes = use_e5_prefixes
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
        self.embedding_dimension = int(self.client.get_sentence_embedding_dimension())

    def _prefix_passages(self, texts: List[str]) -> List[str]:
        if not self.use_e5_prefixes:
            return texts
        return [f"passage: {text}" for text in texts]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return super().embed_documents(self._prefix_passages(texts))

    def embed_query(self, text: str) -> List[float]:
        query = f"query: {text}" if self.use_e5_prefixes else text
        return super().embed_query(query)


@lru_cache(maxsize=1)
def get_embeddings() -> E5HuggingFaceEmbeddings:
    cfg = settings()
    return E5HuggingFaceEmbeddings(
        model_name=cfg.embed_model,
        use_e5_prefixes=cfg.embed_use_e5_prefixes,
    )

