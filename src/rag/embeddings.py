from dataclasses import dataclass
from typing import Iterable, List, Sequence

from sentence_transformers import SentenceTransformer

from src.config import settings


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
        self.model = SentenceTransformer(model_name)
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

