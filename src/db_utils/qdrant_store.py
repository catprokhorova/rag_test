from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from src.config import settings


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    page_title: str
    url: str
    chunk_index: int
    text: str
    score: float


class QdrantStore:
    def __init__(
        self,
        *,
        host: Optional[str] = None,
        port: Optional[int] = None,
        collection_name: Optional[str] = None,
    ):
        cfg = settings()
        self.client = QdrantClient(host=host or cfg.qdrant_host, port=port or cfg.qdrant_port)
        self.collection_name = collection_name or cfg.qdrant_collection

    def ensure_collection(self, *, vector_size: int) -> None:
        """
        Create a collection if it doesn't exist.
        """
        existing = self.client.get_collections().collections
        if any(c.name == self.collection_name for c in existing):
            return

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=qmodels.VectorParams(
                size=vector_size,
                distance=qmodels.Distance.COSINE,
            ),
        )

    def upsert_embeddings(
        self,
        *,
        vectors: Sequence[Sequence[float]],
        payloads: Sequence[Dict],
        ids: Sequence[str],
    ) -> None:
        points = []
        for vec, payload, pid in zip(vectors, payloads, ids):
            points.append(qmodels.PointStruct(id=pid, vector=vec, payload=payload))
        self.client.upsert(collection_name=self.collection_name, points=points)

    def search(
        self,
        *,
        query_vector: Sequence[float],
        limit: int,
        score_threshold: Optional[float] = None,
    ) -> List[RetrievedChunk]:
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=list(query_vector),
            limit=limit,
            score_threshold=score_threshold,
            with_payload=True,
        )
        chunks: List[RetrievedChunk] = []
        for r in results:
            payload = r.payload or {}
            chunks.append(
                RetrievedChunk(
                    chunk_id=str(payload.get("chunk_id") or r.id),
                    page_title=str(payload.get("page_title") or ""),
                    url=str(payload.get("url") or ""),
                    chunk_index=int(payload.get("chunk_index") or 0),
                    text=str(payload.get("text") or ""),
                    score=float(r.score or 0.0),
                )
            )
        return chunks

