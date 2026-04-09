from dataclasses import dataclass
from uuid import NAMESPACE_URL, UUID, uuid5
from typing import Dict, List, Optional, Sequence

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
        existing = self.client.get_collections().collections
        if any(collection.name == self.collection_name for collection in existing):
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
        for vec, payload, point_id in zip(vectors, payloads, ids):
            points.append(
                qmodels.PointStruct(
                    id=self._to_qdrant_point_id(point_id),
                    vector=vec,
                    payload=payload,
                )
            )
        self.client.upsert(collection_name=self.collection_name, points=points)

    @staticmethod
    def _to_qdrant_point_id(point_id: str) -> str | int:
        """
        Convert external chunk IDs to valid Qdrant point IDs.

        Qdrant accepts only unsigned integers or UUID strings.
        """
        if isinstance(point_id, int):
            if point_id < 0:
                raise ValueError("Qdrant point id must be non-negative")
            return point_id

        # Keep numeric string IDs as integers.
        if isinstance(point_id, str) and point_id.isdigit():
            return int(point_id)

        # Pass through valid UUIDs unchanged.
        try:
            UUID(str(point_id))
            return str(point_id)
        except (TypeError, ValueError):
            pass

        # Deterministically map arbitrary strings (e.g. short hashes) to UUID.
        return str(uuid5(NAMESPACE_URL, str(point_id)))

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
        for result in results:
            payload = result.payload or {}
            chunks.append(
                RetrievedChunk(
                    chunk_id=str(payload.get("chunk_id") or result.id),
                    page_title=str(payload.get("page_title") or ""),
                    url=str(payload.get("url") or ""),
                    chunk_index=int(payload.get("chunk_index") or 0),
                    text=str(payload.get("text") or ""),
                    score=float(result.score or 0.0),
                )
            )
        return chunks
