from uuid import NAMESPACE_URL, UUID, uuid5
from typing import Optional

from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from src.config import settings

def qdrant_client_from_settings() -> QdrantClient:
    cfg = settings()
    return QdrantClient(host=cfg.qdrant_host, port=cfg.qdrant_port)


def ensure_collection(*, vector_size: int, collection_name: Optional[str] = None) -> None:
    cfg = settings()
    name = collection_name or cfg.qdrant_collection
    client = qdrant_client_from_settings()
    existing = client.get_collections().collections
    if any(collection.name == name for collection in existing):
        info = client.get_collection(name)
        params = info.config.params.vectors
        if params is None:
            return
        existing_size: Optional[int] = None
        if isinstance(params, qmodels.VectorParams):
            existing_size = params.size
        elif isinstance(params, dict):
            first = next(iter(params.values()), None)
            if isinstance(first, qmodels.VectorParams):
                existing_size = first.size
        if existing_size is not None and existing_size != vector_size:
            raise ValueError(
                f"Qdrant collection {name!r} has vector size {existing_size}, "
                f"but the embedding model produces {vector_size}. "
                "Drop the collection or call ingest with recreate_collection=true."
            )
        return
    client.create_collection(
        collection_name=name,
        vectors_config=qmodels.VectorParams(
            size=vector_size,
            distance=qmodels.Distance.COSINE,
        ),
    )


def delete_collection_if_exists(collection_name: Optional[str] = None) -> None:
    cfg = settings()
    name = collection_name or cfg.qdrant_collection
    client = qdrant_client_from_settings()
    existing = client.get_collections().collections
    if any(collection.name == name for collection in existing):
        client.delete_collection(name)


def to_qdrant_point_id(point_id: str) -> str | int:
    if isinstance(point_id, int):
        if point_id < 0:
            raise ValueError("Qdrant point id must be non-negative")
        return point_id

    if isinstance(point_id, str) and point_id.isdigit():
        return int(point_id)

    try:
        UUID(str(point_id))
        return str(point_id)
    except (TypeError, ValueError):
        return str(uuid5(NAMESPACE_URL, str(point_id)))


def get_vector_store(embeddings) -> Qdrant:
    cfg = settings()
    client = qdrant_client_from_settings()
    return Qdrant(
        client=client,
        collection_name=cfg.qdrant_collection,
        embeddings=embeddings,
    )
