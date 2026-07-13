from dataclasses import dataclass
from typing import List, Optional

from langchain_core.documents import Document
from langfuse import observe
from src.backend.infrastructure.vector_store.qdrant_store import (
    ensure_collection,
    get_vector_store,
)
from src.config import settings
from src.observability.phoenix_client import retriever_span
from src.rag.embeddings import embedding_vector_size, get_embeddings


@dataclass(frozen=True)
class RetrievedContext:
    documents: List[Document]
    scores: List[float]

    def format_for_prompt(self, *, max_chars: int = 12000) -> str:
        """
        Build context block for the LLM prompt from retrieved chunks.
        """
        parts: List[str] = []
        used = 0
        for i, doc in enumerate(self.documents, start=1):
            title = str(doc.metadata.get("page_title") or "")
            block = f"[{i}] {title}\n{doc.page_content}"
            if used + len(block) > max_chars and parts:
                break
            parts.append(block)
            used += len(block)
        return "\n\n".join(parts).strip()

    def sources(self) -> List[dict]:
        return [
            {
                "title": str(doc.metadata.get("page_title") or ""),
                "url": str(doc.metadata.get("url") or ""),
                "chunk_id": str(doc.metadata.get("chunk_id") or ""),
                "chunk_index": int(doc.metadata.get("chunk_index") or 0),
                "score": score,
            }
            for doc, score in zip(self.documents, self.scores)
        ]


class QdrantRetriever:
    def __init__(
        self,
        *,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
    ):
        cfg = settings()
        self.embeddings = get_embeddings()
        ensure_collection(vector_size=embedding_vector_size(self.embeddings))
        self.store = get_vector_store(self.embeddings)
        self.top_k = top_k or cfg.retrieve_top_k
        self.score_threshold = (
            score_threshold
            if score_threshold is not None
            else cfg.retrieve_score_threshold
        )

    @retriever_span("retrieve-docs")
    @observe(name="retrieve-docs", as_type="span", capture_input=False, capture_output=False)
    def retrieve(self, query: str) -> RetrievedContext:
        from langfuse import get_client

        get_client().update_current_span(input={"query": query})

        scored = self.store.similarity_search_with_score(query=query, k=self.top_k)
        if self.score_threshold is not None:
            scored = [(doc, score) for doc, score in scored if score >= self.score_threshold]
        docs = [doc for doc, _ in scored]
        scores = [float(score) for _, score in scored]
        context = RetrievedContext(documents=docs, scores=scores)

        get_client().update_current_span(
            output={
                "num_chunks": len(docs),
                "sources": context.sources(),
            }
        )
        return context

