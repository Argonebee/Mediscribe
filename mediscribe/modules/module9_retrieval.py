from __future__ import annotations

from typing import List

from mediscribe.contracts import SearchResult, SearchQuery
from mediscribe.modules.module8_vector_store import VectorStore


class RetrievalEngine:
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store

    def search(self, query: SearchQuery, query_vector: List[float]) -> List[SearchResult]:
        results = self.vector_store.search(query_vector, top_k=query.top_k)
        items: List[SearchResult] = []
        for record in results:
            if record.similarity_score is None:
                score = 0.0
            else:
                score = record.similarity_score
            if score >= query.similarity_threshold:
                document_id = self._document_id_from_chunk(record.chunk_id)
                items.append(
                    SearchResult(
                        chunk_id=record.chunk_id,
                        document_id=document_id,
                        page_number=1,
                        section=None,
                        similarity_score=score,
                        source_document=record.source_document,
                        matched_text=record.chunk_text or f"Evidence from {record.source_document}",
                    )
                )
        return items

    def _document_id_from_chunk(self, chunk_id: str) -> str:
        if not chunk_id.startswith("chunk_"):
            return "unknown"
        parts = chunk_id[len("chunk_"):].split("_")
        if len(parts) < 2:
            return "unknown"
        if len(parts) >= 3:
            return "_".join(parts[:-1])
        return parts[0]
