from __future__ import annotations

from typing import List

from mediscribe.contracts import Citation, DocumentChunk, EvidenceSet, SearchResult


class SelectorService:
    def select(self, results: List[SearchResult], threshold: float = 0.20) -> EvidenceSet:
        ranked_results = sorted(results, key=lambda result: result.similarity_score, reverse=True)
        selected = []
        citations = []
        for result in ranked_results:
            if result.similarity_score >= threshold:
                chunk = DocumentChunk(
                    chunk_id=result.chunk_id,
                    document_id=result.document_id,
                    text=result.matched_text,
                    page_number=result.page_number,
                    section=result.section,
                    source_file=result.source_document,
                    chunk_position=1,
                    provenance={"source_document": result.source_document},
                )
                selected.append(chunk)
                citations.append(
                    Citation(
                        source_document=result.source_document,
                        page_number=result.page_number,
                        chunk_id=result.chunk_id,
                        snippet=result.matched_text,
                    )
                )
        support_level = "SUPPORTED" if selected else "INSUFFICIENT_EVIDENCE"
        top_confidence = max((r.similarity_score for r in ranked_results), default=0.0)
        return EvidenceSet(selected_chunks=selected, support_level=support_level, citations=citations, confidence_score=top_confidence)
