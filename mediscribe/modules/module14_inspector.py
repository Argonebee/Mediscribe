from __future__ import annotations

from typing import Any, Iterable


class EvidenceInspector:
    def render(self, answer: Any, evidence: Any) -> dict:
        citations = getattr(answer, "citations", []) or []
        return {
            "answer": getattr(answer, "answer_text", str(answer)),
            "evidence_count": len(getattr(evidence, "selected_chunks", [])),
            "citations": [self._normalize_citation(item) for item in citations],
        }

    def _normalize_citation(self, item: Any) -> dict:
        if isinstance(item, dict):
            return {
                "source_document": item.get("source_document"),
                "page_number": item.get("page_number"),
                "chunk_id": item.get("chunk_id"),
                "snippet": item.get("snippet"),
            }
        return {
            "source_document": getattr(item, "source_document", None),
            "page_number": getattr(item, "page_number", None),
            "chunk_id": getattr(item, "chunk_id", None),
            "snippet": getattr(item, "snippet", None),
        }
