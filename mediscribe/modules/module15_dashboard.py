from __future__ import annotations

from typing import Any, Dict, List

from mediscribe.contracts import VectorStoreMetrics


class DashboardService:
    def get_report(
        self,
        metrics: VectorStoreMetrics | dict | None = None,
        documents: List[Dict[str, Any]] | None = None,
        provenance: Dict[str, List[Dict[str, Any]]] | None = None,
    ) -> dict:
        if isinstance(metrics, dict):
            metrics_obj = VectorStoreMetrics(
                document_count=metrics.get("document_count", 0),
                chunk_count=metrics.get("chunk_count", 0),
                degraded_mode=metrics.get("degraded_mode", False),
            )
        else:
            metrics_obj = metrics or VectorStoreMetrics()

        normalized_documents = []
        for doc in documents or []:
            if isinstance(doc, dict):
                normalized_documents.append({
                    "document_id": doc.get("document_id"),
                    "filename": doc.get("filename"),
                    "status": doc.get("status"),
                })
            else:
                normalized_documents.append({
                    "document_id": getattr(doc, "document_id", None),
                    "filename": getattr(doc, "filename", None),
                    "status": getattr(doc, "ingestion_status", None),
                })

        normalized_provenance = {
            document_id: [
                {
                    "chunk_id": item.get("chunk_id"),
                    "page_number": item.get("page_number"),
                    "source_file": item.get("source_file"),
                }
                for item in entries
            ]
            for document_id, entries in (provenance or {}).items()
        }

        return {
            "document_count": metrics_obj.document_count,
            "chunk_count": metrics_obj.chunk_count,
            "degraded_mode": metrics_obj.degraded_mode,
            "documents": normalized_documents,
            "provenance": normalized_provenance,
        }
