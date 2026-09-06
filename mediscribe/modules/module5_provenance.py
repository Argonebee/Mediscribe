from __future__ import annotations

from typing import Dict, List

from mediscribe.contracts import DocumentChunk


class ProvenanceService:
    def __init__(self):
        self.records: Dict[str, DocumentChunk] = {}

    def save_chunk(self, chunk: DocumentChunk) -> DocumentChunk:
        self.records[chunk.chunk_id] = chunk
        return chunk

    def list_chunks(self) -> List[DocumentChunk]:
        return list(self.records.values())

    def get_by_document(self, document_id: str) -> List[DocumentChunk]:
        return [chunk for chunk in self.records.values() if chunk.document_id == document_id]
