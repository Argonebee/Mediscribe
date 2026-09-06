from __future__ import annotations

import json
import os
import tempfile
from typing import Dict, List, Union

from mediscribe.contracts import VectorRecord


class VectorStore:
    def __init__(self, storage_dir: str | None = None):
        self.storage_dir = storage_dir or os.path.join(tempfile.gettempdir(), "mediscribe_vectors")
        os.makedirs(self.storage_dir, exist_ok=True)
        self.records: Dict[str, VectorRecord] = {}
        self._load_records()

    def _load_records(self) -> None:
        self.records = {}
        if not os.path.isdir(self.storage_dir):
            return
        for filename in sorted(os.listdir(self.storage_dir)):
            if not filename.endswith(".json") or filename.endswith(".tmp"):
                continue
            path = os.path.join(self.storage_dir, filename)
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    payload = json.load(handle)
            except (OSError, json.JSONDecodeError):
                continue
            record = VectorRecord(
                vector_id=payload["vector_id"],
                chunk_id=payload["chunk_id"],
                embedding=payload.get("embedding", []),
                metadata_ref=payload["metadata_ref"],
                source_document=payload["source_document"],
                similarity_score=payload.get("similarity_score", 0.0),
                chunk_text=payload.get("chunk_text"),
            )
            self.records[record.vector_id] = record

    def save(self, vector_record: Union[VectorRecord, dict]) -> VectorRecord:
        if isinstance(vector_record, dict):
            vector_record = VectorRecord(
                vector_id=vector_record["vector_id"],
                chunk_id=vector_record["chunk_id"],
                embedding=vector_record["embedding"],
                metadata_ref=vector_record["metadata_ref"],
                source_document=vector_record["source_document"],
                similarity_score=vector_record.get("similarity_score", 0.0),
                chunk_text=vector_record.get("chunk_text"),
            )

        temp_path = os.path.join(self.storage_dir, f"{vector_record.vector_id}.tmp")
        final_path = os.path.join(self.storage_dir, f"{vector_record.vector_id}.json")
        payload = {
            "vector_id": vector_record.vector_id,
            "chunk_id": vector_record.chunk_id,
            "embedding": vector_record.embedding,
            "metadata_ref": vector_record.metadata_ref,
            "source_document": vector_record.source_document,
            "similarity_score": vector_record.similarity_score,
            "chunk_text": vector_record.chunk_text,
        }

        with open(temp_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)

        os.replace(temp_path, final_path)
        self.records[vector_record.vector_id] = vector_record
        return vector_record

    def search(self, query_vector: List[float], top_k: int = 4) -> List[VectorRecord]:
        scored = []
        for record in self.records.values():
            dot = sum(a * b for a, b in zip(query_vector, record.embedding))
            norm_a = (sum(a * a for a in query_vector)) ** 0.5
            norm_b = (sum(b * b for b in record.embedding)) ** 0.5
            similarity = dot / (norm_a * norm_b) if norm_a and norm_b else 0.0
            scored.append((similarity, VectorRecord(
                vector_id=record.vector_id,
                chunk_id=record.chunk_id,
                embedding=record.embedding,
                metadata_ref=record.metadata_ref,
                source_document=record.source_document,
                similarity_score=similarity,
                chunk_text=record.chunk_text,
            )))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [record for _, record in scored[:top_k]]
