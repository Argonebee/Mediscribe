from __future__ import annotations

import hashlib
import json
import os
import tempfile
from typing import Any, List

from mediscribe.contracts import Document


class FileStorage:
    def __init__(self, storage_dir: str | None = None):
        if storage_dir is None:
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            storage_dir = os.path.join(repo_root, ".mediscribe", "storage")
        self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)
        self.inventory_path = os.path.join(self.storage_dir, "documents.json")

    def save_file(self, path: str, payload: Any) -> None:
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(path, "wb") as f:
            if isinstance(payload, (bytes, bytearray)):
                f.write(payload)
            else:
                f.write(str(payload).encode("utf-8"))

    def load_documents(self) -> List[Document]:
        if not os.path.exists(self.inventory_path):
            return []
        try:
            with open(self.inventory_path, "r", encoding="utf-8") as handle:
                items = json.load(handle)
        except (json.JSONDecodeError, OSError):
            return []
        return [Document(**item) for item in items]

    def persist_documents(self, documents: List[Document]) -> None:
        payload = [
            {
                "document_id": doc.document_id,
                "filename": doc.filename,
                "source_path": doc.source_path,
                "ingestion_status": doc.ingestion_status,
                "created_at": doc.created_at,
            }
            for doc in documents
        ]
        with open(self.inventory_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)


class IngestionService:
    def __init__(self, storage: FileStorage | None = None):
        self.storage = storage or FileStorage()
        self._documents: List[Document] = self.storage.load_documents()

    def process_file(self, file_data: Any, metadata: dict) -> List[Document]:
        return self.process_files([file_data], [metadata])

    def process_files(self, file_data_list: List[Any], metadata_list: List[dict]) -> List[Document]:
        documents: List[Document] = []
        for index, file_data in enumerate(file_data_list):
            metadata = metadata_list[index] if index < len(metadata_list) else {}
            filename = metadata.get("filename", f"unknown_{index}.txt")
            source_path = metadata.get("source_path", ".")
            payload_seed = f"{filename}|{source_path}|{len(str(file_data))}|{index}|{str(file_data)[:200]}"
            document_id = f"doc_{hashlib.sha256(payload_seed.encode('utf-8')).hexdigest()}"
            target_path = os.path.join(source_path, filename)

            try:
                self.storage.save_file(target_path, file_data)
                doc = Document(
                    document_id=document_id,
                    filename=filename,
                    source_path=target_path,
                    ingestion_status="SUCCESS",
                )
                documents.append(doc)
            except Exception:
                doc = Document(
                    document_id=document_id,
                    filename=filename,
                    source_path=target_path,
                    ingestion_status="FAILED",
                )
                documents.append(doc)

            existing = [item for item in self._documents if item.document_id == doc.document_id]
            if existing:
                self._documents = [item for item in self._documents if item.document_id != doc.document_id]
            self._documents.append(doc)
        self.storage.persist_documents(self._documents)
        return documents

    def list_documents(self) -> List[Document]:
        self._documents = self.storage.load_documents()
        return sorted(self._documents, key=lambda item: item.created_at, reverse=True)
