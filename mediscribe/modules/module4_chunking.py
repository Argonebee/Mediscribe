from __future__ import annotations

import re
from typing import List, Optional

from mediscribe.contracts import DocumentChunk, DocumentPage


class ChunkingService:
    def __init__(self, max_chars: int = 500, overlap: int = 80):
        self.max_chars = max_chars
        self.overlap = overlap

    def chunk(self, pages: List[DocumentPage], document_id: str, source_file: str) -> List[DocumentChunk]:
        chunks: List[DocumentChunk] = []
        chunk_position = 1

        for page in pages:
            text = page.text_content
            sentences = re.split(r"(?<=[.!?])\s+", text)
            buffer = ""
            for sentence in sentences:
                if not sentence.strip():
                    continue

                if buffer and self._is_medical_pair(buffer, sentence):
                    buffer = (buffer + " " + sentence).strip()
                    continue

                if buffer and self._should_flush(buffer, sentence):
                    chunks.append(self._build_chunk(document_id, source_file, page.page_number, buffer, chunk_position))
                    chunk_position += 1
                    buffer = sentence
                    continue

                candidate = (buffer + " " + sentence).strip()
                if len(candidate) <= self.max_chars:
                    buffer = candidate
                    continue

                if buffer:
                    chunks.append(self._build_chunk(document_id, source_file, page.page_number, buffer, chunk_position))
                    chunk_position += 1
                    buffer = sentence
                else:
                    chunks.append(self._build_chunk(document_id, source_file, page.page_number, sentence, chunk_position))
                    chunk_position += 1

            if buffer:
                chunks.append(self._build_chunk(document_id, source_file, page.page_number, buffer, chunk_position))
                chunk_position += 1

        return chunks

    def _is_medical_pair(self, buffer: str, sentence: str) -> bool:
        combined = (buffer + " " + sentence).strip()
        buffer_has_dose = bool(re.search(r"\b\d+(?:\.\d+)?\s*(?:mg|g|mcg|ml|iu|units?)\b", buffer, flags=re.IGNORECASE))
        sentence_has_contraindication = bool(re.search(r"\b(?:contraindication|warning|avoid|do not use|not recommended|caution)\b", sentence.lower()))
        return buffer_has_dose and sentence_has_contraindication

    def _should_flush(self, buffer: str, sentence: str) -> bool:
        if not buffer:
            return False
        combined = (buffer + " " + sentence).strip()
        if len(combined) > self.max_chars:
            if self._is_medical_pair(buffer, sentence):
                return False
            return True
        if self._is_medical_pair(buffer, sentence):
            return False
        return False

    def _build_chunk(self, document_id: str, source_file: str, page_number: int, text: str, chunk_position: int) -> DocumentChunk:
        return DocumentChunk(
            chunk_id=f"chunk_{document_id}_{page_number}_{chunk_position}",
            document_id=document_id,
            text=text,
            page_number=page_number,
            section=None,
            source_file=source_file,
            chunk_position=chunk_position,
            provenance={"page_number": page_number, "source_file": source_file},
        )
