from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, List, Optional


@dataclass
class Document:
    document_id: str
    filename: str
    source_path: str
    ingestion_status: str = "PENDING"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"))


@dataclass
class DocumentPage:
    page_number: int
    text_content: str


@dataclass
class DocumentChunk:
    chunk_id: str
    document_id: str
    text: str
    page_number: int
    section: Optional[str] = None
    source_file: str = ""
    chunk_position: int = 0
    provenance: Optional[dict] = None


@dataclass
class Citation:
    source_document: str
    page_number: int
    chunk_id: str
    snippet: str


@dataclass
class Embedding:
    chunk_id: str
    model_name: str
    vector: List[float]
    dimension: int


@dataclass
class VectorRecord:
    vector_id: str
    chunk_id: str
    embedding: List[float]
    metadata_ref: str
    source_document: str
    similarity_score: Optional[float] = 0.0
    chunk_text: Optional[str] = None


@dataclass
class SearchQuery:
    query_text: str
    top_k: int = 4
    similarity_threshold: float = 0.20
    filters: Optional[dict] = None


@dataclass
class SearchResult:
    chunk_id: str
    document_id: str
    page_number: int
    section: Optional[str]
    similarity_score: float
    source_document: str
    matched_text: str


@dataclass
class EvidenceSet:
    selected_chunks: List[DocumentChunk]
    support_level: str = "SUPPORTED"
    citations: List[Citation] = field(default_factory=list)
    confidence_score: float = 0.0


@dataclass
class GeneratedAnswer:
    answer_text: str
    citations: List[Citation] = field(default_factory=list)
    confidence: Optional[float] = None
    safety_status: str = "PASS"


@dataclass
class SafetyDecision:
    decision: str
    reasoning: str
    classification: str = "SUPPORTED"


@dataclass
class VectorStoreMetrics:
    document_count: int = 0
    chunk_count: int = 0
    degraded_mode: bool = False
