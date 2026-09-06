from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel

from mediscribe.config import SETTINGS
from mediscribe.contracts import Document, SearchQuery, SearchResult, VectorRecord
from mediscribe.modules.module1_ingestion import IngestionService
from mediscribe.modules.module2_extractor import ExtractorService
from mediscribe.modules.module3_normalizer import NormalizerService
from mediscribe.modules.module4_chunking import ChunkingService
from mediscribe.modules.module6_embedding import EmbeddingService
from mediscribe.modules.module8_vector_store import VectorStore
from mediscribe.modules.module9_retrieval import RetrievalEngine
from mediscribe.modules.module10_selector import SelectorService
from mediscribe.modules.module11_generator import GeneratorService
from mediscribe.modules.module12_verifier import VerifierService
from mediscribe.modules.module13_safety import SafetyService
from mediscribe.modules.module15_dashboard import DashboardService
from mediscribe.modules.module16_hardening import (
    logger,
    validate_input,
    RequestLogger,
    ValidationError,
)
from mediscribe.modules.module17_operations import SimpleCache, VectorStoreHealthCheck

app = FastAPI(title="Mediscribe API", version="0.1.0")

ingest_service = IngestionService()
extractor_service = ExtractorService()
normalizer_service = NormalizerService()
chunking_service = ChunkingService(max_chars=SETTINGS.max_chunk_chars, overlap=SETTINGS.chunk_overlap)
embedding_service = EmbeddingService()
vector_store = VectorStore()
retrieval_engine = RetrievalEngine(vector_store)
selector_service = SelectorService()
generator_service = GeneratorService()
verifier_service = VerifierService()
safety_service = SafetyService()
dashboard_service = DashboardService()

# Operational features
query_cache = SimpleCache(ttl_seconds=300)
health_checker = VectorStoreHealthCheck(vector_store)


class QuestionRequest(BaseModel):
    question: str
    top_k: int = SETTINGS.top_k
    similarity_threshold: float = SETTINGS.similarity_threshold


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok", "service": "mediscribe"}


@app.get("/health/vector-store")
def health_vector_store() -> Dict[str, Any]:
    try:
        endpoint = "/health/vector-store"
        RequestLogger.log_request(endpoint, "GET", {})
        health_status = health_checker.get_health()
        logger.info(f"Vector store health: {health_status['status']}")
        return health_status
    except Exception as e:
        logger.error(f"Error checking vector store health: {e}", exc_info=True)
        return {"status": "error", "error": str(e)}


@app.get("/documents")
def list_documents(filename: Optional[str] = None, status: Optional[str] = None) -> Dict[str, Any]:
    try:
        endpoint = "/documents"
        RequestLogger.log_request(endpoint, "GET", {"filename": filename, "status": status})
        docs = ingest_service.list_documents()
        
        # Apply filters
        if filename:
            docs = [d for d in docs if filename.lower() in d.filename.lower()]
        if status:
            docs = [d for d in docs if d.ingestion_status == status]
        
        result = {
            "documents": [
                {
                    "document_id": doc.document_id,
                    "filename": doc.filename,
                    "source_path": doc.source_path,
                    "status": doc.ingestion_status,
                }
                for doc in docs
            ],
            "filters": {"filename": filename, "status": status},
            "count": len(docs),
        }
        logger.info(f"Listed {len(docs)} documents (filters: filename={filename}, status={status})")
        return result
    except Exception as e:
        logger.error(f"Error listing documents: {e}", exc_info=True)
        RequestLogger.log_error(endpoint, e, "list_operation")
        return {"error": "Failed to list documents", "status": "error"}


@app.get("/dashboard")
def dashboard_summary() -> Dict[str, Any]:
    docs = ingest_service.list_documents()
    provenance: Dict[str, List[Dict[str, Any]]] = {}
    for record in vector_store.records.values():
        provenance.setdefault(record.source_document, []).append({
            "chunk_id": record.chunk_id,
            "page_number": 1,
            "source_file": record.source_document,
        })

    return dashboard_service.get_report(
        metrics={
            "document_count": len(docs),
            "chunk_count": len(vector_store.records),
            "degraded_mode": False,
        },
        documents=docs,
        provenance=provenance,
    )


@app.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)) -> Dict[str, Any]:
    start_time = time.time()
    endpoint = "/documents/upload"
    
    try:
        RequestLogger.log_request(endpoint, "POST", {"file": file.filename})
        
        if not file.filename:
            raise ValidationError("Filename is required")
        
        if file.size and file.size > 50 * 1024 * 1024:  # 50MB limit
            raise ValidationError("File too large (max 50MB)")
        
        content = await file.read()
        metadata = {
            "filename": file.filename or "unknown.txt",
            "source_path": ".",
        }
        logger.info(f"Processing file: {file.filename} ({len(content)} bytes)")
        
        docs = ingest_service.process_file(content, metadata)
        doc = docs[0]
        result = await _index_document(doc, content)
        
        duration_ms = (time.time() - start_time) * 1000
        RequestLogger.log_response(endpoint, 200, duration_ms)
        logger.info(f"File uploaded: {file.filename} -> {doc.document_id}")
        
        return result
    except ValidationError as e:
        logger.warning(f"Validation error in {endpoint}: {e}")
        RequestLogger.log_error(endpoint, e, f"file={file.filename}")
        return {"error": str(e), "status": "validation_failed"}
    except Exception as e:
        logger.error(f"Error uploading file: {e}", exc_info=True)
        RequestLogger.log_error(endpoint, e, f"file={file.filename}")
        return {"error": "Upload failed", "status": "error"}


@app.post("/documents/upload-multiple")
async def upload_multiple_documents(files: List[UploadFile] = File(...)) -> Dict[str, Any]:
    results = []
    for file in files:
        content = await file.read()
        metadata = {"filename": file.filename or "unknown.txt", "source_path": "."}
        docs = ingest_service.process_file(content, metadata)
        result = await _index_document(docs[0], content)
        results.append(result)
    return {"documents": results}


async def _index_document(doc: Document, content: bytes) -> Dict[str, Any]:
    extracted_pages = extractor_service.extract_text(doc.document_id, content.decode("utf-8", errors="ignore"))
    normalized_pages = normalizer_service.normalize(extracted_pages)
    chunks = chunking_service.chunk(normalized_pages, doc.document_id, doc.filename)

    for chunk in chunks:
        embedding = embedding_service.generate(chunk.text, chunk.chunk_id)
        vector_record = VectorRecord(
            vector_id=f"vector_{chunk.chunk_id}",
            chunk_id=chunk.chunk_id,
            embedding=embedding.vector,
            metadata_ref=chunk.chunk_id,
            source_document=chunk.source_file,
            similarity_score=0.0,
            chunk_text=chunk.text,
        )
        vector_store.save(vector_record)

    return {
        "document_id": doc.document_id,
        "filename": doc.filename,
        "status": doc.ingestion_status,
        "chunk_count": len(chunks),
    }


@app.post("/query")
def query_documents(payload: QuestionRequest) -> Dict[str, Any]:
    start_time = time.time()
    endpoint = "/query"
    
    try:
        # Validate input
        RequestLogger.log_request(endpoint, "POST", {"question": payload.question[:50]})
        question = validate_input(payload.question, "question", min_length=3, max_length=1000)
        logger.info(f"Query validated: {len(question)} chars")
        
        # Check cache
        import hashlib
        cache_key = hashlib.md5(f"{payload.question}:{payload.top_k}:{payload.similarity_threshold}".encode()).hexdigest()
        cached_result = query_cache.get(cache_key)
        if cached_result:
            logger.info(f"Cache hit for query: {cache_key}")
            return cached_result
        
        # Process query
        query_vector = embedding_service.generate(payload.question, "query_input").vector
        q = SearchQuery(
            query_text=payload.question,
            top_k=payload.top_k,
            similarity_threshold=payload.similarity_threshold,
        )
        search_results = retrieval_engine.search(q, query_vector)
        evidence = selector_service.select(search_results, threshold=payload.similarity_threshold)
        answer = generator_service.generate(payload.question, evidence)
        citations = verifier_service.verify(answer)
        answer.citations = citations
        safety = safety_service.evaluate(answer)
        
        result = {
            "answer": answer.answer_text,
            "citations": [
                {
                    "source_document": citation.source_document,
                    "page_number": citation.page_number,
                    "chunk_id": citation.chunk_id,
                    "snippet": citation.snippet,
                }
                for citation in citations
            ],
            "evidence_count": len(evidence.selected_chunks),
            "safety": {
                "decision": safety.decision,
                "reasoning": safety.reasoning,
                "classification": safety.classification,
            },
        }
        
        # Cache result
        query_cache.set(cache_key, result)
        
        duration_ms = (time.time() - start_time) * 1000
        RequestLogger.log_response(endpoint, 200, duration_ms)
        logger.info(f"Query processed: {len(citations)} citations, safety={safety.decision}")

        return result
    except ValidationError as e:
        logger.warning(f"Validation error in {endpoint}: {e}")
        RequestLogger.log_error(endpoint, e, "input_validation")
        return {"error": str(e), "status": "validation_failed"}
    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        RequestLogger.log_error(endpoint, e, "query_processing")
        return {"error": "Query processing failed", "status": "error"}


@app.get("/metrics")
def get_metrics() -> Dict[str, Any]:
    """Return operational metrics."""
    try:
        endpoint = "/metrics"
        RequestLogger.log_request(endpoint, "GET", {})
        
        docs = ingest_service.list_documents()
        vector_health = health_checker.get_health()
        
        return {
            "documents": len(docs),
            "chunks": len(vector_store.records),
            "cache": {
                "size": query_cache.size(),
                "ttl_seconds": query_cache.ttl_seconds,
            },
            "vector_store": vector_health,
        }
    except Exception as e:
        logger.error(f"Error getting metrics: {e}", exc_info=True)
        return {"error": "Failed to get metrics", "status": "error"}


@app.post("/cache/clear")
def clear_cache() -> Dict[str, Any]:
    """Clear the query result cache."""
    try:
        endpoint = "/cache/clear"
        RequestLogger.log_request(endpoint, "POST", {})
        query_cache.clear()
        logger.info("Query cache cleared")
        return {"status": "cleared", "message": "Query cache has been cleared"}
    except Exception as e:
        logger.error(f"Error clearing cache: {e}", exc_info=True)
        return {"error": "Failed to clear cache", "status": "error"}
