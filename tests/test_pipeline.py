import os
import shutil
import tempfile

import pytest

from mediscribe.config import Settings
from mediscribe.contracts import Citation, DocumentChunk, SearchResult, VectorRecord
from mediscribe.modules.module1_ingestion import FileStorage, IngestionService
from mediscribe.modules.module2_extractor import ExtractorService
from mediscribe.modules.module6_embedding import EmbeddingService, MockEmbeddingProvider
from mediscribe.modules.module7_api_keys import APIKeyManager
from mediscribe.modules.module8_vector_store import VectorStore
from mediscribe.modules.module3_normalizer import NormalizerService
from mediscribe.modules.module4_chunking import ChunkingService
from mediscribe.modules.module10_selector import SelectorService
from mediscribe.modules.module11_generator import GeneratorService
from mediscribe.modules.module13_safety import SafetyService
from mediscribe.modules.module14_inspector import EvidenceInspector
from mediscribe.modules.module15_dashboard import DashboardService
from mediscribe.modules.module5_provenance import ProvenanceService
from mediscribe.ui_app import build_backend_url, normalize_api_response


def test_ingestion_service_success():
    service = IngestionService()
    result = service.process_file(b"hello", {"filename": "sample.txt", "source_path": "."})
    assert result[0].ingestion_status == "SUCCESS"


def test_extractor_keeps_pages():
    service = ExtractorService()
    pages = service.extract_text("doc_1", "page one\n\n---\n\npage two")
    assert len(pages) == 2
    assert pages[0].page_number == 1


def test_normalizer_cleans_whitespace():
    service = NormalizerService()
    pages = [{"page_number": 1, "text_content": "This    is \\n often  messy"}]
    result = service.normalize([type("Page", (), pages[0])()])
    assert "messy" in result[0].text_content


def test_chunking_keeps_dose_text_together():
    service = ChunkingService(max_chars=200)
    pages = [{"page_number": 1, "text_content": "Dose is 12.5 mg once daily. Contraindication: avoid in severe renal failure."}]
    result = service.chunk([type("Page", (), pages[0])()], "doc_1", "sample.txt")
    assert len(result) >= 1
    assert "12.5 mg" in result[0].text
    assert "Contraindication" in result[0].text


def test_chunking_preserves_medical_sentence_pairs():
    service = ChunkingService(max_chars=50)
    pages = [{"page_number": 1, "text_content": "Dose is 12.5 mg once daily. Contraindication: avoid in severe renal failure. Next recommendation: monitor blood pressure."}]
    result = service.chunk([type("Page", (), pages[0])()], "doc_2", "sample2.txt")
    assert any("12.5 mg" in chunk.text and "Contraindication" in chunk.text for chunk in result)
    assert len(result) >= 2


def test_selector_and_generator_are_grounded():
    selector = SelectorService()
    results = [
        SearchResult(
            chunk_id="chunk_01",
            document_id="doc_01",
            page_number=1,
            section="Dose",
            similarity_score=0.91,
            source_document="sample.txt",
            matched_text="Dose is 12.5 mg once daily.",
        )
    ]
    evidence = selector.select(results, threshold=0.20)
    answer = GeneratorService().generate("What is the dose?", evidence)
    assert "12.5 mg" in answer.answer_text
    assert answer.safety_status == "PASS"


def test_retrieval_returns_real_match_text():
    from mediscribe.modules.module8_vector_store import VectorStore
    from mediscribe.modules.module9_retrieval import RetrievalEngine
    from mediscribe.contracts import SearchQuery, VectorRecord

    vector_store = VectorStore(storage_dir=".tmp_test_vectors")
    record = VectorRecord(
        vector_id="vec_1",
        chunk_id="chunk_doc_1_1",
        embedding=[1.0, 0.0, 0.5],
        metadata_ref="chunk_doc_1_1",
        source_document="sample.txt",
        similarity_score=0.91,
    )
    vector_store.save(record)
    result = RetrievalEngine(vector_store).search(SearchQuery(query_text="dose", top_k=1, similarity_threshold=0.0), [1.0, 0.0, 0.5])
    assert result and result[0].matched_text != "matched evidence"
    assert result[0].document_id == "doc_1"


def test_evidence_inspector_renders_citation_details():
    inspector = EvidenceInspector()
    answer = type(
        "Answer",
        (),
        {
            "answer_text": "Dose is 12.5 mg once daily.",
            "citations": [
                Citation(
                    source_document="sample.txt",
                    page_number=1,
                    chunk_id="chunk_001",
                    snippet="Dose is 12.5 mg once daily.",
                )
            ],
        },
    )()
    evidence = type(
        "Evidence",
        (),
        {"selected_chunks": [DocumentChunk("chunk_001", "doc_001", "Dose is 12.5 mg once daily.", 1, source_file="sample.txt")]},
    )()

    rendered = inspector.render(answer, evidence)
    assert rendered["evidence_count"] == 1
    assert rendered["citations"][0]["snippet"] == "Dose is 12.5 mg once daily."


def test_normalize_api_response_preserves_dashboard_metadata():
    payload = {
        "answer": "Dose is 12.5 mg once daily.",
        "citations": [{"source_document": "sample.txt", "page_number": 1, "chunk_id": "chunk_1", "snippet": "Dose is 12.5 mg once daily."}],
        "document_count": 1,
        "documents": [{"filename": "sample.txt", "status": "SUCCESS"}],
        "provenance": {"doc_1": [{"chunk_id": "chunk_1", "page_number": 1, "source_file": "sample.txt"}]},
        "degraded_mode": False,
        "safety": {"decision": "PASS", "reasoning": "Supported by evidence.", "classification": "SUPPORTED"},
    }
    normalized = normalize_api_response(payload)
    assert normalized["document_count"] == 1
    assert normalized["documents"][0]["filename"] == "sample.txt"
    assert normalized["provenance"]["doc_1"][0]["chunk_id"] == "chunk_1"


def test_api_key_manager_supports_provider_specific_keys():
    manager = APIKeyManager(keys_by_provider={
        "openai": ["openai-key-1", "openai-key-2"],
        "azure": ["azure-key-1"],
    })
    assert manager.get_active_key("openai") == "openai-key-1"
    assert manager.rotate_key("azure") == "azure-key-1"

    settings = Settings(provider_name="azure", api_key="from-env-key")
    assert settings.provider_name == "azure"
    assert settings.api_key == "from-env-key"


def test_embedding_service_defaults_to_mock_provider():
    service = EmbeddingService(use_real=False)
    embedding = service.generate("test text", "chunk_1")
    assert embedding.model_name == "mock-embedding-provider"
    assert len(embedding.vector) == 3072
    assert embedding.dimension == 3072


def test_generator_service_defaults_to_mock_provider():
    from mediscribe.modules.module11_generator import GeneratorService
    service = GeneratorService(use_real=False)
    evidence = type("Evidence", (), {
        "selected_chunks": [type("Chunk", (), {"text": "Dose is 12.5 mg"})()],
        "citations": [],
        "confidence_score": 0.9,
    })()
    answer = service.generate("What is the dose?", evidence)
    assert "12.5 mg" in answer.answer_text
    assert answer.safety_status == "PASS"


def test_safety_service_detects_contraindication_warnings():
    from mediscribe.modules.module13_safety import SafetyService
    service = SafetyService()
    answer = type("Answer", (), {
        "answer_text": "This drug has a serious contraindication in renal failure.",
        "citations": [type("Citation", (), {"source_document": "test.txt"})()],
        "confidence": 0.95,
        "safety_status": "PASS",
    })()
    result = service.evaluate(answer)
    assert result.decision == "PASS"
    assert "CONTRAINDICATION_ALERT" in result.classification


def test_safety_service_detects_interaction_warnings():
    from mediscribe.modules.module13_safety import SafetyService
    service = SafetyService()
    answer = type("Answer", (), {
        "answer_text": "This medication interacts with warfarin, requiring careful monitoring.",
        "citations": [type("Citation", (), {"source_document": "test.txt"})()],
        "confidence": 0.88,
        "safety_status": "PASS",
    })()
    result = service.evaluate(answer)
    assert result.decision == "PASS"
    assert "INTERACTION_ALERT" in result.classification


def test_safety_service_fails_on_low_confidence():
    from mediscribe.modules.module13_safety import SafetyService
    service = SafetyService()
    answer = type("Answer", (), {
        "answer_text": "Based on minimal evidence, the answer might be...",
        "citations": [type("Citation", (), {"source_document": "test.txt"})()],
        "confidence": 0.05,
        "safety_status": "PASS",
    })()
    result = service.evaluate(answer)
    assert result.decision == "FAIL"
    assert "LOW_CONFIDENCE" in result.classification


def test_build_backend_url_normalizes_base_and_path():
    assert build_backend_url("/query", "http://localhost:8000/") == "http://localhost:8000/query"
    assert build_backend_url("query", "http://localhost:8000") == "http://localhost:8000/query"


def test_extractor_supports_pdf_bytes():
    service = ExtractorService()
    pdf_bytes = b"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 200] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>
endobj
4 0 obj
<< /Length 54 >>
stream
BT
/F1 12 Tf
50 100 Td
(Dose is 12.5 mg once daily.) Tj
ET
endstream
endobj
5 0 obj
<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>
endobj
xref
0 6
0000000000 65535 f 
0000000010 00000 n 
0000000062 00000 n 
0000000123 00000 n 
0000000245 00000 n 
0000000421 00000 n 
trailer
<< /Root 1 0 R /Size 6 >>
startxref
500
%%EOF
"""
    pages = service.extract_text("doc_pdf", pdf_bytes)
    assert len(pages) >= 1
    assert "12.5 mg" in pages[0].text_content


def test_ui_normalizes_api_response():
    payload = {
        "answer": "Dose is 12.5 mg once daily.",
        "citations": [{
            "source_document": "sample.txt",
            "page_number": 1,
            "chunk_id": "chunk_001",
            "snippet": "Dose is 12.5 mg once daily.",
        }],
        "safety": {
            "decision": "PASS",
            "reasoning": "Supported by evidence.",
            "classification": "SUPPORTED",
        },
    }
    normalized = normalize_api_response(payload)
    assert normalized["answer"] == "Dose is 12.5 mg once daily."
    assert normalized["citations"][0]["snippet"] == "Dose is 12.5 mg once daily."


def test_ingestion_tracks_uploaded_documents():
    service = IngestionService()
    docs = service.process_file(b"dose 12.5 mg", {"filename": "tracked.txt", "source_path": "."})
    tracked = service.list_documents()
    assert any(doc.filename == "tracked.txt" for doc in tracked)
    assert any(doc.document_id == docs[0].document_id for doc in tracked)


def test_ingestion_persists_documents_across_instances():
    storage_dir = os.path.join(tempfile.gettempdir(), "mediscribe_persistence_test")
    if os.path.exists(storage_dir):
        shutil.rmtree(storage_dir)

    service_a = IngestionService(storage=FileStorage(storage_dir))
    service_a.process_file(b"persisted medical note", {"filename": "persisted.txt", "source_path": "."})

    service_b = IngestionService(storage=FileStorage(storage_dir))
    tracked = service_b.list_documents()
    assert any(doc.filename == "persisted.txt" for doc in tracked)


def test_vector_store_persists_records_across_instances():
    storage_dir = os.path.join(tempfile.gettempdir(), "mediscribe_vector_persistence_test")
    if os.path.exists(storage_dir):
        shutil.rmtree(storage_dir)

    store_a = VectorStore(storage_dir=storage_dir)
    store_a.save(VectorRecord(
        vector_id="vec_persist_1",
        chunk_id="chunk_persist_1",
        embedding=[1.0, 0.0, 0.5],
        metadata_ref="chunk_persist_1",
        source_document="persisted.txt",
        similarity_score=0.91,
        chunk_text="Dose is 12.5 mg once daily.",
    ))

    store_b = VectorStore(storage_dir=storage_dir)
    assert "vec_persist_1" in store_b.records
    assert store_b.records["vec_persist_1"].chunk_text == "Dose is 12.5 mg once daily."


def test_dashboard_reports_uploaded_documents():
    service = DashboardService()
    report = service.get_report(documents=[{"filename": "sample.txt", "status": "SUCCESS"}], metrics={"document_count": 1, "chunk_count": 2, "degraded_mode": False})
    assert report["document_count"] == 1
    assert report["documents"][0]["filename"] == "sample.txt"


def test_provenance_tracks_chunk_sources():
    service = ProvenanceService()
    chunk = DocumentChunk(
        chunk_id="chunk_01",
        document_id="doc_01",
        text="Dose is 12.5 mg once daily.",
        page_number=1,
        source_file="sample.txt",
        provenance={"page_number": 1, "source_file": "sample.txt"},
    )
    saved = service.save_chunk(chunk)
    tracked = service.list_chunks()
    assert saved.chunk_id == "chunk_01"
    assert tracked[0].provenance["source_file"] == "sample.txt"


def test_provenance_can_fetch_document_chunks():
    service = ProvenanceService()
    service.save_chunk(DocumentChunk("chunk_01", "doc_01", "Dose is 12.5 mg once daily.", 1, source_file="sample.txt", provenance={"page_number": 1, "source_file": "sample.txt"}))
    service.save_chunk(DocumentChunk("chunk_02", "doc_01", "Avoid in severe renal impairment.", 1, source_file="sample.txt", provenance={"page_number": 1, "source_file": "sample.txt"}))
    result = service.get_by_document("doc_01")
    assert len(result) == 2
    assert {chunk.chunk_id for chunk in result} == {"chunk_01", "chunk_02"}


def test_dashboard_summary_includes_provenance_and_metrics():
    service = DashboardService()
    report = service.get_report(
        metrics={"document_count": 2, "chunk_count": 5, "degraded_mode": False},
        documents=[
            {"document_id": "doc_1", "filename": "sample.txt", "status": "SUCCESS"},
            {"document_id": "doc_2", "filename": "sample2.txt", "status": "SUCCESS"},
        ],
        provenance={
            "doc_1": [{"chunk_id": "chunk_1", "page_number": 1, "source_file": "sample.txt"}],
            "doc_2": [{"chunk_id": "chunk_2", "page_number": 1, "source_file": "sample2.txt"}],
        },
    )
    assert report["document_count"] == 2
    assert report["provenance"]["doc_1"][0]["chunk_id"] == "chunk_1"
    assert report["documents"][0]["filename"] == "sample.txt"


def test_selector_prefers_highest_confidence_evidence():
    selector = SelectorService()
    results = [
        SearchResult("chunk_low", "doc_1", 1, "Dose", 0.21, "sample.txt", "Dose is 12.5 mg once daily."),
        SearchResult("chunk_high", "doc_2", 1, "Dose", 0.91, "sample2.txt", "Dose is 12.5 mg once daily."),
    ]
    evidence = selector.select(results, threshold=0.20)
    assert evidence.selected_chunks[0].chunk_id == "chunk_high"
    assert evidence.confidence_score == 0.91


def test_safety_rejects_missing_evidence():
    safety = SafetyService()
    answer = type("Ans", (), {"citations": [], "safety_status": "FAIL"})()
    decision = safety.evaluate(answer)
    assert decision.decision == "FAIL"


def test_safety_service_fails_on_insufficient_detail():
    from mediscribe.modules.module13_safety import SafetyService
    service = SafetyService()
    answer = type("Answer", (), {
        "answer_text": "Yes.",
        "citations": [type("Citation", (), {"source_document": "test.txt"})()],
        "confidence": 0.95,
        "safety_status": "PASS",
    })()
    result = service.evaluate(answer)
    assert result.decision == "FAIL"
    assert "INSUFFICIENT_DETAIL" in result.classification


def test_validate_input_rejects_empty_strings():
    from mediscribe.modules.module16_hardening import validate_input, ValidationError
    with pytest.raises(ValidationError):
        validate_input("", "test_field", min_length=1)


def test_validate_input_accepts_valid_text():
    from mediscribe.modules.module16_hardening import validate_input
    result = validate_input("This is a valid medical question?", "question", min_length=3, max_length=1000)
    assert len(result) > 0
    assert result == "This is a valid medical question?"


def test_validate_input_enforces_max_length():
    from mediscribe.modules.module16_hardening import validate_input, ValidationError
    long_text = "x" * 10001
    with pytest.raises(ValidationError):
        validate_input(long_text, "question", max_length=10000)


def test_request_logger_records_operations():
    from mediscribe.modules.module16_hardening import RequestLogger
    # Just verify logger doesn't crash
    RequestLogger.log_request("/query", "POST", {"question": "test"})
    RequestLogger.log_response("/query", 200, 10.5)
    RequestLogger.log_error("/query", ValueError("test"), "context")


def test_simple_cache_stores_and_retrieves():
    from mediscribe.modules.module17_operations import SimpleCache
    cache = SimpleCache(ttl_seconds=300)
    cache.set("key1", {"result": "value"})
    assert cache.get("key1") == {"result": "value"}
    assert cache.size() == 1
    cache.clear()
    assert cache.get("key1") is None


def test_simple_cache_respects_ttl():
    import time
    from mediscribe.modules.module17_operations import SimpleCache
    cache = SimpleCache(ttl_seconds=1)
    cache.set("key1", {"result": "value"})
    assert cache.get("key1") is not None
    time.sleep(1.1)
    assert cache.get("key1") is None


def test_vector_store_health_check_returns_status():
    from mediscribe.modules.module17_operations import VectorStoreHealthCheck
    from mediscribe.modules.module8_vector_store import VectorStore
    store = VectorStore()
    checker = VectorStoreHealthCheck(store)
    health = checker.get_health()
    assert "status" in health
    assert health["status"] in ["healthy", "unhealthy"]


def test_vector_store_health_check_warns_on_empty():
    from mediscribe.modules.module17_operations import VectorStoreHealthCheck
    from mediscribe.modules.module8_vector_store import VectorStore
    import tempfile
    import os
    
    storage_dir = os.path.join(tempfile.gettempdir(), "mediscribe_empty_health_test")
    if os.path.exists(storage_dir):
        import shutil
        shutil.rmtree(storage_dir)
    
    store = VectorStore(storage_dir=storage_dir)
    checker = VectorStoreHealthCheck(store)
    health = checker.get_health()
    assert "Vector store is empty" in health.get("warnings", [])
