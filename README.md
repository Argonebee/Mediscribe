# Mediscribe: Medical Document Intelligence Platform

A production-grade FastAPI backend and Streamlit frontend for evidence-grounded medical question answering with clinical safety gating.

**Status:** 40 passing tests | 17 modular components | Real OpenAI integration | Persistent storage

---

## Overview

Mediscribe enables medical professionals to upload clinical documents and receive evidence-grounded answers to medical questions. Every answer is:
- **Evidence-grounded:** Cites source documents with page numbers and snippets
- **Safety-checked:** Detects contraindications, drug interactions, and confidence thresholds
- **Persistent:** Documents and embeddings survive service restarts
- **Production-ready:** Comprehensive error handling, logging, caching, and monitoring

### Architecture

```
Document Upload
    ↓
Ingestion (FileStorage)
    ↓
Extraction (PDF/Text → Pages)
    ↓
Normalization (Whitespace cleanup)
    ↓
Chunking (Semantic sentence grouping)
    ↓
Embedding (OpenAI text-embedding-3-small)
    ↓
Vector Store (JSON-based persistence)
    ↓
Query → Retrieval → Selection → Generation → Verification → Safety
    ↓
Answer + Citations + Safety Status
```

---

## Features

### Core Capabilities
- **Document Ingestion:** Handles PDF, text, and markdown with deterministic document IDs
- **Semantic Chunking:** Preserves medical context with sentence-pair grouping
- **Dual Embeddings:** OpenAI real provider + deterministic mock for testing
- **Dual LLM:** OpenAI GPT-4 real provider + mock template generator
- **Evidence Grounding:** All answers cite specific source documents and page numbers
- **Provenance Tracking:** Full lineage from document → chunks → citations

### Safety & Clinical Rules
- **Contraindication Detection:** Identifies serious warnings (e.g., "avoid in renal failure")
- **Drug Interaction Detection:** Flags concurrent medication risks
- **Confidence Thresholding:** Rejects answers below similarity threshold
- **Answer Quality Checks:** Prevents trivial or hallucinated responses
- **Safety Classification:** Returns SUPPORTED, CONTRAINDICATION_ALERT, INTERACTION_ALERT, or LOW_CONFIDENCE

### Production Features
- **Error Handling:** Try/except on all endpoints with structured logging
- **Input Validation:** Sanitizes and validates all medical text inputs
- **Request Logging:** Full audit trail of requests, responses, and errors
- **Query Caching:** 5-minute TTL cache reduces redundant computation
- **Health Monitoring:** Vector store health checks with warnings
- **Document Filtering:** Search by filename or status on `/documents` endpoint
- **Metrics Dashboard:** Real-time operational metrics via `/metrics` endpoint
- **Persistent Storage:** File-based storage in `.mediscribe/storage/` survives restarts

### API Endpoints (9 total)

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | Service health check |
| `/health/vector-store` | GET | Vector store health & warnings |
| `/documents` | GET | List/filter documents (by filename, status) |
| `/documents/upload` | POST | Upload single medical document |
| `/documents/upload-multiple` | POST | Batch upload documents |
| `/dashboard` | GET | Operational metrics (documents, chunks, provenance) |
| `/query` | POST | Submit medical question (cached) |
| `/metrics` | GET | System metrics (docs, chunks, cache, vector store) |
| `/cache/clear` | POST | Flush query result cache |

---

## Quick Start

### Prerequisites
- **Python 3.13.11+**
- **Optional:** OPENAI_API_KEY for real embeddings/LLM (otherwise uses mock)

### Installation

1. Clone and install:
```bash
git clone <repo-url>
cd mediscribe
pip install -r requirements.txt
```

2. (Optional) Set OpenAI API key:
```bash
# Windows PowerShell
$env:OPENAI_API_KEY="sk-your-key-here"

# Linux/Mac
export OPENAI_API_KEY="sk-your-key-here"
```

### Running Backend

```bash
# Terminal 1: Start API server
python -m uvicorn mediscribe.api:app --reload --port 8000
```

- Backend: `http://localhost:8000`
- API Docs: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### Running Frontend

```bash
# Terminal 2: Start Streamlit UI
streamlit run mediscribe/ui_app.py
```

- Frontend: `http://localhost:8501`
- Configure backend URL in sidebar (default: `http://localhost:8000`)

### Running Tests

```bash
# Terminal 3: Run test suite
python -m pytest -q
```

**Result:** 40 passing tests covering:
- End-to-end pipeline (ingestion → retrieval → generation → safety)
- Persistence across service restarts
- Provider routing (mock/real)
- UI contracts and normalization
- Clinical safety rules
- Production error handling
- Caching and health monitoring

---

## Example Usage

### Upload a Document
```bash
curl -F "file=@dosage-guide.pdf" http://localhost:8000/documents/upload
```

### Ask a Medical Question
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the standard dose for metformin?",
    "top_k": 4,
    "similarity_threshold": 0.2
  }'
```

**Response:**
```json
{
  "answer": "The standard initial dose of metformin is 500 mg once or twice daily...",
  "citations": [
    {
      "source_document": "dosage-guide.pdf",
      "page_number": 12,
      "snippet": "Standard initial dose: 500 mg once or twice daily"
    }
  ],
  "evidence_count": 3,
  "safety": {
    "decision": "PASS",
    "reasoning": "The answer is tied to available evidence.",
    "classification": "SUPPORTED"
  }
}
```

### Get Metrics
```bash
curl http://localhost:8000/metrics
```

### Filter Documents
```bash
curl "http://localhost:8000/documents?filename=dosage&status=SUCCESS"
```

---

## 🏗️ Project Structure

```
mediscribe/
├── mediscribe/
│   ├── api.py                          # FastAPI app (9 endpoints)
│   ├── ui_app.py                       # Streamlit frontend
│   ├── config.py                       # Settings & provider config
│   ├── contracts.py                    # Pydantic models
│   └── modules/
│       ├── module1_ingestion.py        # Document intake & inventory
│       ├── module2_extractor.py        # PDF/text extraction
│       ├── module3_normalizer.py       # Whitespace cleanup
│       ├── module4_chunking.py         # Semantic sentence grouping
│       ├── module5_provenance.py       # Lineage tracking
│       ├── module6_embedding.py        # Embeddings (mock/OpenAI)
│       ├── module7_api_keys.py         # Per-provider key rotation
│       ├── module8_vector_store.py     # Vector persistence (JSON)
│       ├── module9_retrieval.py        # Similarity search
│       ├── module10_selector.py        # Evidence selection
│       ├── module11_generator.py       # LLM (mock/OpenAI)
│       ├── module12_verifier.py        # Citation extraction
│       ├── module13_safety.py          # Safety evaluation
│       ├── module14_inspector.py       # Evidence rendering
│       ├── module15_dashboard.py       # Metrics aggregation
│       ├── module16_hardening.py       # Error handling & validation
│       └── module17_operations.py      # Caching & health checks
├── tests/
│   └── test_pipeline.py                # 40 end-to-end tests
├── requirements.txt                    # Dependencies
├── README.md                           # This file
├── .gitignore                          # Git exclusions
└── .mediscribe/
    └── storage/                        # Runtime data (documents, vectors)
```

---

## Configuration

Edit `mediscribe/config.py`:

```python
class Settings:
    # LLM Configuration
    provider_name: str = "demo"           # "demo" or "openai"
    api_key: str = "demo-key"             # OPENAI_API_KEY if provider_name="openai"
    llm_temperature: float = 0.3          # Answer creativity (0.0 = deterministic)
    
    # Retrieval Configuration
    top_k: int = 4                        # Number of chunks to retrieve
    similarity_threshold: float = 0.20    # Minimum match confidence
    
    # Chunking Configuration
    max_chunk_chars: int = 500            # Max chars per chunk
    chunk_overlap: int = 100              # Overlap between chunks
```

### Provider Routing

**Use real OpenAI:**
```python
# Set environment variable before starting app
export OPENAI_API_KEY="sk-..."

# Or edit config.py
provider_name = "openai"
```

**Use mock (no API key needed):**
```python
provider_name = "demo"  # Default
```

Both providers have identical interfaces; switch seamlessly between mock (testing) and real (production).

---

## Testing

### Run All Tests
```bash
python -m pytest -q
```

### Run Specific Test Category
```bash
# Persistence tests
python -m pytest -q -k persist

# Safety tests
python -m pytest -q -k safety

# Provider tests
python -m pytest -q -k provider

# Caching tests
python -m pytest -q -k cache
```

### Test Coverage
- **Ingestion & Persistence:** Documents load from disk on startup
- **Extraction & Chunking:** PDF parsing and semantic grouping
- **Embeddings:** Mock deterministic + OpenAI real provider
- **Retrieval:** Similarity search with thresholding
- **Generation:** LLM mock templates + OpenAI real
- **Safety:** Contraindication/interaction detection, confidence checks
- **UI Contracts:** Response normalization and dashboard metadata
- **Error Handling:** Validation, exception handling, logging
- **Caching:** TTL-based query caching
- **Health:** Vector store health checks and metrics

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| fastapi | ≥0.115.0 | REST API framework |
| uvicorn | ≥0.30.0 | ASGI server |
| streamlit | ≥1.36.0 | Web frontend |
| httpx | ≥0.27.0 | Async HTTP client |
| pydantic | (bundled) | Request/response validation |
| openai | ≥1.0.0 | Real LLM & embedding provider |
| python-dotenv | ≥1.0.0 | Environment variable management |
| pytest | ≥8.0.0 | Testing framework |

---

## Safety & Clinical Considerations

⚠️ **This is a prototype.** For production use in clinical settings:
- Add audit logging for compliance (HIPAA, GDPR)
- Implement authentication/authorization
- Add rate limiting and DDoS protection
- Use HTTPS/TLS for all communications
- Add data retention and purge policies
- Implement clinical review workflows
- Add confidence scores to UI
- Consider regulatory approval requirements

---

## Architecture Highlights

### Modular Design (17 Components)
Each module is independently testable and can be swapped:
```python
from mediscribe.modules.module6_embedding import EmbeddingService
service = EmbeddingService(use_real=True)  # Real OpenAI
service = EmbeddingService(use_real=False) # Mock (testing)
```

### Provider Abstraction
Mock and real providers implement identical interfaces:
```python
class EmbeddingProvider:
    def embed(text: str) -> Embedding: ...

class MockEmbeddingProvider(EmbeddingProvider): ...
class OpenAIEmbeddingProvider(EmbeddingProvider): ...
```

### Persistent Vector Store
Vectors saved as JSON for durability:
```
.mediscribe/storage/
├── documents.json          # Document inventory
└── vector_*/
    ├── vec_1.json         # Vector record
    ├── vec_2.json
    ...
```

### Evidence-Grounded Pipeline
Every answer produces:
1. **Answer:** LLM-generated response
2. **Citations:** Source documents + page numbers
3. **Safety:** Contraindication/interaction status
4. **Provenance:** Full lineage from document → chunk → answer

---

## Troubleshooting

### "OPENAI_API_KEY not set"
- Ensure environment variable is set before starting the app
- Or use mock provider (default): `provider_name = "demo"`

### "Vector store is empty"
- Upload at least one document via `/documents/upload`
- Check health: `curl http://localhost:8000/health/vector-store`

### Tests fail with "permission denied"
- Ensure `.mediscribe/storage/` is writable
- Windows: Run PowerShell as Administrator
- Linux/Mac: `chmod 755 .mediscribe/storage/`

### Slow query responses
- Check cache: `curl http://localhost:8000/metrics`
- Clear cache: `curl -X POST http://localhost:8000/cache/clear`
- Reduce `top_k` in config (default: 4)

---

## Further Reading

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference)
- [pytest Testing Framework](https://docs.pytest.org/)

---

## License

MIT License - See LICENSE file for details.

---

## Contributing

Pull requests welcome. Please ensure:
1. All 40 tests pass: `pytest -q`
2. Code follows PEP 8 style
3. New features include regression tests
4. Error handling uses try/except + logging

---

**Built with ❤️ for medical document intelligence.**
