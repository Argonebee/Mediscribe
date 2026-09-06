from __future__ import annotations

import os
from typing import Any, Dict, List

try:
    import streamlit as st
except ModuleNotFoundError:  # pragma: no cover - allows tests to import module safely
    st = None

try:
    import httpx
except ImportError:
    httpx = None


def build_backend_url(path: str, base_url: str = "http://localhost:8000") -> str:
    normalized_base = base_url.rstrip("/")
    normalized_path = path.strip().lstrip("/")
    return f"{normalized_base}/{normalized_path}" if normalized_path else normalized_base


def api_request(method: str, endpoint: str, base_url: str = "http://localhost:8000", **kwargs) -> Dict[str, Any] | None:
    """Make a synchronous API request to the backend."""
    if not httpx:
        return None
    url = build_backend_url(endpoint, base_url)
    try:
        with httpx.Client(timeout=30.0) as client:
            if method.upper() == "GET":
                response = client.get(url, **kwargs)
            elif method.upper() == "POST":
                response = client.post(url, **kwargs)
            else:
                return None
            response.raise_for_status()
            return response.json()
    except Exception as e:
        st.error(f"API request failed: {e}")
        return None


def normalize_api_response(payload: Dict[str, Any]) -> Dict[str, Any]:
    citations = payload.get("citations", []) or []
    normalized_citations: List[Dict[str, Any]] = []
    for citation in citations:
        if isinstance(citation, dict):
            normalized_citations.append({
                "source_document": citation.get("source_document"),
                "page_number": citation.get("page_number"),
                "chunk_id": citation.get("chunk_id"),
                "snippet": citation.get("snippet"),
            })
        else:
            normalized_citations.append({
                "source_document": getattr(citation, "source_document", None),
                "page_number": getattr(citation, "page_number", None),
                "chunk_id": getattr(citation, "chunk_id", None),
                "snippet": getattr(citation, "snippet", None),
            })

    documents = payload.get("documents", []) or []
    normalized_documents: List[Dict[str, Any]] = []
    for document in documents:
        if isinstance(document, dict):
            normalized_documents.append({
                "document_id": document.get("document_id"),
                "filename": document.get("filename"),
                "status": document.get("status"),
            })
        else:
            normalized_documents.append({
                "document_id": getattr(document, "document_id", None),
                "filename": getattr(document, "filename", None),
                "status": getattr(document, "status", None) or getattr(document, "ingestion_status", None),
            })

    provenance = payload.get("provenance", {}) or {}
    normalized_provenance = {}
    for key, entries in provenance.items():
        normalized_provenance[key] = [
            {
                "chunk_id": item.get("chunk_id"),
                "page_number": item.get("page_number"),
                "source_file": item.get("source_file"),
            }
            if isinstance(item, dict) else {
                "chunk_id": getattr(item, "chunk_id", None),
                "page_number": getattr(item, "page_number", None),
                "source_file": getattr(item, "source_file", None),
            }
            for item in (entries or [])
        ]

    return {
        "answer": payload.get("answer", ""),
        "citations": normalized_citations,
        "document_count": payload.get("document_count", len(normalized_documents)),
        "chunk_count": payload.get("chunk_count"),
        "degraded_mode": payload.get("degraded_mode", False),
        "documents": normalized_documents,
        "provenance": normalized_provenance,
        "safety": payload.get("safety", {}),
    }


if st is not None:
    st.set_page_config(page_title="Mediscribe", layout="wide")

    st.title("Mediscribe")
    st.caption("Medical document intelligence prototype")

    with st.sidebar:
        st.header("Configuration")
        backend_base = st.text_input("Backend URL", value="http://localhost:8000")
        st.divider()
        st.header("System")
        st.write("- Modular document ingestion")
        st.write("- Semantic chunking")
        st.write("- Evidence-based retrieval")
        st.write("- Safety-aware answer generation")

    col1, col2 = st.columns([2, 1])
    with col1:
        uploaded_files = st.file_uploader("Upload medical documents", accept_multiple_files=True)
        if uploaded_files and st.button("Upload Documents"):
            for file in uploaded_files:
                with st.spinner(f"Uploading {file.name}..."):
                    files = {"file": (file.name, file.read())}
                    result = api_request("POST", "/documents/upload", backend_base, files=files)
                    if result:
                        st.success(f"Uploaded: {file.name}")
                    else:
                        st.error(f"Failed to upload: {file.name}")

    with col2:
        if st.button("Refresh Dashboard"):
            with st.spinner("Loading dashboard..."):
                dashboard = api_request("GET", "/dashboard", backend_base)
                if dashboard:
                    st.metric("Documents", dashboard.get("document_count", 0))
                    st.metric("Chunks", dashboard.get("chunk_count", 0))

    st.divider()
    question = st.text_area("Ask a question about the uploaded literature")

    if st.button("Run Query"):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Processing query..."):
                response = api_request(
                    "POST",
                    "/query",
                    backend_base,
                    json={"question": question, "top_k": 4, "similarity_threshold": 0.20}
                )
                if response:
                    normalized = normalize_api_response(response)
                    st.subheader("Answer")
                    st.write(normalized["answer"])
                    
                    if normalized["citations"]:
                        st.subheader(f"Citations ({len(normalized['citations'])})")
                        for i, citation in enumerate(normalized["citations"], 1):
                            with st.expander(f"Citation {i}: {citation.get('source_document', 'Unknown')}"):
                                st.write(f"**Page:** {citation.get('page_number', 'N/A')}")
                                st.write(f"**Snippet:** {citation.get('snippet', 'N/A')}")
                    
                    safety = normalized.get("safety", {})
                    if safety.get("decision") == "PASS":
                        st.success(f"Safety: {safety.get('reasoning', 'Passed')}")
                    else:
                        st.warning(f"Safety: {safety.get('reasoning', 'Check failed')}")
                else:
                    st.error("Failed to process query.")

    with st.expander("Evidence Inspector"):
        if st.button("Load Evidence Details"):
            with st.spinner("Loading evidence..."):
                documents = api_request("GET", "/documents", backend_base)
                if documents:
                    st.json(documents)
