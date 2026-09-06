from fastapi.testclient import TestClient

from mediscribe.api import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_upload_and_query_demo():
    uploaded = client.post(
        "/documents/upload",
        files={"file": ("sample.txt", b"Dose is 12.5 mg once daily. Contraindication: avoid in severe renal failure.")},
    )
    assert uploaded.status_code == 200
    payload = {"question": "What is the dose?", "top_k": 4, "similarity_threshold": 0.20}
    response = client.post("/query", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert "answer" in body
    assert "safety" in body


def test_list_documents_endpoint():
    client.post(
        "/documents/upload",
        files={"file": ("listed_sample.txt", b"Dose is 12.5 mg once daily.")},
    )
    response = client.get("/documents")
    assert response.status_code == 200
    body = response.json()
    assert any(item["filename"] == "listed_sample.txt" for item in body["documents"])


def test_dashboard_summary_endpoint():
    client.post(
        "/documents/upload",
        files={"file": ("dashboard_sample.txt", b"Dose is 12.5 mg once daily.")},
    )
    response = client.get("/dashboard")
    assert response.status_code == 200
    body = response.json()
    assert body["document_count"] >= 1
    assert "documents" in body
