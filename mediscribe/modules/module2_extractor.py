from __future__ import annotations

from typing import List

from mediscribe.contracts import DocumentPage


import re


class ExtractorService:
    def extract_text(self, document_id: str, text: str | bytes, pages: int | None = None) -> List[DocumentPage]:
        if isinstance(text, (bytes, bytearray)):
            decoded = self._decode_pdf_bytes(bytes(text))
            text = decoded

        cleaned = text.replace("\r\n", "\n").replace("\r", "\n")
        split_pages = cleaned.split("\n\n---\n\n") if "\n\n---\n\n" in cleaned else [cleaned]
        if pages is None:
            pages = len(split_pages)

        result: List[DocumentPage] = []
        for idx, page_text in enumerate(split_pages[:pages], start=1):
            cleaned_page = page_text.strip()
            if cleaned_page:
                result.append(DocumentPage(page_number=idx, text_content=cleaned_page))
        return result

    def _decode_pdf_bytes(self, payload: bytes) -> str:
        decoded = payload.decode("latin-1", errors="ignore")
        text_matches = re.findall(r"\((.*?)\)", decoded, flags=re.DOTALL)
        if text_matches:
            merged = "\n\n".join(m.replace("\\(", "(").replace("\\)", ")").replace("\\n", " ") for m in text_matches)
            if merged.strip():
                return merged.strip()
        return decoded.strip() or "PDF content could not be extracted."
