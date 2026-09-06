from __future__ import annotations

import re
from typing import List

from mediscribe.contracts import DocumentPage


class NormalizerService:
    def normalize(self, pages: List[DocumentPage]) -> List[DocumentPage]:
        normalized: List[DocumentPage] = []
        for page in pages:
            text = page.text_content
            text = text.replace("\t", " ")
            text = re.sub(r"\n{3,}", "\n\n", text)
            text = re.sub(r"(?<!\d)\s+\n\s+", " ", text)
            text = re.sub(r"\s{2,}", " ", text)
            text = text.strip()
            normalized.append(DocumentPage(page_number=page.page_number, text_content=text))
        return normalized
