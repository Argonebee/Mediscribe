from __future__ import annotations

from typing import List

from mediscribe.contracts import Citation, GeneratedAnswer


class VerifierService:
    def verify(self, answer: GeneratedAnswer) -> List[Citation]:
        if not answer.citations:
            return []
        return answer.citations
