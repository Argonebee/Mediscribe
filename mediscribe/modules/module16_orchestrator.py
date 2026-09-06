from __future__ import annotations

from typing import Any, Dict, List

from mediscribe.contracts import SearchQuery
from mediscribe.modules.module10_selector import SelectorService
from mediscribe.modules.module11_generator import GeneratorService
from mediscribe.modules.module12_verifier import VerifierService
from mediscribe.modules.module13_safety import SafetyService


class Orchestrator:
    def __init__(self, selector: SelectorService | None = None, generator: GeneratorService | None = None, verifier: VerifierService | None = None, safety: SafetyService | None = None):
        self.selector = selector or SelectorService()
        self.generator = generator or GeneratorService()
        self.verifier = verifier or VerifierService()
        self.safety = safety or SafetyService()

    def answer_question(self, question: str, search_results: List[Any], threshold: float = 0.20) -> Dict[str, Any]:
        evidence = self.selector.select(search_results, threshold)
        answer = self.generator.generate(question, evidence)
        citations = self.verifier.verify(answer)
        answer.citations = citations
        safety = self.safety.evaluate(answer)
        return {
            "answer": answer.answer_text,
            "citations": citations,
            "evidence_count": len(evidence.selected_chunks),
            "safety": {
                "decision": safety.decision,
                "reasoning": safety.reasoning,
                "classification": safety.classification,
            },
        }
