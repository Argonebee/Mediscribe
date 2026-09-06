from __future__ import annotations

import os

from mediscribe.config import SETTINGS
from mediscribe.contracts import EvidenceSet, GeneratedAnswer


class MockGeneratorProvider:
    def generate(self, question: str, evidence_summary: str) -> str:
        if not evidence_summary:
            return "The provided medical literature does not contain sufficient evidence to answer this question."
        return f"Based on the available evidence, the answer is: {evidence_summary}. If the literature is incomplete, the system should state that the evidence is insufficient."


class OpenAIGeneratorProvider:
    def __init__(self, api_key: str | None = None, model: str = "gpt-4"):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package required for real LLM generation; install with: pip install openai")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", SETTINGS.api_key)
        self.model = model
        self.client = OpenAI(api_key=self.api_key)

    def generate(self, question: str, evidence_summary: str) -> str:
        if not evidence_summary:
            return "The provided medical literature does not contain sufficient evidence to answer this question."
        
        prompt = f"""You are a medical document intelligence assistant. Based on the following medical evidence, provide a clear, accurate answer to the question.

Question: {question}

Evidence from medical documents:
{evidence_summary}

Provide a concise, evidence-grounded answer. If the evidence is insufficient, state that clearly."""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=SETTINGS.llm_temperature,
                max_tokens=500,
            )
            return response.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"Failed to generate answer: {e}")


class GeneratorService:
    def __init__(self, provider: MockGeneratorProvider | OpenAIGeneratorProvider | None = None, use_real: bool | None = None):
        if use_real is None:
            use_real = SETTINGS.provider_name != "demo"
        
        if provider is not None:
            self.provider = provider
        elif use_real:
            self.provider = OpenAIGeneratorProvider()
        else:
            self.provider = MockGeneratorProvider()
        
        self.is_real = isinstance(self.provider, OpenAIGeneratorProvider)

    def generate(self, question: str, evidence: EvidenceSet) -> GeneratedAnswer:
        evidence_summary = "; ".join(chunk.text for chunk in evidence.selected_chunks[:3]) if evidence.selected_chunks else ""
        answer_text = self.provider.generate(question, evidence_summary)
        
        return GeneratedAnswer(
            answer_text=answer_text,
            citations=evidence.citations,
            confidence=evidence.confidence_score,
            safety_status="PASS" if evidence.selected_chunks else "FAIL",
        )
