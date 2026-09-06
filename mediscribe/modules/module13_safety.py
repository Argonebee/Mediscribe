from __future__ import annotations

import re

from mediscribe.config import SETTINGS
from mediscribe.contracts import GeneratedAnswer, SafetyDecision


class SafetyService:
    # Clinical warning keywords that require explicit handling
    CONTRAINDICATION_KEYWORDS = [
        "contraindication", "contraindicated", "do not use", "avoid",
        "not recommended", "warning", "serious risk", "severe", "potentially fatal"
    ]
    
    INTERACTION_KEYWORDS = [
        "interaction", "interacts with", "combined with", "concomitant",
        "concurrent use", "should not be taken with", "incompatible"
    ]
    
    CONFIDENCE_THRESHOLD = SETTINGS.similarity_threshold

    def evaluate(self, answer: GeneratedAnswer) -> SafetyDecision:
        # Check 1: Evidence presence
        if not answer.citations:
            return SafetyDecision(
                decision="FAIL",
                reasoning="No supported evidence was found for this answer.",
                classification="INSUFFICIENT_EVIDENCE",
            )
        
        # Check 2: Safety status marker
        if answer.safety_status == "FAIL":
            return SafetyDecision(
                decision="FAIL",
                reasoning="The model output was marked as unsupported.",
                classification="INSUFFICIENT_EVIDENCE",
            )
        
        # Check 3: Confidence score thresholding
        if answer.confidence is not None and answer.confidence < self.CONFIDENCE_THRESHOLD:
            return SafetyDecision(
                decision="FAIL",
                reasoning=f"Confidence score {answer.confidence:.2f} is below required threshold {self.CONFIDENCE_THRESHOLD}.",
                classification="LOW_CONFIDENCE",
            )
        
        # Check 4: Answer length validation (catch hallucinations or trivial responses)
        answer_length = len(answer.answer_text.strip())
        if answer_length < 10:
            return SafetyDecision(
                decision="FAIL",
                reasoning="Answer is too brief; may not have sufficient supporting evidence.",
                classification="INSUFFICIENT_DETAIL",
            )
        
        # Check 5: Contraindication detection
        if self._contains_contraindication_warning(answer.answer_text):
            return SafetyDecision(
                decision="PASS",
                reasoning="Answer contains explicit contraindication warning; requires clinical review.",
                classification="CONTRAINDICATION_ALERT",
            )
        
        # Check 6: Interaction detection
        if self._contains_interaction_warning(answer.answer_text):
            return SafetyDecision(
                decision="PASS",
                reasoning="Answer contains interaction warning; pharmacist review recommended.",
                classification="INTERACTION_ALERT",
            )
        
        # Default pass
        return SafetyDecision(
            decision="PASS",
            reasoning="The answer is tied to available evidence and remains transparent about limitations.",
            classification="SUPPORTED",
        )
    
    def _contains_contraindication_warning(self, text: str) -> bool:
        """Check if answer contains contraindication keywords."""
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in self.CONTRAINDICATION_KEYWORDS)
    
    def _contains_interaction_warning(self, text: str) -> bool:
        """Check if answer contains interaction keywords."""
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in self.INTERACTION_KEYWORDS)
