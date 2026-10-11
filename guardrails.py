"""Lightweight, fail-safe guardrails for a medical-information RAG application.

No external dependencies. Designed for LangChain/Gradio integration.
This is a safety aid, not a clinical safety certification.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import re
from typing import Iterable, Mapping, Any

MAX_MESSAGE_LENGTH = 4000
MAX_OUTPUT_LENGTH = 24000
ALLOWED_VERDICTS = frozenset({
    "Supported", "Partially Supported", "Not Supported",
    "Contradicted", "Insufficient Evidence",
})

class Risk(str, Enum):
    LOW = "low"
    REVIEW = "review"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass(frozen=True)
class InputDecision:
    allowed: bool
    risk: Risk
    message: str
    reasons: tuple[str, ...] = ()
    requires_buffering: bool = True  # Default safe: do not stream unvalidated medical content
    requires_human_or_model_review: bool = False

@dataclass(frozen=True)
class OutputDecision:
    allowed: bool
    reasons: tuple[str, ...] = ()

CRISIS_REPLY = (
    "If you or someone else is in immediate danger in Australia, call 000. "
    "For urgent mental health support, contact your local mental health crisis service "
    "or Lifeline on 13 11 14. If you're outside Australia, contact local emergency services. "
    "I can help with general information, but I cannot assess an emergency here.\n\n"
)
SAFE_FALLBACK = (
    "I couldn't safely verify this assessment. Please rephrase your question or "
    "consult a qualified healthcare professional for personal treatment advice.\n\n"
)

# Patterns are triage signals, not diagnoses or exhaustive threat detection.
_CRISIS = [
    re.compile(p, re.I) for p in (
        r"\b(?:i (?:am|want to|plan to|intend to) (?:going to )?(?:kill myself|end my life)|suicide plan)\b",
        r"\b(?:i(?:'m| am) about to (?:hurt|harm|kill) (?:myself|someone)|kill someone (?:today|now))\b",
    )
]
_MEDICATION = re.compile(
    r"\b(?:stop|quit|skip|reduce|halve|taper|increase|double|change|discontinue)\b"
    r".{0,90}\b(?:medication|meds|medicine|dose|antipsychotic|olanzapine|clozapine|risperidone|aripiprazole|quetiapine)\b"
    r"|\b(?:antipsychotic|olanzapine|clozapine|risperidone|aripiprazole|quetiapine)\b"
    r".{0,90}\b(?:stop|quit|skip|reduce|halve|taper|increase|double|change|discontinue)\b",
    re.I | re.S,
)
_PERSONAL = re.compile(r"\b(?:i|my|me|myself|should i|can i|for me)\b", re.I)
_INJECTION = re.compile(
    r"(?:ignore (?:all |your )?(?:previous|system|developer) (?:instructions|prompts)|"
    r"reveal (?:your |the )?(?:system|developer) prompt|"
    r"(?:disable|bypass) (?:all |your )?(?:safety|guardrails)|"
    r"\b(?:system|developer)\s*:\s*(?:override|ignore))",
    re.I,
)
_UNSAFE_OUTPUT = re.compile(
    r"\b(?:stop|quit|skip|halve|double|discontinue)\b.{0,65}"
    r"\b(?:your|prescribed)\b.{0,35}"
    r"\b(?:medication|medicine|dose|antipsychotic|olanzapine|clozapine|risperidone)\b",
    re.I | re.S,
)
_CITATION = re.compile(r"\[source:([A-Za-z0-9_.:-]{1,100})\]")


def validate_input(message: str) -> InputDecision:
    """Return a decision without invoking an LLM or network service."""
    if not isinstance(message, str):
        return InputDecision(False, Risk.REVIEW, "", ("non_string_input",))
    cleaned = message.strip()
    if len(cleaned) < 3 or len(cleaned) > MAX_MESSAGE_LENGTH:
        return InputDecision(False, Risk.REVIEW, cleaned, ("invalid_length",))
    if any(ord(ch) < 32 and ch not in "\n\r\t" for ch in cleaned):
        return InputDecision(False, Risk.REVIEW, cleaned, ("control_character",))
    if any(p.search(cleaned) for p in _CRISIS):
        return InputDecision(False, Risk.CRITICAL, cleaned, ("possible_immediate_danger",), True, True)
    reasons = []
    if _INJECTION.search(cleaned):
        reasons.append("possible_prompt_injection")
    if _MEDICATION.search(cleaned) and _PERSONAL.search(cleaned):
        reasons.append("personal_medication_change")
    if reasons:
        return InputDecision(True, Risk.HIGH, cleaned, tuple(reasons), True, True)
    # Even apparently low-risk medical answers should be buffered until checked.
    return InputDecision(True, Risk.LOW, cleaned, (), True, False)


def validate_evidence(evidence: Iterable[Mapping[str, Any]], max_chunks: int = 10,
                      max_chunk_chars: int = 6000) -> list[dict[str, Any]]:
    """Accept only bounded, well-formed evidence; never treat content as instructions.

    The caller must supply IDs from its own trusted retrieval process.
    """
    cleaned = []
    seen = set()
    for item in evidence:
        if len(cleaned) >= max_chunks:
            break
        if not isinstance(item, Mapping):
            continue
        identifier, content = item.get("id"), item.get("content")
        if not isinstance(identifier, str) or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,100}", identifier):
            continue
        if not isinstance(content, str) or not content.strip() or identifier in seen:
            continue
        seen.add(identifier)
        cleaned.append({
            "id": identifier,
            "content": content[:max_chunk_chars],
            "source": str(item.get("source") or "unknown")[:300],
            "source_url": item.get("source_url") if isinstance(item.get("source_url"), str) and item["source_url"].startswith("https://") else None,
            "source_type": str(item.get("source_type") or "unverified_transcript")[:80],
            "verified": item.get("verified") is True,  # Must be set by trusted ingestion, never the LLM.
        })
    return cleaned


def validate_output(answer: str, *, allowed_source_ids: Iterable[str],
                    verdict: str | None = None, require_citation: bool = True,
                    input_decision: InputDecision | None = None) -> OutputDecision:
    """Fast checks only. Cannot establish factual accuracy or clinical safety."""
    errors = []
    if not isinstance(answer, str) or not answer.strip() or len(answer) > MAX_OUTPUT_LENGTH:
        return OutputDecision(False, ("invalid_output",))
    allowed = set(allowed_source_ids)
    cited = set(_CITATION.findall(answer))
    if cited - allowed:
        errors.append("unknown_citation_id")
    if require_citation and not cited:
        errors.append("missing_citations")
    if verdict is not None and verdict not in ALLOWED_VERDICTS:
        errors.append("invalid_verdict")
    if _UNSAFE_OUTPUT.search(answer):
        errors.append("possible_unsupervised_medication_instruction")
    if input_decision and input_decision.risk in (Risk.HIGH, Risk.CRITICAL):
        errors.append("high_risk_requires_independent_review")
    return OutputDecision(not errors, tuple(errors))


def safe_response(decision: InputDecision | OutputDecision) -> str:
    if isinstance(decision, InputDecision) and decision.risk is Risk.CRITICAL:
        return CRISIS_REPLY
    return SAFE_FALLBACK


def needs_review(decision: InputDecision) -> bool:
    """Whether to route to a separate safety/clinical review policy."""
    return decision.requires_human_or_model_review
