"""Evaluation metrics for analyst outputs."""

from __future__ import annotations

import re
from typing import Any


def _joined_text(parsed_response: dict[str, Any]) -> str:
    parts = [parsed_response.get("summary", "")]
    for key in ["key_insights", "patterns", "recommendations"]:
        parts.extend(parsed_response.get(key, []))
    return " ".join(str(part) for part in parts).lower()


def keyword_score(parsed_response: dict[str, Any], expected_keywords: list[str]) -> float:
    if not expected_keywords:
        return 0.0
    text = _joined_text(parsed_response)
    hits = sum(1 for keyword in expected_keywords if keyword.lower() in text)
    return hits / len(expected_keywords)


def recommendation_score(parsed_response: dict[str, Any]) -> float:
    recommendations = parsed_response.get("recommendations", [])
    if not recommendations:
        return 0.0
    action_terms = {"improve", "prioritize", "reduce", "increase", "target", "optimize", "focus"}
    coverage = min(len(recommendations) / 3, 1.0)
    actionability = 1.0 if any(any(term in rec.lower() for term in action_terms) for rec in recommendations) else 0.5
    return round(coverage * actionability, 4)


def completeness_score(parsed_response: dict[str, Any]) -> float:
    fields = ["summary", "key_insights", "patterns", "recommendations", "confidence"]
    available = 0
    for field in fields:
        value = parsed_response.get(field)
        if isinstance(value, list) and value:
            available += 1
        elif isinstance(value, str) and value.strip():
            available += 1
    return available / len(fields)


def groundedness_score(parsed_response: dict[str, Any], context: str) -> float:
    if not context:
        return 0.0
    text = _joined_text(parsed_response)
    tokens = [token for token in re.findall(r"[a-zA-Z0-9_]+", text) if len(token) > 3]
    if not tokens:
        return 0.0
    context_text = context.lower()
    overlaps = sum(1 for token in tokens if token in context_text)
    return min(overlaps / len(tokens), 1.0)
