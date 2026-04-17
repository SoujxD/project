"""Data analyst RAG agent."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from agents.eda_agent import EDAAgent
from utils.dataset_adapter import load_analysis_dataset
from utils.llm_client import LLMClient
from utils.parser import extract_json
from utils.retriever import EcommerceRetriever, RetrievalResult


PROMPT_STYLES = {
    "basic": (
        "You are a senior business analyst. Answer the question using the provided dataset context when available. "
        "Be concise, practical, and honest about uncertainty."
    ),
    "structured_json": (
        "You are a rigorous analytics agent. Return valid JSON only and organize the answer into summary, key insights, "
        "patterns, recommendations, and confidence."
    ),
    "executive": (
        "You are preparing a short executive briefing for business stakeholders. Focus on decision-ready findings, "
        "commercial implications, and next steps."
    ),
    "evidence_constrained": (
        "Use only the supplied dataset evidence. If support is weak, say so clearly. Avoid claims that cannot be tied "
        "to the retrieved context."
    ),
}


DEFAULT_MODELS = [
    "meta-llama/llama-3.1-8b-instruct",
    "mistralai/mistral-7b-instruct",
    "google/gemma-2-9b-it",
    "qwen/qwen-2.5-7b-instruct",
]


@dataclass(slots=True)
class AnalystAgentResult:
    """Single analyst response payload."""

    question: str
    model: str
    prompt_style: str
    rag_enabled: bool
    raw_response: str
    parsed_response: dict[str, Any]
    retrieved_context: str
    retrieval_results: list[RetrievalResult]


class AnalystRAGAgent:
    """RAG-enabled business analyst for the e-commerce dataset."""

    def __init__(
        self,
        dataset_path: str | Path,
        llm_client: LLMClient | None = None,
        models: list[str] | None = None,
    ) -> None:
        self.dataset_path = Path(dataset_path)
        adapted = load_analysis_dataset(self.dataset_path)
        self.dataframe = adapted.dataframe
        output_dir = Path(__file__).resolve().parents[1] / "outputs"
        self.eda_agent = EDAAgent(output_dir=output_dir)
        cache_key = self._dataset_cache_key()
        self.eda_report = self.eda_agent.analyze_dataset(self.dataset_path, include_charts=False, cache_key=cache_key)
        self.retriever = EcommerceRetriever(self.dataframe)
        self.llm_client = llm_client or LLMClient()
        self.models = models or DEFAULT_MODELS

    def _dataset_cache_key(self) -> str:
        digest = hashlib.sha256()
        with self.dataset_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()[:16]

    def available_models(self) -> list[str]:
        return self.models

    def available_prompt_styles(self) -> list[str]:
        return list(PROMPT_STYLES)

    def _build_prompt(self, question: str, prompt_style: str, context: str, rag_enabled: bool) -> str:
        instruction = PROMPT_STYLES[prompt_style]
        context_block = context if rag_enabled and context else "RAG disabled. Respond from general statistical reasoning only."
        eda_handoff = self.eda_report.get("handoff_summary", {})
        eda_chunks = "\n".join(self.eda_report.get("retrieval_chunks", [])[:8])
        return f"""
{instruction}

Prompt style:
{prompt_style}

Question:
{question}

EDA handoff:
{eda_handoff}

EDA evidence:
{eda_chunks}

Dataset context:
{context_block}

Return JSON in exactly this schema:
{{
  "summary": "short paragraph",
  "key_insights": ["insight 1", "insight 2"],
  "patterns": ["pattern 1", "pattern 2"],
  "recommendations": ["recommendation 1", "recommendation 2"],
  "confidence": "high"
}}
""".strip()

    def answer_question(
        self,
        question: str,
        model: str,
        prompt_style: str = "structured_json",
        rag_enabled: bool = True,
        top_k: int = 5,
    ) -> AnalystAgentResult:
        """Generate an analyst answer and include retrieval traces."""
        if prompt_style not in PROMPT_STYLES:
            raise ValueError(f"Unsupported prompt style: {prompt_style}")

        context = ""
        retrieval_results: list[RetrievalResult] = []
        if rag_enabled:
            context, retrieval_results = self.retriever.build_context(question, top_k=top_k)

        prompt = self._build_prompt(question, prompt_style, context, rag_enabled)
        llm_response = self.llm_client.generate(prompt=prompt, model=model)
        parsed = extract_json(llm_response.text)

        return AnalystAgentResult(
            question=question,
            model=model,
            prompt_style=prompt_style,
            rag_enabled=rag_enabled,
            raw_response=llm_response.text,
            parsed_response=parsed,
            retrieved_context=context,
            retrieval_results=retrieval_results,
        )
