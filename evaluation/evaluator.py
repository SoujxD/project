"""Experiment runner for multi-model and multi-prompt evaluation."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from agents.analyst_agent import AnalystRAGAgent
from evaluation.metrics import (
    completeness_score,
    groundedness_score,
    keyword_score,
    recommendation_score,
)


class EvaluationPipeline:
    """Runs full experiments and aggregates results."""

    def __init__(
        self,
        agent: AnalystRAGAgent,
        questions_path: str | Path | None,
        output_dir: str | Path,
    ) -> None:
        self.agent = agent
        self.questions_path = Path(questions_path) if questions_path else None
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.questions = self._load_questions()

    def _default_dataset_path(self) -> Path:
        return Path(__file__).resolve().parents[1] / "data" / "dataset.csv"

    def _load_questions(self) -> list[dict[str, Any]]:
        is_default_dataset = self.agent.dataset_path.resolve() == self._default_dataset_path().resolve()
        if is_default_dataset and self.questions_path and self.questions_path.exists():
            return json.loads(self.questions_path.read_text())
        return self._build_adaptive_questions()

    def _target_metric(self) -> pd.Series | None:
        schema = self.agent.eda_report.get("profile", {}).get("schema", {})
        return self.agent.eda_agent._target_metric(self.agent.dataframe, schema)

    def _group_top(self, column: str, metric: pd.Series | None) -> tuple[str, float] | None:
        if column not in self.agent.dataframe.columns:
            return None
        working = self.agent.dataframe.copy()
        if metric is not None:
            working["_metric"] = metric
            grouped = working.groupby(column, dropna=False)["_metric"].mean().sort_values(ascending=False)
        else:
            grouped = working[column].astype(str).value_counts(dropna=False).sort_values(ascending=False)
        if grouped.empty:
            return None
        return str(grouped.index[0]), float(grouped.iloc[0])

    def _metric_direction(self, column: str, metric: pd.Series | None) -> tuple[str, float | None, float | None] | None:
        if metric is None or column not in self.agent.dataframe.columns:
            return None
        series = pd.to_numeric(self.agent.dataframe[column], errors="coerce")
        positive = series[metric == 1].dropna()
        negative = series[metric == 0].dropna()
        if positive.empty or negative.empty:
            return None
        pos_mean = float(positive.mean())
        neg_mean = float(negative.mean())
        direction = "higher" if pos_mean >= neg_mean else "lower"
        return direction, pos_mean, neg_mean

    def _build_adaptive_questions(self) -> list[dict[str, Any]]:
        dataframe = self.agent.dataframe
        profile = self.agent.eda_report.get("profile", {})
        schema = profile.get("schema", {})
        metric = self._target_metric()
        questions: list[dict[str, Any]] = []

        def add_question(
            question: str,
            category: str,
            expected_variables: list[str],
            expected_type: str,
            ground_truth: str,
            numeric_answer: float | None,
            expected_keywords: list[str],
        ) -> None:
            questions.append(
                {
                    "id": len(questions) + 1,
                    "category": category,
                    "question": question,
                    "expected_variables": expected_variables,
                    "expected_type": expected_type,
                    "ground_truth": ground_truth,
                    "numeric_answer": numeric_answer,
                    "expected_keywords": expected_keywords,
                }
            )

        target_column = schema.get("target")
        time_column = schema.get("time")
        customer_column = schema.get("customer")
        channel_column = schema.get("channel")
        engagement_column = schema.get("engagement")
        friction_column = schema.get("friction")

        if metric is not None and time_column:
            top = self._group_top(time_column, metric)
            if top:
                add_question(
                    f"Which {time_column} value has the highest outcome rate?",
                    "trend",
                    [time_column, target_column] if target_column else [time_column],
                    "categorical",
                    f"{top[0]} has the highest detected outcome rate.",
                    top[1],
                    [time_column.lower(), "outcome", top[0].lower()],
                )

        if customer_column:
            top = self._group_top(customer_column, metric)
            if top:
                add_question(
                    f"Which {customer_column} segment performs best?",
                    "segmentation",
                    [customer_column] + ([target_column] if target_column else []),
                    "categorical",
                    f"{top[0]} is the strongest {customer_column} segment in this dataset.",
                    top[1],
                    [customer_column.lower(), "segment", str(top[0]).lower()],
                )

        if channel_column:
            top = self._group_top(channel_column, metric)
            if top:
                add_question(
                    f"Which {channel_column} group should we prioritize?",
                    "segmentation",
                    [channel_column] + ([target_column] if target_column else []),
                    "categorical",
                    f"{top[0]} is the strongest {channel_column} grouping in this dataset.",
                    top[1],
                    [channel_column.lower(), "prioritize", str(top[0]).lower()],
                )

        if engagement_column:
            direction = self._metric_direction(engagement_column, metric)
            if direction:
                add_question(
                    f"How does {engagement_column} relate to the detected outcome?",
                    "pattern",
                    [engagement_column] + ([target_column] if target_column else []),
                    "directional",
                    f"{engagement_column} is {direction[0]} for positive outcomes ({direction[1]:.2f} vs {direction[2]:.2f}).",
                    None,
                    [engagement_column.lower(), "outcome", direction[0]],
                )

        if friction_column:
            direction = self._metric_direction(friction_column, metric)
            if direction:
                add_question(
                    f"What pattern does {friction_column} show across outcomes?",
                    "pattern",
                    [friction_column] + ([target_column] if target_column else []),
                    "directional",
                    f"{friction_column} is {direction[0]} for positive outcomes ({direction[1]:.2f} vs {direction[2]:.2f}).",
                    None,
                    [friction_column.lower(), "pattern", direction[0]],
                )

        numeric_columns = [column for column in profile.get("numeric_columns", []) if column in dataframe.columns]
        for column in numeric_columns[:2]:
            direction = self._metric_direction(column, metric)
            if direction:
                add_question(
                    f"How does {column} differ between positive and negative outcomes?",
                    "comparison",
                    [column] + ([target_column] if target_column else []),
                    "directional",
                    f"{column} is {direction[0]} for positive outcomes ({direction[1]:.2f} vs {direction[2]:.2f}).",
                    None,
                    [column.lower(), "positive", direction[0]],
                )

        warnings = self.agent.eda_report.get("quality_checks", {}).get("warnings", [])
        if warnings:
            add_question(
                "What data quality issue matters most before deeper analysis?",
                "quality",
                [],
                "descriptive",
                warnings[0],
                None,
                ["quality", "issue", warnings[0].split()[0].lower()],
            )

        recommended = self.agent.eda_report.get("suggested_questions", [])
        if customer_column or channel_column:
            summary_bits = [bit for bit in [customer_column, channel_column, time_column] if bit]
            add_question(
                "What actions would you recommend based on the strongest detected patterns?",
                "recommendation",
                summary_bits,
                "actionable",
                "Prioritize the strongest groups, monitor the main metric over time, and validate weak areas before acting broadly.",
                None,
                ["prioritize", "monitor", "validate"],
            )

        if target_column:
            add_question(
                f"What patterns stand out among rows with positive {target_column} outcomes?",
                "pattern",
                [target_column] + [column for column in [engagement_column, customer_column, channel_column] if column],
                "descriptive",
                f"Positive {target_column} outcomes concentrate around the strongest detected groups and metrics.",
                None,
                [str(target_column).lower(), "positive", "patterns"],
            )

        while len(questions) < 10:
            fallback_column = customer_column or channel_column or time_column or target_column or dataframe.columns[0]
            add_question(
                f"Which signal in {fallback_column} deserves the next round of analysis?",
                "follow_up",
                [fallback_column] if fallback_column else [],
                "descriptive",
                f"{fallback_column} deserves a deeper follow-up because it is one of the strongest detected signals.",
                None,
                [str(fallback_column).lower(), "signal", "analysis"],
            )

        return questions[:10]

    def run(
        self,
        models: list[str] | None = None,
        prompt_styles: list[str] | None = None,
        rag_options: list[bool] | None = None,
        limit: int | None = None,
    ) -> pd.DataFrame:
        """Run a complete factorial evaluation."""
        models = models or self.agent.available_models()
        prompt_styles = prompt_styles or self.agent.available_prompt_styles()
        rag_options = rag_options or [True, False]
        selected_questions = self.questions[:limit] if limit else self.questions

        records: list[dict[str, Any]] = []
        for model in models:
            for prompt_style in prompt_styles:
                for rag_enabled in rag_options:
                    for item in selected_questions:
                        result = self.agent.answer_question(
                            question=item["question"],
                            model=model,
                            prompt_style=prompt_style,
                            rag_enabled=rag_enabled,
                        )
                        parsed = result.parsed_response
                        record = {
                            "question_id": item["id"],
                            "question": item["question"],
                            "category": item["category"],
                            "model": model,
                            "prompt_style": prompt_style,
                            "rag_enabled": rag_enabled,
                            "summary": parsed.get("summary", ""),
                            "raw_response": result.raw_response,
                            "parsed_response": json.dumps(parsed),
                            "retrieved_context": result.retrieved_context,
                            "keyword_score": keyword_score(parsed, item.get("expected_keywords", [])),
                            "recommendation_score": recommendation_score(parsed),
                            "completeness_score": completeness_score(parsed),
                            "groundedness_score": groundedness_score(parsed, result.retrieved_context),
                        }
                        record["overall_score"] = round(
                            (
                                record["keyword_score"]
                                + record["recommendation_score"]
                                + record["completeness_score"]
                                + record["groundedness_score"]
                            )
                            / 4,
                            4,
                        )
                        records.append(record)

        df = pd.DataFrame(records)
        df.to_csv(self.output_dir / "results.csv", index=False)
        self.aggregate_results(df)
        return df

    def aggregate_results(self, df: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """Create grouped summary tables and export them."""
        metric_columns = [
            "keyword_score",
            "recommendation_score",
            "completeness_score",
            "groundedness_score",
            "overall_score",
        ]
        model_table = df.groupby("model")[metric_columns].mean().sort_values("overall_score", ascending=False)
        prompt_table = df.groupby("prompt_style")[metric_columns].mean().sort_values("overall_score", ascending=False)
        rag_table = df.groupby("rag_enabled")[metric_columns].mean().sort_values("overall_score", ascending=False)
        category_table = df.groupby("category")[metric_columns].mean().sort_values("overall_score", ascending=False)

        tables = {
            "model_comparison": model_table,
            "prompt_comparison": prompt_table,
            "rag_comparison": rag_table,
            "category_comparison": category_table,
        }
        for name, table in tables.items():
            table.to_csv(self.output_dir / f"{name}.csv")
        return tables

    def top_examples(self, df: pd.DataFrame, top_n: int = 5) -> list[dict[str, Any]]:
        """Return the best-scoring rows for presentation generation."""
        top_df = df.sort_values("overall_score", ascending=False).head(top_n)
        return top_df.to_dict(orient="records")
