"""CLI entrypoint for running demos, evaluation, and presentation generation."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from agents.analyst_agent import AnalystRAGAgent


BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "data" / "dataset.csv"
QUESTIONS_PATH = BASE_DIR / "data" / "evaluation_questions.json"
OUTPUT_DIR = BASE_DIR / "outputs"


def build_agent() -> AnalystRAGAgent:
    return AnalystRAGAgent(dataset_path=DATASET_PATH)


def run_demo(question: str, model: str, prompt_style: str, rag_enabled: bool) -> None:
    agent = build_agent()
    result = agent.answer_question(question=question, model=model, prompt_style=prompt_style, rag_enabled=rag_enabled)
    print("Question:", question)
    print("Retrieved context:\n", result.retrieved_context or "[RAG disabled]")
    print("Parsed response:\n", result.parsed_response)


def run_evaluation(limit: int | None = None) -> pd.DataFrame:
    from evaluation.evaluator import EvaluationPipeline

    agent = build_agent()
    pipeline = EvaluationPipeline(agent=agent, questions_path=QUESTIONS_PATH, output_dir=OUTPUT_DIR)
    return pipeline.run(limit=limit)


def run_presentation() -> Path:
    from agents.presentation_agent import PresentationGeneratorAgent

    presenter = PresentationGeneratorAgent(output_dir=OUTPUT_DIR)
    output_path, _, _ = presenter.create_presentation(dataset_path=DATASET_PATH, output_path=OUTPUT_DIR / "presentation.pptx")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-agent business analytics system")
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo_parser = subparsers.add_parser("demo", help="Run a single analyst-agent demo")
    demo_parser.add_argument("--question", required=True)
    demo_parser.add_argument("--model", default="meta-llama/llama-3.1-8b-instruct")
    demo_parser.add_argument("--prompt-style", default="structured_json")
    demo_parser.add_argument("--no-rag", action="store_true")

    eval_parser = subparsers.add_parser("evaluate", help="Run the experiment pipeline")
    eval_parser.add_argument("--limit", type=int, default=None)

    subparsers.add_parser("presentation", help="Generate the PowerPoint deck")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "demo":
        run_demo(args.question, args.model, args.prompt_style, not args.no_rag)
    elif args.command == "evaluate":
        df = run_evaluation(limit=args.limit)
        print(f"Saved {len(df)} evaluation rows to {OUTPUT_DIR / 'results.csv'}")
    elif args.command == "presentation":
        path = run_presentation()
        print(f"Saved presentation to {path}")


if __name__ == "__main__":
    main()
