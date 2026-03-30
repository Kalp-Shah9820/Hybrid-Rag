"""
RAGAS-based Evaluation Module
Evaluates the RAG system on faithfulness, answer relevancy,
context precision, and context recall.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import pandas as pd
from loguru import logger

from src.config import settings
from src.graph import run_agent


def build_eval_dataset(
    questions: List[str],
    ground_truths: List[str] | None = None,
) -> List[dict]:
    """
    Run each question through the RAG agent and collect results
    in the format expected by RAGAS.
    """
    results = []
    for i, q in enumerate(questions):
        logger.info(f"Evaluating question {i + 1}/{len(questions)}: {q[:60]}...")
        answer = run_agent(q)

        entry = {
            "question": q,
            "answer": answer.answer,
            "contexts": [c.chunk.content for c in answer.chunks_used],
        }
        if ground_truths and i < len(ground_truths):
            entry["ground_truth"] = ground_truths[i]
        results.append(entry)

    return results


def evaluate(
    questions: List[str],
    ground_truths: List[str] | None = None,
    output_path: str | None = None,
) -> dict:
    """
    Run RAGAS evaluation on the RAG system.
    Returns a dict of metric_name -> score.
    """
    try:
        from ragas import evaluate as ragas_evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        )
        from datasets import Dataset
    except ImportError:
        logger.error(
            "❌ RAGAS not installed. Run: pip install ragas"
        )
        return {}

    # Build dataset
    eval_data = build_eval_dataset(questions, ground_truths)

    # Convert to HuggingFace Dataset
    ds = Dataset.from_list(eval_data)

    # Select metrics
    metric_map = {
        "faithfulness": faithfulness,
        "answer_relevancy": answer_relevancy,
        "context_precision": context_precision,
        "context_recall": context_recall,
    }
    active_metrics = [
        metric_map[m]
        for m in settings.evaluation.metrics
        if m in metric_map
    ]

    logger.info(f"📊 Running RAGAS evaluation with {len(active_metrics)} metrics...")
    result = ragas_evaluate(ds, metrics=active_metrics)

    scores = {k: round(v, 4) for k, v in result.items() if isinstance(v, float)}
    logger.info(f"📊 Evaluation results: {scores}")

    # Save results
    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump({"scores": scores, "details": eval_data}, f, indent=2)
        logger.info(f"💾 Results saved to {output_path}")

    return scores


# ── Sample evaluation questions ──────────────────────────────
SAMPLE_QUESTIONS = [
    "What were the major stock market trends this week?",
    "How did Tesla stock perform recently?",
    "What impact did Federal Reserve decisions have on the market?",
    "Which sectors showed the strongest growth?",
    "What are analysts predicting for the tech sector?",
]

if __name__ == "__main__":
    scores = evaluate(
        SAMPLE_QUESTIONS,
        output_path="data/eval_results.json",
    )
    print("\n📊 Final Scores:")
    for metric, score in scores.items():
        print(f"  {metric}: {score:.4f}")
