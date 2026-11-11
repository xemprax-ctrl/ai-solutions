from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parent
QUESTION_LOG = ROOT / "question_log.jsonl"

CATEGORY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "AI and ML": ("ai", "machine learning", "model", "embedding", "vector", "neural"),
    "RAG and Search": ("rag", "retrieval", "chromadb", "search", "semantic", "chunk"),
    "Security and Privacy": ("security", "secure", "privacy", "encrypt", "password", "protect"),
    "APIs and Integration": ("api", "openrouter", "endpoint", "http", "integration", "token"),
    "Python and Development": ("python", "code", "bug", "error", "package", "git", "debug"),
}


def classify_question(question: str) -> str:
    normalized = question.casefold()
    scores = {
        category: sum(keyword in normalized for keyword in keywords)
        for category, keywords in CATEGORY_KEYWORDS.items()
    }
    best_category, best_score = max(scores.items(), key=lambda item: item[1])
    return best_category if best_score else "Other technical"


def record_question(question: str, log_path: Path = QUESTION_LOG) -> None:
    if not question.strip():
        return
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question": question.strip(),
        "category": classify_question(question),
    }
    with log_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(entry, ensure_ascii=False) + "\n")


def load_questions(log_path: Path = QUESTION_LOG) -> list[dict[str, str]]:
    if not log_path.exists():
        return []
    questions: list[dict[str, str]] = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        question = entry.get("question")
        if isinstance(question, str) and question.strip():
            questions.append(
                {
                    "question": question.strip(),
                    "category": str(entry.get("category") or classify_question(question)),
                }
            )
    return questions


def build_report(
    questions: list[str] | None = None,
    output_path: Path = ROOT / "question_analytics.png",
) -> dict[str, Any]:
    entries = (
        [{"question": question, "category": classify_question(question)} for question in questions]
        if questions is not None
        else load_questions()
    )
    if not entries:
        return {
            "question_count": 0,
            "message": "No questions recorded yet. Run the assistant first.",
        }

    category_counts = Counter(entry["category"] for entry in entries)
    question_counts = Counter(entry["question"] for entry in entries)
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "Install chart support first: python -m pip install matplotlib"
        ) from exc

    ordered_categories = category_counts.most_common()
    labels = [item[0] for item in ordered_categories]
    values = [item[1] for item in ordered_categories]
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.bar(labels, values, color="#176b87")
    axis.set_title("Most Asked Technical Question Categories")
    axis.set_xlabel("Category")
    axis.set_ylabel("Number of questions")
    axis.tick_params(axis="x", rotation=25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)

    return {
        "question_count": len(entries),
        "chart": str(output_path),
        "top_categories": [
            {"category": category, "count": count}
            for category, count in ordered_categories
        ],
        "top_questions": [
            {"question": question, "count": count}
            for question, count in question_counts.most_common(10)
        ],
    }


def demo_questions() -> list[str]:
    return [
        "How do I protect my vector embeddings?",
        "How do I protect my vector embeddings?",
        "How do I secure an API key?",
        "How do I secure an API key?",
        "How does RAG retrieval work?",
        "How does RAG retrieval work?",
        "How do I improve semantic search?",
        "How do I write a Python endpoint?",
        "How do I debug a Python package error?",
        "What is machine learning?",
    ]
