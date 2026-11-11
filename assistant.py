from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path


ROOT = Path(__file__).parent
KNOWLEDGE_FILE = ROOT / "knowledge_base.json"


@dataclass(frozen=True)
class KnowledgeItem:
    id: str
    title: str
    keywords: list[str]
    answer_en: str
    source: str


@dataclass(frozen=True)
class AssistantResponse:
    question: str
    answer: str
    source: str | None
    confidence: float
    topic: str | None
    question_language: str
    answer_language: str
    status: str
    latency_ms: int


def load_knowledge(path: Path = KNOWLEDGE_FILE) -> list[KnowledgeItem]:
    records = json.loads(path.read_text(encoding="utf-8"))
    return [KnowledgeItem(**record) for record in records]


def normalize(text: str) -> str:
    text = text.lower().strip()
    return re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)


def _score(question: str, item: KnowledgeItem) -> float:
    normalized = normalize(question)
    words = set(normalized.split())
    matches = sum(
        1 for keyword in item.keywords
        if normalize(keyword) in normalized or normalize(keyword) in words
    )
    keyword_score = min(matches / 3, 1.0)
    title_score = SequenceMatcher(None, normalized, normalize(item.title)).ratio()
    return min(0.75 * keyword_score + 0.25 * title_score, 1.0)


def answer_question(
    question: str,
    knowledge: list[KnowledgeItem],
    answer_language: str = "auto",
) -> AssistantResponse:
    started = time.perf_counter()
    if not question.strip():
        raise ValueError("The question cannot be empty.")
    if answer_language not in {"auto", "en"}:
        raise ValueError("Answer language must be auto or en.")

    question_language = "en"
    selected_language = "en"
    item, confidence = max(
        ((item, _score(question, item)) for item in knowledge),
        key=lambda result: result[1],
    )
    if confidence < 0.30:
        answer = (
            "I could not find a trusted answer in the current knowledge base."
        )
        response = AssistantResponse(
            question, answer, None, round(confidence, 2), None,
            question_language, selected_language, "not_found",
            round((time.perf_counter() - started) * 1000),
        )
    else:
        response = AssistantResponse(
            question,
            item.answer_en,
            item.source,
            round(confidence, 2),
            item.title,
            question_language,
            selected_language,
            "grounded",
            round((time.perf_counter() - started) * 1000),
        )
    return response
