from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from functools import wraps
from typing import Any, Callable, TypeVar

from assistant import answer_question, load_knowledge, normalize


F = TypeVar("F", bound=Callable[..., Any])


class LoopDetector:
    def __init__(self, max_steps: int = 3) -> None:
        if max_steps <= 0:
            raise ValueError("max_steps must be greater than zero.")
        self.max_steps = max_steps
        self._seen: set[str] = set()

    def check(self, value: str) -> bool:
        normalized = normalize(value)
        if normalized in self._seen:
            return False
        if len(self._seen) >= self.max_steps:
            return False
        self._seen.add(normalized)
        return True

    def process(self, values: list[str]) -> dict[str, Any]:
        processed: list[str] = []
        loop_detected: str | None = None
        for value in values:
            if not self.check(value):
                loop_detected = value
                break
            processed.append(value)
        return {
            "processed": processed,
            "loop_detected": loop_detected,
            "stopped": loop_detected is not None,
        }


@dataclass
class TraceNode:
    name: str
    inputs: dict[str, Any]
    started_at: float
    ended_at: float | None = None
    output: Any = None
    children: list[TraceNode] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["duration_ms"] = (
            round((self.ended_at - self.started_at) * 1000, 2)
            if self.ended_at is not None
            else None
        )
        return result


_trace_stack: list[TraceNode] = []
_trace_roots: list[TraceNode] = []


def observe(func: F) -> F:
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        node = TraceNode(func.__name__, {"args": args, "kwargs": kwargs}, time.perf_counter())
        if _trace_stack:
            _trace_stack[-1].children.append(node)
        else:
            _trace_roots.append(node)
        _trace_stack.append(node)
        try:
            node.output = func(*args, **kwargs)
            return node.output
        finally:
            node.ended_at = time.perf_counter()
            _trace_stack.pop()

    return wrapper  # type: ignore[return-value]


def clear_trace() -> None:
    _trace_roots.clear()
    _trace_stack.clear()


def trace_tree() -> list[dict[str, Any]]:
    return [node.to_dict() for node in _trace_roots]


@observe
def project_assistant(question: str) -> str:
    response = answer_question(question, load_knowledge(), "en")
    return response.answer


def evaluate_answer(question: str, expected: str) -> dict[str, Any]:
    answer = project_assistant(question)
    normalized_answer = normalize(answer)
    normalized_expected = normalize(expected)
    answer_tokens = set(normalized_answer.split())
    expected_tokens = set(normalized_expected.split())
    overlap = (
        len(answer_tokens & expected_tokens) / len(expected_tokens)
        if expected_tokens
        else 0.0
    )
    exact_match = normalized_expected in normalized_answer
    return {
        "question": question,
        "answer": answer,
        "expected": expected,
        "metrics": {
            "answer_relevancy": round(overlap, 2),
            "faithfulness": 1.0 if exact_match else 0.0,
            "task_completion": 1.0 if answer else 0.0,
            "exact_match": exact_match,
        },
        "evaluation": "Passed" if exact_match else "Failed",
    }


def run_agent_evaluation() -> dict[str, Any]:
    cases = [
        (
            "How should I protect user data?",
            "Collect only the data you need, explain why it is used, and store it securely with restricted access.",
        ),
        (
            "How do I start an AI project?",
            "Define the problem and a measurable success metric, prepare suitable data, build a prototype, and test it before scaling.",
        ),
    ]
    clear_trace()
    results = [evaluate_answer(question, expected) for question, expected in cases]
    passed = sum(result["evaluation"] == "Passed" for result in results)
    return {
        "cases": results,
        "passed": passed,
        "total": len(results),
        "pass_rate": passed / len(results),
        "trace": trace_tree(),
        "loop_check": LoopDetector(max_steps=3).process(
            ["How should I protect user data?", "How should I protect user data?"]
        ),
    }


def run_deepeval_evaluation() -> dict[str, Any]:
    try:
        from deepeval import evaluate
        from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
        from deepeval.test_case import LLMTestCase
    except ImportError as exc:
        raise RuntimeError(
            "Install DeepEval first: pip install deepeval"
        ) from exc

    question = "How should I protect user data?"
    response = project_assistant(question)
    test_case = LLMTestCase(
        input=question,
        actual_output=response,
        retrieval_context=[response],
    )
    result = evaluate(
        [test_case],
        [
            AnswerRelevancyMetric(threshold=0.7),
            FaithfulnessMetric(threshold=0.7),
        ],
    )
    return {"result": result}
