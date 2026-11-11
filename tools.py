from __future__ import annotations

import asyncio
import os
import time
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field

from assistant import AssistantResponse, answer_question, load_knowledge


class KnowledgeSearchInput(BaseModel):
    question: str = Field(min_length=1)
    answer_language: str = Field(pattern="^en$")


class TopicListInput(BaseModel):
    language: str = Field(pattern="^en$")


class EmptyInput(BaseModel):
    pass


class ToolDefinition(BaseModel):
    name: str
    description: str
    input_model: type[BaseModel]
    handler: Callable[..., Any]

    def schema_for_model(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_model.model_json_schema(),
            },
        }


def search_knowledge(question: str, answer_language: str) -> dict[str, Any]:
    data: AssistantResponse = answer_question(
        question, load_knowledge(), answer_language
    )
    return data.__dict__


def list_topics(language: str) -> list[str]:
    items = load_knowledge()
    return [item.title.split(" / ")[-1] for item in items]


def api_status() -> dict[str, str | bool]:
    configured = bool(os.getenv("OPENROUTER_API_KEY", "").strip())
    return {
        "configured": configured,
        "message": "API key is configured" if configured else "API key is not configured",
    }


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    def register(self, definition: ToolDefinition) -> None:
        if definition.name in self._tools:
            raise ValueError(f"Tool is already registered: {definition.name}")
        self._tools[definition.name] = definition

    def names(self) -> list[str]:
        return list(self._tools)

    def schemas(self) -> list[dict[str, Any]]:
        return [tool.schema_for_model() for tool in self._tools.values()]

    def execute(self, name: str, arguments: dict[str, Any]) -> Any:
        if name not in self._tools:
            raise KeyError(f"Tool does not exist: {name}")
        tool = self._tools[name]
        validated = tool.input_model.model_validate(arguments)
        return tool.handler(**validated.model_dump())

    async def execute_parallel(
        self, calls: list[tuple[str, dict[str, Any]]]
    ) -> list[Any]:
        return await asyncio.gather(
            *(asyncio.to_thread(self.execute, name, arguments) for name, arguments in calls)
        )

def build_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(ToolDefinition(
        name="search_knowledge",
        description="Search the grounded project knowledge base and return an answer.",
        input_model=KnowledgeSearchInput,
        handler=search_knowledge,
    ))
    registry.register(ToolDefinition(
        name="list_topics",
        description="List topics available in the project knowledge base.",
        input_model=TopicListInput,
        handler=list_topics,
    ))
    registry.register(ToolDefinition(
        name="api_status",
        description="Check whether the optional API key is configured.",
        input_model=EmptyInput,
        handler=api_status,
    ))
    return registry


async def run_parallel_demo() -> dict[str, Any]:
    registry = build_registry()
    calls = [
        ("list_topics", {"language": "en"}),
        ("api_status", {}),
    ]
    started = time.perf_counter()
    results = await registry.execute_parallel(calls)
    elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
    return {"results": results, "elapsed_ms": elapsed_ms, "call_count": len(calls)}


def run_integration_test() -> list[tuple[str, str]]:
    registry = build_registry()
    cases = [
        ("search_knowledge", {"question": "How should I protect user data?", "answer_language": "en"}),
        ("list_topics", {"language": "en"}),
        ("api_status", {}),
    ]
    results: list[tuple[str, str]] = []
    for name, arguments in cases:
        registry.execute(name, arguments)
        results.append((name, "success"))
    try:
        registry.execute("unknown_tool", {})
    except KeyError:
        results.append(("unknown_tool", "not_found"))
    return results
