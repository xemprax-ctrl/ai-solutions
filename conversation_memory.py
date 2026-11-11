from __future__ import annotations

import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Message:
    role: str
    content: str


class ConversationMemory:
    def __init__(self, database_path: Path | str = "conversation.db") -> None:
        self.database_path = Path(database_path)
        self._connection = sqlite3.connect(self.database_path)
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self._connection.commit()

    def save(self, role: str, content: str) -> None:
        if not role.strip() or not content.strip():
            raise ValueError("Role and content cannot be empty.")
        self._connection.execute(
            "INSERT INTO messages (role, content) VALUES (?, ?)",
            (role.strip(), content.strip()),
        )
        self._connection.commit()

    def all_messages(self) -> list[Message]:
        rows = self._connection.execute(
            "SELECT role, content FROM messages ORDER BY id"
        ).fetchall()
        return [Message(role=row[0], content=row[1]) for row in rows]

    def clear(self) -> None:
        self._connection.execute("DELETE FROM messages")
        self._connection.commit()

    def recent_messages(self, window_size: int = 3) -> list[Message]:
        if window_size <= 0:
            raise ValueError("window_size must be greater than zero.")
        return self.all_messages()[-window_size:]

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> ConversationMemory:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def recursive_summary(messages: list[Message], previous_summary: str = "") -> str:
    if not messages and not previous_summary:
        return "No conversation to summarize."
    facts = [message.content.rstrip(".") for message in messages]
    current = "; ".join(facts)
    if previous_summary and current:
        return f"{previous_summary.rstrip('.')}; {current}."
    return f"{previous_summary.rstrip('.')}." if previous_summary else f"{current}."


def fidelity_gate(summary: str, important_information: list[str]) -> bool:
    normalized_summary = summary.casefold()
    return all(item.casefold() in normalized_summary for item in important_information)


def hybrid_compact(
    summary: str, messages: list[Message], recent_count: int = 2
) -> list[dict[str, str]]:
    if recent_count <= 0:
        raise ValueError("recent_count must be greater than zero.")
    return [{"role": "summary", "content": summary}] + [
        asdict(message) for message in messages[-recent_count:]
    ]


def run_memory_demo(database_path: Path | str = "conversation.db") -> dict[str, object]:
    messages = [
        Message("user", "I want to start an AI project."),
        Message("user", "How should I protect the project data?"),
        Message("user", "Which training program should I review first?"),
    ]
    with ConversationMemory(database_path) as memory:
        memory.clear()
        for message in messages:
            memory.save(message.role, message.content)
        stored = memory.all_messages()
        recent = memory.recent_messages(3)
    summary = "The user wants to start an AI project, protect project data, and review a training program."
    important_information = ["AI project", "protect project data", "training program"]
    return {
        "total_messages": len(stored),
        "recent_messages": [asdict(message) for message in recent],
        "summary": summary,
        "fidelity_gate": "Passed"
        if fidelity_gate(summary, important_information)
        else "Failed",
        "compact_context": hybrid_compact(summary, stored),
    }
