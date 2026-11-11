from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path
from urllib import error, request

from assistant import answer_question, load_knowledge
from agent_evaluation import run_agent_evaluation, run_deepeval_evaluation
from conversation_memory import run_memory_demo
from ingestion import run_pipeline
from rag_service import RAGService, default_evaluation_tests
from tools import build_registry, run_integration_test, run_parallel_demo


ROOT = Path(__file__).parent
API_URL = "https://openrouter.ai/api/v1/chat/completions"


def load_dotenv(path: Path = ROOT / ".env") -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        os.environ.setdefault(name.strip(), value.strip().strip("\"'"))


def ask_api(question: str, answer_language: str, api_key: str) -> str:
    language = "English"
    retrieval = RAGService()
    documents = retrieval.search(question, n_results=3)
    if not documents:
        raise RuntimeError("No trusted context was found for this question.")
    context = "\n".join(
        f"- {document.content} "
        f"(source: {document.metadata.get('source', 'unknown')})"
        for document in documents
    )
    payload = {
        "model": os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a grounded AI solutions advisor. Generate a concise answer "
                    "only from the provided retrieved context. If it is insufficient, "
                    "say so clearly and do not invent facts. "
                    f"Answer in {language}."
                ),
            },
            {"role": "system", "content": f"Knowledge context:\n{context}"},
            {"role": "user", "content": question},
        ],
        "temperature": 0.2,
        "max_tokens": 500,
    }
    req = request.Request(
        API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    req.add_header("Authorization", "Bearer " + api_key)
    try:
        with request.urlopen(req, timeout=60) as response:
            body = json.loads(response.read().decode("utf-8"))
    except (error.HTTPError, error.URLError, TimeoutError) as exc:
        raise RuntimeError(f"Could not connect to the AI service: {exc}") from exc
    try:
        return body["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("Unexpected API response.") from exc


def choose_language(prompt: str) -> str:
    while True:
        value = input(prompt).strip().lower()
        if value == "en":
            return value
        print("Please enter en for English.")


def print_response(response: dict[str, object]) -> None:
    print(f"\nAnswer: {response['answer']}")
    print(f"Source: {response.get('source') or 'None'}")
    print(f"Confidence: {float(response['confidence']):.0%}")


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="English AI solutions assistant")
    parser.add_argument("--question", help="Question to answer")
    parser.add_argument("--question-language", choices=("en",), default="en", help="Question language")
    parser.add_argument("--answer-language", choices=("en",), default="en", help="Answer language")
    parser.add_argument("--use-api", action="store_true", help="Use the API instead of local answers")
    parser.add_argument("--show-tools", action="store_true", help="Show Lab 3 tools and schemas")
    parser.add_argument("--test-tools", action="store_true", help="Run the Lab 3 integration test")
    parser.add_argument("--parallel-tools", action="store_true", help="Run Lab 3 tools in parallel")
    parser.add_argument("--ingest", action="store_true", help="Run the Lab 4 ingestion pipeline")
    parser.add_argument("--embeddings", action="store_true", help="Include embeddings in Lab 4 output")
    parser.add_argument("--rag", metavar="QUESTION", help="Run semantic search and RAG (Lab 5)")
    parser.add_argument("--filter-id", help="Filter RAG results by knowledge ID (Lab 5)")
    parser.add_argument("--evaluate-retrieval", action="store_true", help="Evaluate retrieval quality (Lab 5)")
    parser.add_argument("--memory-demo", action="store_true", help="Run the conversation memory demo (Lab 6)")
    parser.add_argument("--agent-evaluation", action="store_true", help="Run loop detection, tracing, and evaluation (Lab 7)")
    parser.add_argument("--deepeval-evaluation", action="store_true", help="Run DeepEval RAG metrics (Lab 7)")
    args = parser.parse_args()

    if args.show_tools:
        registry = build_registry()
        print("Available tools:", ", ".join(registry.names()))
        print(json.dumps(registry.schemas(), ensure_ascii=False, indent=2))
        return
    if args.test_tools:
        for name, status in run_integration_test():
            print(f"{name}: {status}")
        return
    if args.parallel_tools:
        result = asyncio.run(run_parallel_demo())
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    if args.ingest or args.embeddings:
        result = run_pipeline(include_embeddings=args.embeddings)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    if args.rag:
        service = RAGService()
        where = {"knowledge_id": args.filter_id} if args.filter_id else None
        print(json.dumps(
            service.rag_service(args.rag, where=where),
            ensure_ascii=False,
            indent=2,
        ))
        return
    if args.evaluate_retrieval:
        service = RAGService()
        result = service.evaluate(default_evaluation_tests())
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    if args.memory_demo:
        print(json.dumps(run_memory_demo(), ensure_ascii=False, indent=2))
        return
    if args.agent_evaluation:
        print(json.dumps(run_agent_evaluation(), ensure_ascii=False, indent=2))
        return
    if args.deepeval_evaluation:
        print(json.dumps(run_deepeval_evaluation(), ensure_ascii=False, indent=2))
        return

    question = args.question or input("Enter your question: ").strip()
    question_language = args.question_language or choose_language("Question language (en): ")
    answer_language = args.answer_language or choose_language("Answer language (en): ")
    if not question:
        raise SystemExit("The question cannot be empty.")

    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if args.use_api:
        if not api_key:
            raise SystemExit("Set OPENROUTER_API_KEY in .env or the environment first.")
        started = time.perf_counter()
        answer = ask_api(question, answer_language, api_key)
        print(f"\nAnswer: {answer}")
        print(f"Latency: {(time.perf_counter() - started) * 1000:.0f} ms")
        return

    response = answer_question(question, load_knowledge(), answer_language)
    print_response(response.__dict__)


if __name__ == "__main__":
    main()
