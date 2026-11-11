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
from question_analytics import build_report, demo_questions, record_question
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
        name = name.strip()
        if name.startswith("export "):
            name = name[7:].strip()
        os.environ.setdefault(name, value.strip().strip("\"'"))


def ask_api(
    question: str,
    answer_language: str,
    api_key: str,
    grounded: bool = True,
) -> str:
    if answer_language not in {"en", "ar"}:
        raise ValueError("Answer language must be en or ar.")
    language = "English" if answer_language == "en" else "Arabic"
    context = ""
    if grounded:
        retrieval = RAGService()
        documents = retrieval.search(question, n_results=3)
        if not documents:
            raise RuntimeError("No trusted context was found for this question.")
        context = "\n".join(
            f"- {document.content} "
            f"(source: {document.metadata.get('source', 'unknown')})"
            for document in documents
        )
    system_prompt = (
        "You are a grounded AI solutions advisor. Generate a concise answer "
        "only from the provided retrieved context. If it is insufficient, "
        "say so clearly and do not invent facts. Mention the relevant source "
        "when it helps the user verify the answer. "
        if grounded
        else
        "You are a helpful technical AI assistant. Answer the user's technical "
        "question directly and accurately. Explain assumptions, give practical "
        "steps or code when useful, and clearly say when a detail is uncertain. "
        "When a technical term is misspelled, infer its meaning from the surrounding "
        "context instead of inventing unrelated interpretations."
    )
    response_format = (
        " Keep the answer under 180 words. Return exactly two parts: "
        "Summary: two or three short sentences. Steps: up to 4 numbered steps. "
        "If the question is ambiguous, state one short assumption."
    )
    payload = {
        "model": os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
        "messages": [
            {
                "role": "system",
                "content": f"{system_prompt}{response_format} Answer in {language}.",
            },
            {"role": "user", "content": question},
        ],
        "temperature": 0.2,
        "max_tokens": int(os.getenv("OPENROUTER_MAX_TOKENS", "220")),
    }
    if grounded:
        payload["messages"].insert(
            1, {"role": "system", "content": f"Knowledge context:\n{context}"}
        )
    req = request.Request(
        API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": os.getenv("OPENROUTER_SITE_URL", "http://localhost"),
            "X-Title": os.getenv("OPENROUTER_APP_NAME", "Grounded AI Solutions Assistant"),
        },
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=60) as response:
            body = json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace").strip()
        raise RuntimeError(
            f"OpenRouter returned HTTP {exc.code}: {details or exc.reason}"
        ) from exc
    except (error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not connect to OpenRouter: {exc}") from exc

    if body.get("error"):
        raise RuntimeError(f"OpenRouter error: {body['error']}")
    try:
        content = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("OpenRouter returned an unexpected response.") from exc
    if isinstance(content, str):
        answer = content.strip()
    elif isinstance(content, list):
        answer = " ".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and isinstance(part.get("text"), str)
        ).strip()
    else:
        answer = ""
    if not answer:
        raise RuntimeError("OpenRouter returned an empty answer.")
    return answer


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
    parser = argparse.ArgumentParser(description="Grounded AI solutions assistant")
    parser.add_argument("--question", help="Question to answer")
    parser.add_argument("--question-language", choices=("en", "ar"), default="en", help="Question language")
    parser.add_argument("--answer-language", choices=("en", "ar"), default="en", help="Answer language")
    parser.add_argument("--use-api", action="store_true", help="Use the API instead of local answers")
    parser.add_argument(
        "--general",
        action="store_true",
        help="Answer any technical question directly with OpenRouter instead of RAG",
    )
    parser.add_argument("--analytics", action="store_true", help="Generate a chart from recorded questions")
    parser.add_argument("--analytics-demo", action="store_true", help="Generate a chart from sample technical questions")
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

    if args.analytics or args.analytics_demo:
        questions = demo_questions() if args.analytics_demo else None
        print(json.dumps(build_report(questions), ensure_ascii=False, indent=2))
        return
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
    record_question(question)

    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if args.answer_language == "ar" and not args.use_api:
        raise SystemExit("Arabic answers require OpenRouter. Add --use-api and configure OPENROUTER_API_KEY.")
    if args.use_api:
        if not api_key:
            raise SystemExit("Set OPENROUTER_API_KEY in .env or the environment first.")
        started = time.perf_counter()
        answer = ask_api(question, answer_language, api_key, grounded=not args.general)
        print(f"\nAnswer: {answer}")
        print(f"Latency: {(time.perf_counter() - started) * 1000:.0f} ms")
        return

    response = answer_question(question, load_knowledge(), answer_language)
    print_response(response.__dict__)


if __name__ == "__main__":
    main()
