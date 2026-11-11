from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from assistant import KnowledgeItem, load_knowledge
from ingestion import recursive_chunks


MODEL_NAME = "all-MiniLM-L6-v2"
COLLECTION_NAME = "project_knowledge"


@dataclass(frozen=True)
class RetrievedDocument:
    id: str
    content: str
    distance: float | None
    metadata: dict[str, str]


@dataclass(frozen=True)
class RetrievalEvaluation:
    question: str
    expected: str
    rank: int | None
    reciprocal_rank: float


def _dependencies() -> tuple[Any, Any]:
    try:
        import chromadb
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "Install the Lab 5 dependencies first: pip install -r requirements.txt"
        ) from exc
    return chromadb, SentenceTransformer


def _document_text(item: KnowledgeItem) -> str:
    return (
        f"{item.title}. {item.answer_en} "
        f"Source: {item.source}."
    )


class RAGService:
    def __init__(
        self,
        knowledge: list[KnowledgeItem] | None = None,
        chunk_size: int = 300,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero.")
        chromadb, sentence_transformer = _dependencies()
        self._model = sentence_transformer(MODEL_NAME)
        self._client = chromadb.Client()
        self._collection = self._client.get_or_create_collection(COLLECTION_NAME)
        self._knowledge = knowledge or load_knowledge()
        self._index(self._knowledge, chunk_size)

    def _index(self, knowledge: list[KnowledgeItem], chunk_size: int) -> None:
        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict[str, str]] = []
        for item in knowledge:
            chunks = recursive_chunks(_document_text(item), chunk_size=chunk_size)
            for index, chunk in enumerate(chunks):
                ids.append(f"{item.id}_{index}")
                documents.append(chunk)
                metadatas.append(
                    {
                        "knowledge_id": item.id,
                        "title": item.title,
                        "source": item.source,
                    }
                )
        if not documents:
            raise ValueError("Cannot build an index from an empty knowledge base.")
        embeddings = self._model.encode(documents).tolist()
        self._collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    @property
    def count(self) -> int:
        return int(self._collection.count())

    def search(
        self,
        question: str,
        n_results: int = 3,
        where: dict[str, str] | None = None,
    ) -> list[RetrievedDocument]:
        if not question.strip():
            raise ValueError("The question cannot be empty.")
        if n_results <= 0:
            raise ValueError("n_results must be greater than zero.")
        query_vector = self._model.encode([question]).tolist()
        results = self._collection.query(
            query_embeddings=query_vector,
            n_results=n_results,
            where=where,
        )
        ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        distances = results.get("distances", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        return [
            RetrievedDocument(
                id=ids[index],
                content=documents[index],
                distance=distances[index] if index < len(distances) else None,
                metadata=metadatas[index] or {},
            )
            for index in range(len(documents))
        ]

    def rag_service(
        self,
        question: str,
        n_results: int = 1,
        where: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        retrieved = self.search(question, n_results=n_results, where=where)
        context = "\n".join(item.content for item in retrieved)
        answer = (
            f"Based on project knowledge: {context}"
            if context
            else "I could not find a trusted answer in the current knowledge base."
        )
        return {
            "question": question,
            "context": context,
            "answer": answer,
            "documents": [asdict(item) for item in retrieved],
            "validation": self.validate_output(context, retrieved),
        }

    @staticmethod
    def validate_output(
        context: str, retrieved: list[RetrievedDocument]
    ) -> dict[str, bool]:
        return {
            "has_context": bool(context.strip()),
            "has_documents": bool(retrieved),
            "grounded": bool(context.strip()) == bool(retrieved),
        }

    def evaluate(
        self, tests: list[tuple[str, str]], k: int = 3
    ) -> dict[str, Any]:
        if not tests:
            raise ValueError("The evaluation set cannot be empty.")
        evaluations: list[RetrievalEvaluation] = []
        for question, expected in tests:
            documents = self.search(question, n_results=k)
            rank = next(
                (
                    index
                    for index, document in enumerate(documents, start=1)
                    if expected.lower() in document.content.lower()
                ),
                None,
            )
            evaluations.append(
                RetrievalEvaluation(
                    question=question,
                    expected=expected,
                    rank=rank,
                    reciprocal_rank=1 / rank if rank else 0,
                )
            )
        hits = sum(item.rank is not None for item in evaluations)
        return {
            "k": k,
            "tests": [asdict(item) for item in evaluations],
            "hit_rate": hits / len(evaluations),
            "mrr": sum(item.reciprocal_rank for item in evaluations)
            / len(evaluations),
        }


def default_evaluation_tests() -> list[tuple[str, str]]:
    return [
        ("How should I protect user data?", "Data privacy"),
        ("What should I check before joining a training program?", "Training"),
        ("How do I start an artificial intelligence project?", "Starting an AI project"),
    ]
