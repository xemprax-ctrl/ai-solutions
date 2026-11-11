from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).parent
KNOWLEDGE_FILE = ROOT / "knowledge_base.json"


@dataclass(frozen=True)
class Document:
    source: str
    content: str


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def load_knowledge_document(path: Path = KNOWLEDGE_FILE) -> Document:
    records = json.loads(path.read_text(encoding="utf-8"))
    content = "\n".join(
        f"{item['title']}. {item['answer_en']} "
        f"Source: {item['source']}."
        for item in records
    )
    return Document(source=path.name, content=clean_text(content))


def recursive_chunks(
    content: str,
    chunk_size: int = 300,
    chunk_overlap: int = 0,
) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be between zero and chunk_size - 1.")
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError as exc:
        raise RuntimeError(
            "Install the package to split documents: pip install langchain-text-splitters"
        ) from exc

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[". ", "\n", " ", ""],
    )
    return splitter.split_text(content)


def create_embeddings(chunks: list[str]) -> dict[str, Any]:
    if not chunks:
        raise ValueError("Cannot create embeddings for a document without chunks.")
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "Install the package to create embeddings: pip install sentence-transformers"
        ) from exc

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks)
    if len(embeddings) != len(chunks):
        raise RuntimeError("Validation failed: embedding count does not match chunk count.")
    return {
        "embedding_count": len(embeddings),
        "vector_dimensions": int(embeddings.shape[1]),
        "sample_vector": embeddings[0][:5].round(3).tolist(),
        "validation": len(embeddings) == len(chunks),
    }


def run_pipeline(
    path: Path = KNOWLEDGE_FILE,
    chunk_size: int = 300,
    chunk_overlap: int = 0,
    include_embeddings: bool = False,
) -> dict[str, Any]:
    document = load_knowledge_document(path)
    chunks = recursive_chunks(document.content, chunk_size, chunk_overlap)
    result: dict[str, Any] = {
        "document": asdict(document),
        "chunk_count": len(chunks),
        "chunks": chunks,
    }
    if include_embeddings:
        result["embeddings"] = create_embeddings(chunks)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Lab 4 ingestion and chunking pipeline")
    parser.add_argument("--chunk-size", type=int, default=300)
    parser.add_argument("--chunk-overlap", type=int, default=0)
    parser.add_argument("--embeddings", action="store_true")
    args = parser.parse_args()
    result = run_pipeline(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        include_embeddings=args.embeddings,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
