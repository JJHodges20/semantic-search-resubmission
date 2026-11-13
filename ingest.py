"""Document loading, chunking, and ChromaDB ingestion utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import chromadb


SUPPORTED_EXTENSIONS = {".txt", ".md"}

DEFAULT_DB_PATH = "chroma_data"

DEFAULT_COLLECTION = "course_docs"


def load_documents(docs_dir: str | Path = "docs") -> list[dict]:
    """Load supported text documents from a directory."""

    docs_path = Path(docs_dir)

    if not docs_path.exists():
        raise FileNotFoundError(
            f"Document directory not found: {docs_path}"
        )

    documents = []

    for path in sorted(docs_path.iterdir()):

        if (
            path.is_file()
            and path.suffix.lower() in SUPPORTED_EXTENSIONS
        ):

            text = path.read_text(
                encoding="utf-8"
            ).strip()

            if text:

                documents.append(
                    {
                        "document_id": path.stem,
                        "source": path.name,
                        "text": text,
                    }
                )

    return documents


def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 75,
) -> list[str]:
    """Split text into overlapping character chunks."""

    if chunk_size <= 0:
        raise ValueError(
            "chunk_size must be greater than 0"
        )

    if overlap < 0:
        raise ValueError(
            "overlap cannot be negative"
        )

    if overlap >= chunk_size:
        raise ValueError(
            "overlap must be smaller than chunk_size"
        )

    clean_text = " ".join(
        text.split()
    )

    if not clean_text:
        return []

    chunks = []

    start = 0

    text_length = len(clean_text)

    while start < text_length:

        target_end = min(
            start + chunk_size,
            text_length,
        )

        end = target_end

        # Try to end the chunk at a whitespace boundary
        # instead of cutting a word in half.
        if target_end < text_length:

            boundary = clean_text.rfind(
                " ",
                start + max(
                    1,
                    chunk_size // 2,
                ),
                target_end,
            )

            if boundary > start:
                end = boundary

        chunk = clean_text[
            start:end
        ].strip()

        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        next_start = end - overlap

        if next_start <= start:
            next_start = end

        start = next_start

    return chunks


def build_chunks(
    documents: Iterable[dict],
    chunk_size: int = 500,
    overlap: int = 75,
) -> list[dict]:
    """Convert documents into searchable chunks with metadata."""

    chunks = []

    for document in documents:

        document_chunks = chunk_text(
            document["text"],
            chunk_size=chunk_size,
            overlap=overlap,
        )

        for index, chunk in enumerate(
            document_chunks
        ):

            chunks.append(
                {
                    "id": (
                        f'{document["document_id"]}'
                        f"_chunk_{index}"
                    ),
                    "text": chunk,
                    "metadata": {
                        "source": document["source"],
                        "document_id": document[
                            "document_id"
                        ],
                        "chunk_index": index,
                        "chunk_size": chunk_size,
                        "overlap": overlap,
                    },
                }
            )

    return chunks


def get_client(
    db_path: str = DEFAULT_DB_PATH,
):
    """Return a persistent ChromaDB client."""

    return chromadb.PersistentClient(
        path=db_path
    )


def reset_collection(
    client,
    collection_name: str,
) -> None:
    """Delete a collection if it already exists."""

    try:

        client.delete_collection(
            collection_name
        )

    except Exception as exc:
        # Ignore only a missing collection; do not conceal database errors.
        if "does not exist" not in str(exc).lower() and "not found" not in str(exc).lower():
            raise


def index_documents(
    docs_dir: str | Path = "docs",
    db_path: str = DEFAULT_DB_PATH,
    collection_name: str = DEFAULT_COLLECTION,
    chunk_size: int = 500,
    overlap: int = 75,
    reset: bool = True,
):
    """Load, chunk, and store documents in ChromaDB."""

    documents = load_documents(
        docs_dir
    )

    if not documents:
        raise ValueError(
            f"No .txt or .md documents "
            f"found in {docs_dir}"
        )

    chunks = build_chunks(
        documents,
        chunk_size=chunk_size,
        overlap=overlap,
    )

    client = get_client(
        db_path
    )

    if reset:

        reset_collection(
            client,
            collection_name,
        )

    collection = (
        client.get_or_create_collection(
            name=collection_name
        )
    )

    # Batch writes to avoid exceeding ChromaDB batch limits on larger corpora.
    for offset in range(0, len(chunks), 100):
        batch = chunks[offset:offset + 100]
        collection.upsert(
            ids=[item["id"] for item in batch],
            documents=[item["text"] for item in batch],
            metadatas=[item["metadata"] for item in batch],
        )

    stats = {
        "documents": len(documents),
        "chunks": len(chunks),
        "chunk_size": chunk_size,
        "overlap": overlap,
    }

    return collection, stats


if __name__ == "__main__":

    collection, stats = (
        index_documents()
    )

    print("Index complete")

    print(
        f"Documents: "
        f"{stats['documents']}"
    )

    print(
        f"Chunks: "
        f"{stats['chunks']}"
    )

    print(
        f"Chunk size: "
        f"{stats['chunk_size']}"
    )

    print(
        f"Overlap: "
        f"{stats['overlap']}"
    )

    print(
        f"Collection count: "
        f"{collection.count()}"
    )