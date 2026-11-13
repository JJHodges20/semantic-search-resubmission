"""Semantic search functions for ChromaDB."""

from __future__ import annotations

from typing import Iterable

from ingest import (
    DEFAULT_COLLECTION,
    DEFAULT_DB_PATH,
    get_client,
)


def get_collection(
    db_path: str = DEFAULT_DB_PATH,
    collection_name: str = DEFAULT_COLLECTION,
):
    """Open or create a persistent ChromaDB collection."""

    client = get_client(
        db_path
    )

    return client.get_or_create_collection(
        name=collection_name
    )


def get_sources(
    collection,
) -> list[str]:
    """Return unique source filenames in the collection."""

    if collection.count() == 0:
        return []

    data = collection.get(
        include=["metadatas"]
    )

    metadatas = (
        data.get("metadatas")
        or []
    )

    return sorted(
        {
            metadata["source"]
            for metadata in metadatas
            if metadata
            and "source" in metadata
        }
    )


def relevance_label(
    distance: float,
) -> str:
    """Convert distance into a relevance label."""

    if distance <= 0.45:
        return "🟢 High"

    if distance <= 0.85:
        return "🟡 Medium"

    return "🔴 Low"


def semantic_search(
    query: str,
    n_results: int = 5,
    max_distance: float = 1.0,
    sources: Iterable[str] | None = None,
    db_path: str = DEFAULT_DB_PATH,
    collection_name: str = DEFAULT_COLLECTION,
) -> list[dict]:
    """Search ChromaDB and return ranked matching chunks."""

    query = query.strip()

    if not query:
        return []

    if n_results <= 0:
        return []

    collection = get_collection(
        db_path,
        collection_name,
    )

    if collection.count() == 0:
        return []

    query_args = {
        "query_texts": [query],
        "n_results": min(
            n_results,
            collection.count(),
        ),
        "include": [
            "documents",
            "metadatas",
            "distances",
        ],
    }

    selected_sources = list(sources) if sources is not None else []
    if sources is not None and not selected_sources:
        return []

    # Apply source filtering through ChromaDB.
    if selected_sources:

        if len(selected_sources) == 1:

            query_args["where"] = {
                "source": selected_sources[0]
            }

        else:

            query_args["where"] = {
                "source": {
                    "$in": selected_sources
                }
            }

    results = collection.query(
        **query_args
    )

    documents = (
        results.get("documents")
        or [[]]
    )[0]

    metadatas = (
        results.get("metadatas")
        or [[]]
    )[0]

    distances = (
        results.get("distances")
        or [[]]
    )[0]

    ids = (
        results.get("ids")
        or [[]]
    )[0]

    ranked = []

    for (
        chunk_id,
        text,
        metadata,
        distance,
    ) in zip(
        ids,
        documents,
        metadatas,
        distances,
    ):

        if distance <= max_distance:

            ranked.append(
                {
                    "id": chunk_id,
                    "text": text,
                    "source": metadata.get(
                        "source",
                        "Unknown",
                    ),
                    "document_id": metadata.get(
                        "document_id",
                        "Unknown",
                    ),
                    "chunk_index": metadata.get(
                        "chunk_index",
                        0,
                    ),
                    "distance": float(
                        distance
                    ),
                    "relevance": relevance_label(
                        float(distance)
                    ),
                }
            )

    return ranked