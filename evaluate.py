"""Evaluation and chunk-size experiment for semantic search."""

from __future__ import annotations

import csv

from pathlib import Path

from ingest import (
    index_documents,
)

from search import (
    semantic_search,
)


# ==========================================================
# Test Queries
# ==========================================================

TEST_QUERIES = [
    {
        "query": (
            "How does Streamlit keep values "
            "after the script reruns?"
        ),
        "relevant_sources": {
            "streamlit_guide.txt"
        },
    },
    {
        "query": (
            "How does FastAPI validate "
            "incoming request data?"
        ),
        "relevant_sources": {
            "fastapi_basics.txt"
        },
    },
    {
        "query": (
            "How do embeddings make "
            "semantic search possible?"
        ),
        "relevant_sources": {
            "embeddings_vector_search.txt"
        },
    },
    {
        "query": (
            "What is the purpose of a "
            "primary key in a "
            "relational database?"
        ),
        "relevant_sources": {
            "database_fundamentals.txt"
        },
    },
    {
        "query": (
            "How should a RAG assistant "
            "defend against instructions "
            "hidden in retrieved context?"
        ),
        "relevant_sources": {
            "rag_systems.txt",
            "prompt_security.txt",
        },
    },
]


# ==========================================================
# Metrics
# ==========================================================

def precision_recall_f1(
    retrieved_sources: list[str],
    relevant_sources: set[str],
):
    """Calculate source-level precision, recall, and F1."""

    retrieved = set(
        retrieved_sources
    )

    if not retrieved:
        return 0.0, 0.0, 0.0

    true_positive = len(
        retrieved
        & relevant_sources
    )

    precision = (
        true_positive
        / len(retrieved)
    )

    recall = (
        true_positive
        / len(relevant_sources)
        if relevant_sources
        else 0.0
    )

    if precision + recall:

        f1 = (
            2
            * precision
            * recall
            / (precision + recall)
        )

    else:
        f1 = 0.0

    return (
        precision,
        recall,
        f1,
    )


# ==========================================================
# Evaluate One Collection
# ==========================================================

def evaluate_collection(
    collection_name: str,
    chunk_size: int,
    overlap: int,
    n_results: int = 3,
    threshold: float = 1.0,
):
    """Evaluate one collection using the fixed test queries."""

    rows = []

    print()

    print(
        f"=== chunk_size={chunk_size}, "
        f"overlap={overlap}, "
        f"n_results={n_results}, "
        f"threshold={threshold} ==="
    )

    for number, test in enumerate(
        TEST_QUERIES,
        start=1,
    ):

        results = semantic_search(
            query=test["query"],
            n_results=n_results,
            max_distance=threshold,
            collection_name=collection_name,
        )

        retrieved_sources = [
            result["source"]
            for result in results
        ]

        precision, recall, f1 = (
            precision_recall_f1(
                retrieved_sources,
                test[
                    "relevant_sources"
                ],
            )
        )

        print(
            f"Query {number}: "
            f"P={precision * 100:.1f}% "
            f"R={recall * 100:.1f}% "
            f"F1={f1 * 100:.1f}%"
        )

        if results:

            for result in results:

                print(
                    f"  - "
                    f"{result['source']} "
                    f"| chunk "
                    f"{result['chunk_index']} "
                    f"| distance="
                    f"{result['distance']:.3f}"
                )

        else:

            print(
                "  - No results "
                "above threshold"
            )

        rows.append(
            {
                "chunk_size": chunk_size,
                "query_number": number,
                "query": test["query"],
                "expected_sources": (
                    "; ".join(
                        sorted(
                            test[
                                "relevant_sources"
                            ]
                        )
                    )
                ),
                "retrieved_sources": (
                    "; ".join(
                        retrieved_sources
                    )
                ),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "top_distance": (
                    results[0]["distance"]
                    if results
                    else ""
                ),
            }
        )

    average_precision = (
        sum(
            row["precision"]
            for row in rows
        )
        / len(rows)
    )

    average_recall = (
        sum(
            row["recall"]
            for row in rows
        )
        / len(rows)
    )

    average_f1 = (
        sum(
            row["f1"]
            for row in rows
        )
        / len(rows)
    )

    print(
        f"AVERAGE: "
        f"P={average_precision * 100:.1f}% "
        f"R={average_recall * 100:.1f}% "
        f"F1={average_f1 * 100:.1f}%"
    )

    return rows, {
        "precision": average_precision,
        "recall": average_recall,
        "f1": average_f1,
    }


# ==========================================================
# Chunking Experiment
# ==========================================================

def run_chunking_experiment(
    chunk_sizes=(200, 500),
    overlap=50,
    n_results=3,
    threshold=1.0,
    output_csv="chunking_experiment.csv",
    output_report="chunking_experiment_report.md",
):
    """Compare search performance across multiple chunk sizes."""

    all_rows = []

    summaries = {}

    for chunk_size in chunk_sizes:

        collection_name = (
            f"course_docs_"
            f"{chunk_size}"
        )

        _, stats = index_documents(
            collection_name=(
                collection_name
            ),
            chunk_size=chunk_size,
            overlap=overlap,
            reset=True,
        )

        print(
            f"\nIndexed "
            f"{stats['documents']} docs "
            f"into {stats['chunks']} chunks "
            f"for chunk_size="
            f"{chunk_size}."
        )

        rows, summary = (
            evaluate_collection(
                collection_name=(
                    collection_name
                ),
                chunk_size=chunk_size,
                overlap=overlap,
                n_results=n_results,
                threshold=threshold,
            )
        )

        all_rows.extend(
            rows
        )

        summaries[
            chunk_size
        ] = summary

    fieldnames = [
        "chunk_size",
        "query_number",
        "query",
        "expected_sources",
        "retrieved_sources",
        "precision",
        "recall",
        "f1",
        "top_distance",
    ]

    with Path(
        output_csv
    ).open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            all_rows
        )

    print(
        "\n=== CHUNKING COMPARISON ==="
    )

    for (
        chunk_size,
        summary,
    ) in summaries.items():

        print(
            f"{chunk_size} chars -> "
            f"P="
            f"{summary['precision'] * 100:.1f}% "
            f"R="
            f"{summary['recall'] * 100:.1f}% "
            f"F1="
            f"{summary['f1'] * 100:.1f}%"
        )

    best_size = max(
        summaries,
        key=lambda size: (
            summaries[size]["f1"]
        ),
    )

    print(
        f"\nBest average F1 "
        f"in this run: "
        f"{best_size}-character chunks"
    )

    print(
        f"Detailed results saved to: "
        f"{output_csv}"
    )

    # Keep an auditable written comparison alongside the raw CSV.
    by_size = {size: {r["query_number"]: r for r in all_rows
                      if r["chunk_size"] == size} for size in summaries}
    size_columns = list(summaries)
    report = ["# Chunking experiment - generated results", "",
              "All configurations use the same five predefined queries. ",
              f"Settings: overlap={overlap}, top-N={n_results}, maximum distance={threshold}.", "",
              "Metrics are *source-level* (unique source filenames); repeated chunks from",
              "one file do not count as additional false positives. Distance scores are",
              "model-specific and should not be treated as calibrated probabilities.", "",
              "| Chunk size | Mean precision | Mean recall | Mean F1 |",
              "|---|---:|---:|---:|"]
    for size, summary in summaries.items():
        report.append(f"| {size} | {summary['precision']:.3f} | {summary['recall']:.3f} | {summary['f1']:.3f} |")
    report += ["", "## Per-query evidence", "",
               "| Query | Expected source(s) | " + " | ".join(f"{size} chars: returned / F1" for size in size_columns) + " |",
               "|---|---|" + "---|" * len(size_columns)]
    for number, test in enumerate(TEST_QUERIES, 1):
        parts = [str(number), ", ".join(sorted(test["relevant_sources"]))]
        for size in size_columns:
            row = by_size[size][number]
            parts.append(f"{row['retrieved_sources'] or '(none)'} / {row['f1']:.3f}")
        report.append("| " + " | ".join(parts) + " |")
    report += ["", f"**Winner on average source-level F1:** {best_size}-character chunks.",
               "", "Individual queries can tie; inspect the CSV for ranked chunks",
               "and scores. This result applies only to this corpus, query set and model.",
               ""]
    Path(output_report).write_text("\n".join(report), encoding="utf-8")
    print(f"Generated experiment report: {output_report}")

    print(
        "\nANALYSIS GUIDE:"
    )

    print(
        "- Look for queries where one "
        "chunk size retrieves the expected "
        "source earlier."
    )

    print(
        "- Smaller chunks can isolate "
        "precise facts but may lose "
        "surrounding context."
    )

    print(
        "- Larger chunks preserve context "
        "but can mix several concepts "
        "inside one embedding."
    )

    print(
        "- Use the CSV to write the README "
        "findings from your actual results."
    )


if __name__ == "__main__":

    run_chunking_experiment()