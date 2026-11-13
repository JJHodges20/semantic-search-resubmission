# Semantic Search Engine | Coding Temple Resubmission

A local retrieval application that loads eight course-topic documents, chunks them with configurable character size and overlap, embeds them in **persistent ChromaDB**, and displays ranked matches in **Streamlit**.

## Project layout

```text
semantic-search-engine/
├── app.py                        # Streamlit UI and sidebar controls
├── ingest.py                     # Load, chunk, index and persist
├── search.py                     # Search, metadata filters, ranked results
├── evaluate.py                   # Five-query precision/recall/F1 experiment
├── docs/                         # Eight .txt source documents
├── chunking_experiment.csv       # Recorded results from original run
├── chunking_experiment_report.md # Evidence and interpretation
├── requirements.txt
├── .gitignore
└── README.md
```

`chroma_data/` is created when indexing and intentionally excluded from version control. The dataset contains eight `.txt` files about FastAPI, Streamlit, databases, RAG, prompt security, APIs, embeddings, and web development. To confirm your instructor's five-page threshold, check the actual rendered pagination; source text is roughly 8 KB and page count depends on formatting.

## Setup (Windows PowerShell / VS Code)

From the project folder:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python ingest.py
python -m streamlit run app.py
```

If `py` is unavailable, invoke the **actual Python interpreter** using its full path; do not use a Windows Store alias. Python 3.11 or 3.12 is a conservative choice for dependency compatibility. The package ranges in `requirements.txt` intentionally constrain major versions; for a strict reproducible environment, save and submit a tested `python -m pip freeze` lock snapshot.

## Architecture / data flow

1. `ingest.load_documents()` reads nonempty `.txt` and `.md` files in `docs/`.
2. `chunk_text()` splits normalized text into overlapping, character-based segments and avoids splitting words when possible.
3. `build_chunks()` attaches a stable chunk ID, source filename, document ID, index, chunk size, and overlap.
4. ChromaDB embeds and stores text in a persistent local collection (`chroma_data/`). The default embedding function may download a model on first use, so the first run may require Internet access.
5. `semantic_search()` queries ChromaDB with natural language, applies a source metadata filter inside the query, enforces maximum distance, and returns ranked chunks.
6. `app.py` displays source, text, distance, relevance label, indexing controls, statistics, and feedback for blank/no-match queries.
7. `evaluate.py` rebuilds two independent collections and compares identical questions. It writes the CSV **and** a generated Markdown report of the new run.

**Distance note:** smaller Chroma distances indicate closer matches, but the simple High/Medium/Low relevance labels are heuristic, not calibrated probabilities.

## How to use the UI

- Run `python ingest.py` first or click **Re-index documents** in the sidebar.
- Type a question. Use the **Results to show** slider (top N) and **Maximum distance** slider (threshold).
- Use the **Source files** multiselect to restrict retrieval; the filter is passed to ChromaDB rather than merely hiding results after retrieval.
- Inspect document count, number of indexed chunks, filename, chunk index, distance, relevance indicator, and matched excerpt.
- When you change chunk size/overlap, click **Re-index documents** so the stored vectors match the controls.

## Chunking experiment and evidence

Run:

```powershell
python evaluate.py
```

The script rebuilds the 200- and 500-character collections with 50-character overlap, executes the same five test queries against both, computes source-level precision/recall/F1, and saves `chunking_experiment.csv` and `chunking_experiment_report.md`.

**Recorded baseline in uploaded project** (not rerun during this documentation revision):

| Chunk size | Mean precision | Mean recall | Mean F1 |
|---|---:|---:|---:|
| 200 | 0.900 | 1.000 | 0.933 |
| 500 | 1.000 | 1.000 | 1.000 |

| Test question | 200 F1 | 500 F1 | Better |
|---|---:|---:|---|
| Streamlit session state | 1.000 | 1.000 | Tie |
| FastAPI request validation | 1.000 | 1.000 | Tie |
| How embeddings support search | 0.667 | 1.000 | 500 |
| Purpose of primary keys | 1.000 | 1.000 | Tie |
| Defending RAG from injected instructions | 1.000 | 1.000 | Tie |

**Finding:** 500-character chunks were better overall under this experiment's *source-level F1* metric. The difference came entirely from the embedding question, where 200-character search returned an extra irrelevant source (`rag_systems.txt`). Four of five queries tied. A plausible explanation is that 500-character chunks preserve more context about a topic, although this small test does not prove causal improvement.

**Limitations:** The evaluation scores unique source filenames, not individual passage relevance; duplicate chunks from the same source do not count as additional false positives. The query set is small and based on the source corpus. A larger, independent test set and manual relevance labels would make the comparison more reliable. Read `chunking_experiment_report.md` and the CSV for full per-query evidence, retrieved filenames and distance scores.

## Verification and presentation

```powershell
python -m compileall -q app.py ingest.py search.py evaluate.py
python -m pytest -q
python evaluate.py
python -m streamlit run app.py
```

Install `pytest` separately (`python -m pip install pytest`) to run the optional development tests; it is not a runtime dependency.

**Five-minute demonstration:** (1) show a search and source + distance + label (about 2 minutes); (2) demonstrate real source filtering and reindexing, then compare the five-row experiment table (about 2 minutes); (3) explain the source-level metric limitation and one improvement (about 1 minute).

## Engineering limitations and next steps

The current program is a retrieval tool, **not** a generative question-answering assistant. It is designed for a small local corpus; production deployment would benefit from a locked environment, larger test corpus, caching, embedding-model configuration, batch handling, security/privacy checks, and a human-labeled evaluation set. Avoid committing personal documents or a prebuilt `chroma_data/` database to Git.
