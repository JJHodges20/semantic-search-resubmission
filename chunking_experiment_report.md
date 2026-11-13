# Chunking experiment - generated results

All configurations use the same five predefined queries. 
Settings: overlap=50, top-N=3, maximum distance=1.0.

Metrics are *source-level* (unique source filenames); repeated chunks from
one file do not count as additional false positives. Distance scores are
model-specific and should not be treated as calibrated probabilities.

| Chunk size | Mean precision | Mean recall | Mean F1 |
|---|---:|---:|---:|
| 200 | 0.900 | 1.000 | 0.933 |
| 500 | 1.000 | 1.000 | 1.000 |

## Per-query evidence

| Query | Expected source(s) | 200 chars: returned / F1 | 500 chars: returned / F1 |
|---|---|---|---|
| 1 | streamlit_guide.txt | streamlit_guide.txt; streamlit_guide.txt; streamlit_guide.txt / 1.000 | streamlit_guide.txt; streamlit_guide.txt / 1.000 |
| 2 | fastapi_basics.txt | fastapi_basics.txt; fastapi_basics.txt; fastapi_basics.txt / 1.000 | fastapi_basics.txt / 1.000 |
| 3 | embeddings_vector_search.txt | embeddings_vector_search.txt; embeddings_vector_search.txt; rag_systems.txt / 0.667 | embeddings_vector_search.txt; embeddings_vector_search.txt; embeddings_vector_search.txt / 1.000 |
| 4 | database_fundamentals.txt | database_fundamentals.txt; database_fundamentals.txt; database_fundamentals.txt / 1.000 | database_fundamentals.txt; database_fundamentals.txt / 1.000 |
| 5 | prompt_security.txt, rag_systems.txt | prompt_security.txt; rag_systems.txt; rag_systems.txt / 1.000 | prompt_security.txt; rag_systems.txt; rag_systems.txt / 1.000 |

**Winner on average source-level F1:** 500-character chunks.

Individual queries can tie; inspect the CSV for ranked chunks
and scores. This result applies only to this corpus, query set and model.
