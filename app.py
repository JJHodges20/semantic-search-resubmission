"""Streamlit interface for the semantic search project."""

import streamlit as st

from ingest import (
    DEFAULT_COLLECTION,
    DEFAULT_DB_PATH,
    index_documents,
)

from search import (
    get_collection,
    get_sources,
    semantic_search,
)


st.set_page_config(
    page_title="Course Semantic Search",
    page_icon="🔎",
    layout="wide",
)


@st.cache_resource
def cached_collection():
    """Cache ChromaDB across Streamlit reruns."""

    return get_collection(
        DEFAULT_DB_PATH,
        DEFAULT_COLLECTION,
    )


collection = cached_collection()


# ==========================================================
# Sidebar
# ==========================================================

with st.sidebar:

    st.title("📚 Search Controls")

    chunk_size = st.number_input(
        "Index chunk size",
        min_value=100,
        max_value=1200,
        value=500,
        step=50,
        help=(
            "Character count used when "
            "re-indexing documents."
        ),
    )

    overlap = st.number_input(
        "Chunk overlap",
        min_value=0,
        max_value=300,
        value=75,
        step=25,
    )

    if st.button(
        "🔄 Re-index documents",
        use_container_width=True,
    ):

        if overlap >= chunk_size:

            st.error(
                "Overlap must be smaller "
                "than chunk size."
            )

        else:

            try:

                with st.spinner(
                    "Indexing documents..."
                ):

                    _, stats = (
                        index_documents(
                            chunk_size=int(
                                chunk_size
                            ),
                            overlap=int(
                                overlap
                            ),
                            reset=True,
                        )
                    )

                    cached_collection.clear()

                    collection = (
                        cached_collection()
                    )

                st.success(
                    f"Indexed "
                    f"{stats['documents']} files "
                    f"into "
                    f"{stats['chunks']} chunks."
                )

            except Exception as exc:

                st.error(
                    f"Indexing failed: {exc}"
                )

    total_chunks = (
        collection.count()
    )

    sources = get_sources(
        collection
    )

    col_a, col_b = st.columns(2)

    col_a.metric(
        "Chunks",
        total_chunks,
    )

    col_b.metric(
        "Sources",
        len(sources),
    )

    st.divider()

    n_results = st.slider(
        "Results to show",
        min_value=1,
        max_value=10,
        value=5,
    )

    max_distance = st.slider(
        "Maximum distance",
        min_value=0.10,
        max_value=2.00,
        value=1.00,
        step=0.05,
        help=(
            "Smaller distance means a "
            "closer semantic match."
        ),
    )

    selected_sources = (
        st.multiselect(
            "Source files",
            options=sources,
            default=sources,
        )
    )


# ==========================================================
# Main Interface
# ==========================================================

st.title(
    "🔎 Course Semantic Search"
)

st.write(
    "Ask a question and retrieve the "
    "most semantically similar chunks "
    "from the indexed course documents."
)

query = st.text_input(
    "Search query",
    placeholder=(
        "How does Streamlit preserve "
        "data between reruns?"
    ),
)


# ==========================================================
# Search Results
# ==========================================================

if collection.count() == 0:

    st.info(
        "Use **Re-index documents** "
        "in the sidebar to build the "
        "search index first."
    )

elif query.strip():

    if not selected_sources:

        st.warning(
            "Select at least one source "
            "file in the sidebar."
        )

    else:

        results = semantic_search(
            query=query,
            n_results=n_results,
            max_distance=max_distance,
            sources=selected_sources,
        )

        if not results:

            st.warning(
                "No results met the "
                "current distance threshold. "
                "Try increasing it."
            )

        else:

            st.subheader(
                f"Showing "
                f"{len(results)} result(s)"
            )

            for rank, result in enumerate(
                results,
                start=1,
            ):

                with st.container(
                    border=True
                ):

                    meta_col, score_col = (
                        st.columns(
                            [3, 1]
                        )
                    )

                    with meta_col:

                        st.markdown(
                            f"### {rank}. "
                            f"{result['source']}"
                        )

                        st.caption(
                            f"Chunk "
                            f"{result['chunk_index']}"
                        )

                    with score_col:

                        st.markdown(
                            f"**"
                            f"{result['relevance']}"
                            f"**"
                        )

                        st.caption(
                            f"Distance: "
                            f"{result['distance']:.3f}"
                        )

                    preview = (
                        result["text"][:220]
                    )

                    if (
                        len(result["text"])
                        > 220
                    ):

                        preview += "..."

                    st.write(
                        preview
                    )

                    with st.expander(
                        "View full matched chunk"
                    ):

                        st.write(
                            result["text"]
                        )

elif query == "":

    st.caption(
        "Enter a question above "
        "to begin searching."
    )