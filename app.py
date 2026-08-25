from __future__ import annotations

import shutil
import uuid
import tempfile
from pathlib import Path

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

import config
from src import ingestion, vectorstore
from src.exceptions import (
    DocumentLoadError,
    ProviderConfigurationError,
    RetrievalError,
    VectorStoreError,
)
from src.llm_factory import get_embeddings, get_llm
from src.rag_chain import ConversationalRAGChain, RetrievalDiagnostics


st.set_page_config(
    page_title="DocuMind — RAG Chatbot",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


LLM_PROVIDER_LABELS = {
    "groq": "Groq",
    "openai": "OpenAI",
    "ollama": "Ollama",
}

EMBEDDING_PROVIDER_LABELS = {
    "huggingface": "HuggingFace",
    "openai": "OpenAI",
    "ollama": "Ollama",
}

DEFAULT_MODEL_BY_PROVIDER = {
    "groq": config.GROQ_MODEL,
    "openai": config.OPENAI_MODEL,
    "ollama": config.OLLAMA_MODEL,
}

DEFAULT_EMBEDDING_MODEL_BY_PROVIDER = {
    "huggingface": config.HUGGINGFACE_EMBEDDING_MODEL,
    "openai": config.OPENAI_EMBEDDING_MODEL,
    "ollama": config.OLLAMA_EMBEDDING_MODEL,
}

RESTORED_SENTINEL = "(restored from persisted index)"


st.markdown(
    """
    <style>
        .block-container {
            max-width: 1400px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        .dm-hero {
            padding: 1.4rem 1.6rem;
            margin-bottom: 1.2rem;
            border: 1px solid rgba(128, 128, 128, 0.20);
            border-radius: 18px;
            background: var(--secondary-background-color);
        }

        .dm-hero-centered {
            text-align: center;
        }

        .dm-welcome {
            max-width: 760px;
            margin: 8vh auto 4vh;
            padding: 2.8rem 2rem;
            text-align: center;
        }

        .dm-welcome-icon {
            font-size: 3rem;
            margin-bottom: 0.6rem;
        }

        .dm-welcome h2 {
            margin-bottom: 0.5rem;
        }

        .dm-welcome p {
            max-width: 650px;
            margin: 0.5rem auto;
            line-height: 1.6;
        }

        .dm-hero-title {
            margin: 0;
            font-size: 2rem;
            font-weight: 700;
        }

        .dm-hero-subtitle {
            margin-top: 0.45rem;
            color: var(--text-color);
            opacity: 0.72;
            font-size: 0.95rem;
        }

        .dm-status {
            padding: 0.8rem 1rem;
            border: 1px solid rgba(128, 128, 128, 0.18);
            border-radius: 12px;
            background: var(--secondary-background-color);
        }

        .dm-muted {
            opacity: 0.70;
            font-size: 0.88rem;
        }

        [data-testid="stSidebar"] {
            border-right: 1px solid rgba(128, 128, 128, 0.14);
        }

        [data-testid="stExpander"] {
            border-radius: 12px;
        }

        .stButton > button,
        .stDownloadButton > button {
            border-radius: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def _initialize_session_state() -> None:
    defaults = {
        "messages": [],
        "vectorstore": None,
        "kb_chunk_count": 0,
        "kb_sources": [],
        "kb_embedding_provider": None,
        "kb_embedding_model": None,
        "startup_load_attempted": False,
        "groq_api_key": "",
        "openai_api_key": "",
        "kb_collection_name": None,
    }

    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


_initialize_session_state()
if st.session_state.kb_collection_name is None:
    st.session_state.kb_collection_name = (
        f"{config.CHROMA_COLLECTION_NAME}_{uuid.uuid4().hex}"
    )

@st.cache_resource(show_spinner="Loading embedding model...")
def _load_embeddings(
    provider: str,
    model: str,
    api_key: str | None,
):
    return get_embeddings(
        provider=provider,
        model=model,
        api_key=api_key,
    )


@st.cache_resource(show_spinner=False)
def _load_llm(
    provider: str,
    model: str,
    api_key: str | None,
    base_url: str | None,
):
    return get_llm(
        provider=provider,
        model=model,
        api_key=api_key,
        base_url=base_url,
    )


def _get_embedding_api_key(provider: str) -> str | None:
    if provider == "openai":
        return st.session_state.openai_api_key or None
    return None


def _save_uploads_to_temp(
    uploaded_files,
) -> tuple[list[str], Path | None]:
    if not uploaded_files:
        return [], None

    tmp_dir = Path(
        tempfile.mkdtemp(prefix="rag_upload_")
    )

    paths: list[str] = []

    try:
        for uploaded_file in uploaded_files:
            safe_name = Path(uploaded_file.name).name
            destination = tmp_dir / safe_name
            destination.write_bytes(
                uploaded_file.getvalue()
            )
            paths.append(str(destination))

        return paths, tmp_dir

    except Exception:
        shutil.rmtree(
            tmp_dir,
            ignore_errors=True,
        )
        raise


def _history_as_messages(
    messages: list[dict],
) -> list:
    history = []

    for message in messages:
        if message.get("role") == "user":
            history.append(
                HumanMessage(
                    content=message.get("content", "")
                )
            )
        elif message.get("role") == "assistant":
            history.append(
                AIMessage(
                    content=message.get("content", "")
                )
            )

    return history


def _render_sources(
    sources: list,
) -> None:
    if not sources:
        return

    with st.expander(
        f"Sources · {len(sources)} chunk(s)",
        expanded=False,
    ):
        for index, document in enumerate(
            sources,
            start=1,
        ):
            label = (
                document.metadata.get("display_source")
                or document.metadata.get("source")
                or f"source {index}"
            )

            preview = document.page_content.strip()

            st.markdown(
                f"**[{index}] {label}**"
            )

            if len(preview) > 400:
                preview = preview[:400] + "..."

            st.caption(preview)


def _render_diagnostics(
    diagnostics: RetrievalDiagnostics | None,
) -> None:
    if diagnostics is None:
        return

    with st.expander(
        "Retrieval diagnostics",
        expanded=False,
    ):
        st.caption(
            f"Standalone query: "
            f"{diagnostics.standalone_question!r}"
        )

        st.caption(
            f"Retrieved: "
            f"{diagnostics.retrieved_count} chunk(s)"
        )

        if diagnostics.scores:
            scores = ", ".join(
                f"{score:.3f}"
                for score in diagnostics.scores
            )
            st.caption(
                f"Scores: {scores}"
            )

        threshold = (
            "disabled"
            if diagnostics.score_threshold is None
            else f"{diagnostics.score_threshold:.2f}"
        )

        st.caption(
            f"Threshold: {threshold} · "
            f"Abstained: {diagnostics.abstained}"
        )

        latency = (
            f"{diagnostics.retrieval_latency_ms:.0f} ms retrieval"
        )

        if diagnostics.generation_latency_ms is not None:
            latency += (
                f" + "
                f"{diagnostics.generation_latency_ms:.0f} ms generation"
            )

        st.caption(latency)


def _try_auto_load_existing_kb() -> bool:
    if st.session_state.vectorstore is not None:
        return True

    metadata = vectorstore.load_index_metadata(
    collection_name=st.session_state.kb_collection_name,
        )

    if metadata is None:
        return False

    embedding_api_key = _get_embedding_api_key(
        metadata.embedding_provider
    )

    try:
        embeddings = _load_embeddings(
            metadata.embedding_provider,
            metadata.embedding_model,
            embedding_api_key,
        )

        vector_store = vectorstore.load_vectorstore(
            embeddings=embeddings,
            embedding_provider=metadata.embedding_provider,
            embedding_model=metadata.embedding_model,
            collection_name=st.session_state.kb_collection_name,
        )

        st.session_state.vectorstore = vector_store
        st.session_state.kb_chunk_count = (
            vector_store._collection.count()
        )
        st.session_state.kb_embedding_provider = (
            metadata.embedding_provider
        )
        st.session_state.kb_embedding_model = (
            metadata.embedding_model
        )

        try:
            st.session_state.kb_sources = (
                vectorstore.get_distinct_sources(vector_store)
            )
        except VectorStoreError:
            st.session_state.kb_sources = [
                RESTORED_SENTINEL
            ]

        return True

    except (
        ProviderConfigurationError,
        VectorStoreError,
    ):
        return False


if not st.session_state.startup_load_attempted:
    st.session_state.startup_load_attempted = True
    _try_auto_load_existing_kb()


with st.sidebar:
    st.markdown("## Configuration")
    st.caption(
        "Model, embeddings, retrieval, and knowledge-base controls."
    )

    kb_tab, models_tab, advanced_tab = st.tabs(
        ["Knowledge Base", "Models", "Advanced"]
    )

    with models_tab:
        with st.container(border=True):
            st.markdown("### Chat model")

            llm_provider = st.selectbox(
                "Provider",
                options=list(LLM_PROVIDER_LABELS.keys()),
                index=list(LLM_PROVIDER_LABELS.keys()).index(
                    config.LLM_PROVIDER
                ),
                format_func=lambda provider: LLM_PROVIDER_LABELS[provider],
            )

            llm_model = st.text_input(
                "Model",
                value=DEFAULT_MODEL_BY_PROVIDER[llm_provider],
            )

            ollama_base_url = config.OLLAMA_BASE_URL

            if llm_provider == "ollama":
                ollama_base_url = st.text_input(
                    "Ollama base URL",
                    value=config.OLLAMA_BASE_URL,
                )

        with st.container(border=True):
            st.markdown("### Embeddings")

            embedding_provider = st.selectbox(
                "Provider",
                options=list(EMBEDDING_PROVIDER_LABELS.keys()),
                index=list(EMBEDDING_PROVIDER_LABELS.keys()).index(
                    config.EMBEDDING_PROVIDER
                ),
                format_func=lambda provider: (
                    EMBEDDING_PROVIDER_LABELS[provider]
                ),
            )

            embedding_model = st.text_input(
                "Model",
                value=DEFAULT_EMBEDDING_MODEL_BY_PROVIDER[
                    embedding_provider
                ],
            )

    with advanced_tab:
        with st.container(border=True):
            st.markdown("### Retrieval")

            enable_abstention = st.checkbox(
                "Enable relevance abstention",
                value=(
                    config.RETRIEVAL_SCORE_THRESHOLD is not None
                ),
                help=(
                    "Abstain when the strongest retrieved "
                    "chunk is below the configured threshold."
                ),
            )

            score_threshold = None

            if enable_abstention:
                default_threshold = (
                    config.RETRIEVAL_SCORE_THRESHOLD
                    if config.RETRIEVAL_SCORE_THRESHOLD is not None
                    else 0.2
                )

                score_threshold = st.slider(
                    "Relevance threshold",
                    min_value=-1.0,
                    max_value=1.0,
                    value=float(default_threshold),
                    step=0.05,
                )

            show_diagnostics = st.checkbox(
                "Show retrieval diagnostics",
                value=config.ENABLE_RETRIEVAL_DIAGNOSTICS,
            )

        with st.container(border=True):
            st.markdown("### API keys")

            st.session_state.groq_api_key = st.text_input(
                "Groq API key",
                type="password",
                value=st.session_state.groq_api_key,
                help="Used only for the selected Groq chat provider.",
            )

            st.session_state.openai_api_key = st.text_input(
                "OpenAI API key",
                type="password",
                value=st.session_state.openai_api_key,
                help="Used for OpenAI chat or OpenAI embeddings.",
            )

    with kb_tab:
        with st.container(border=True):
            st.markdown("### Sources")

            existing_index = vectorstore.index_exists(
                collection_name=st.session_state.kb_collection_name,
            )

            if (
                st.session_state.vectorstore is None
                and existing_index
            ):
                st.info("A persisted knowledge base is available.")

                if st.button(
                    "Reload knowledge base",
                    use_container_width=True,
                ):
                    if _try_auto_load_existing_kb():
                        st.toast("Knowledge base loaded!", icon="📚")
                        st.rerun()
                    else:
                        st.error(
                            "The existing knowledge base could not be loaded. "
                            "Check the required embedding provider configuration."
                        )

            uploaded_files = st.file_uploader(
                "Upload documents",
                type=["pdf", "txt", "md"],
                accept_multiple_files=True,
                help="PDF, TXT, and Markdown files.",
            )

            url_text = st.text_area(
                "Web URLs",
                height=90,
                placeholder=(
                    "https://example.com/documentation\n"
                    "https://example.com/guide"
                ),
            )

            build_clicked = st.button(
                "Build knowledge base",
                type="primary",
                use_container_width=True,
            )

        if build_clicked:
            temp_dir: Path | None = None

            try:
                file_paths, temp_dir = _save_uploads_to_temp(
                    uploaded_files
                )

                urls = [
                    url.strip()
                    for url in url_text.splitlines()
                    if url.strip()
                ]

                if not file_paths and not urls:
                    st.warning("Add at least one file or URL.")
                else:
                    with st.status(
                        "Building Knowledge Base...",
                        expanded=True,
                    ) as build_status:
                        build_status.update(
                            label="Reading and chunking sources..."
                        )

                        chunks = ingestion.ingest(
                            file_paths=file_paths,
                            urls=urls,
                        )
                        if not chunks:
                            build_status.update(
                                label="No readable text found.",
                                state="error",
                            )
                            st.warning("No readable text was found.")
                        else:
                            build_status.update(
                                label=(
                                    f"Chunking complete — "
                                    f"{len(chunks):,} chunks created."
                                )
                            )

                            embedding_api_key = _get_embedding_api_key(
                                embedding_provider
                            )

                            build_status.update(
                                label=(
                                    f"Loading {embedding_provider} "
                                    "embedding model..."
                                )
                            )

                            embeddings = _load_embeddings(
                                embedding_provider,
                                embedding_model,
                                embedding_api_key,
                            )

                            build_status.update(
                                label=(
                                    f"Embedding and indexing "
                                    f"{len(chunks):,} chunks..."
                                )
                            )

                            vector_store = vectorstore.build_vectorstore(
                                chunks=chunks,
                                embeddings=embeddings,
                                embedding_provider=embedding_provider,
                                embedding_model=embedding_model,
                                collection_name=st.session_state.kb_collection_name,
                            )

                            sources = sorted(
                                {
                                    chunk.metadata.get(
                                        "source",
                                        "unknown",
                                    )
                                    for chunk in chunks
                                }
                            )

                            st.session_state.vectorstore = vector_store
                            st.session_state.kb_chunk_count = len(chunks)
                            st.session_state.kb_sources = sources
                            st.session_state.kb_embedding_provider = (
                                embedding_provider
                            )
                            st.session_state.kb_embedding_model = (
                                embedding_model
                            )
                            st.session_state.messages = []

                            build_status.update(
                                label="Knowledge Base ready!",
                                state="complete",
                                expanded=False,
                            )
                            st.toast(
                                "Knowledge base ready!",
                                icon="📚",
                            )

            except DocumentLoadError as exc:
                st.error(f"Document ingestion failed: {exc}")

            except (
                VectorStoreError,
                ProviderConfigurationError,
            ) as exc:
                st.error(f"Knowledge-base build failed: {exc}")

            except Exception as exc:
                st.error(f"Unexpected build error: {exc}")

            finally:
                if (
                    temp_dir is not None
                    and temp_dir.exists()
                ):
                    shutil.rmtree(
                        temp_dir,
                        ignore_errors=True,
                    )

        if st.session_state.kb_chunk_count:
            st.markdown(
                f"""
                <div class="dm-status">
                    <strong>Knowledge base ready</strong>
                    <br>
                    <span class="dm-muted">
                        {st.session_state.kb_chunk_count:,}
                        chunks ·
                        {len(st.session_state.kb_sources)}
                        source(s)
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.write("")

        if st.button(
            "Clear chat",
            use_container_width=True,
        ):
            st.session_state.messages = []
            st.toast("Chat cleared!", icon="🧹")
            st.rerun()


st.markdown(
    """
    <div class="dm-hero dm-hero-centered">
        <div class="dm-hero-title">DocuMind</div>
        <div class="dm-hero-subtitle">
            Document-grounded conversational RAG with source-aware retrieval.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


if st.session_state.vectorstore is None:
    st.markdown(
        """
        <div class="dm-welcome">
            <div class="dm-welcome-icon">📚</div>
            <h2>Welcome to DocuMind</h2>
            <p>
                Build a document-grounded knowledge base and ask questions
                using conversational RAG with source-aware retrieval.
            </p>
            <p class="dm-muted">
                Start by adding PDFs, text/Markdown files, or web URLs
                from the <strong>Knowledge Base</strong> tab in the sidebar.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    with st.container(border=True):
        st.markdown(
            f"**Knowledge base:** "
            f"{st.session_state.kb_chunk_count:,} chunks"
        )
        st.caption(
            " · ".join(
                [
                    str(
                        st.session_state.kb_embedding_provider
                        or "unknown"
                    ),
                    str(
                        st.session_state.kb_embedding_model
                        or "unknown"
                    ),
                ]
            )
        )

st.write("")

for message in st.session_state.messages:
    role = message.get("role", "assistant")

    with st.chat_message(role, avatar="👤" if role == "user" else "📚"):
        st.markdown(
            message.get("content", "")
        )

        if message.get("sources"):
            _render_sources(
                message["sources"]
            )

        if show_diagnostics:
            _render_diagnostics(
                message.get("diagnostics")
            )

question = st.chat_input(
    "Ask a question about your documents..."
)

if question:
    question = question.strip()

    if not question:
        st.warning(
            "Please enter a question."
        )
        st.stop()

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
            "sources": None,
            "diagnostics": None,
        }
    )

    with st.chat_message("user", avatar="👤"):
        st.markdown(question)

    with st.chat_message("assistant", avatar="📚"):
        answer = None
        sources = []
        diagnostics = None

        try:
            if llm_provider == "ollama":
                llm = _load_llm(
                    "ollama",
                    llm_model,
                    None,
                    ollama_base_url,
                )
            elif llm_provider == "groq":
                llm = _load_llm(
                    "groq",
                    llm_model,
                    st.session_state.groq_api_key
                    or None,
                    None,
                )
            else:
                llm = _load_llm(
                    "openai",
                    llm_model,
                    st.session_state.openai_api_key
                    or None,
                    None,
                )

            chain = ConversationalRAGChain(
                vectorstore=(
                    st.session_state.vectorstore
                ),
                llm=llm,
                score_threshold=(
                    score_threshold
                    if enable_abstention
                    else None
                ),
            )

            history = _history_as_messages(
                st.session_state.messages[:-1]
            )

            token_stream, sources, diagnostics = (
                chain.stream(
                    question,
                    chat_history=history,
                )
            )

            answer = st.write_stream(
                token_stream
            )

            _render_sources(sources)

            if show_diagnostics:
                _render_diagnostics(
                    diagnostics
                )

        except (
            VectorStoreError,
            ProviderConfigurationError,
            RetrievalError,
        ) as exc:
            answer = (
                "Unable to answer the question: "
                f"{exc}"
            )
            st.error(answer)

        except Exception as exc:
            answer = (
                "An unexpected error occurred "
                "while generating the answer."
            )
            st.error(answer)

            if show_diagnostics:
                st.caption(
                    f"Debug detail: {exc}"
                )

        if answer is not None:
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                    "diagnostics": diagnostics,
                }
            )