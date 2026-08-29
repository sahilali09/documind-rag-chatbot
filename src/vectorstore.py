from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from rank_bm25 import BM25Okapi

import config
from src.exceptions import VectorStoreError

# Manage persistent Chroma indexes and hybrid semantic/lexical retrieval.

INDEX_META_SUFFIX = ".index_meta.json"

SEMANTIC_CANDIDATE_K = 8
LEXICAL_CANDIDATE_K = 8
RRF_K = 60
NEIGHBOR_WINDOW = 1


# Store retrieval-affecting settings separately so incompatible indexes are rejected.
@dataclass(frozen=True)
class IndexMetadata:
    """Configuration recorded when a Chroma index was built."""

    embedding_provider: str
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    collection_name: str
    index_version: str


def _meta_path(
    persist_directory: str,
    collection_name: str,
) -> Path:
    return Path(persist_directory) / f"{collection_name}{INDEX_META_SUFFIX}"


def _save_index_metadata(
    persist_directory: str,
    meta: IndexMetadata,
) -> None:
    path = _meta_path(persist_directory, meta.collection_name)
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        path.write_text(
            json.dumps(asdict(meta), indent=2),
            encoding="utf-8",
        )
    except OSError as exc:
        raise VectorStoreError(
            f"Failed to save index metadata at {path}: {exc}"
        ) from exc


def load_index_metadata(
    persist_directory: str | None = None,
    collection_name: str | None = None,
) -> IndexMetadata | None:
    """Load index metadata, or return None when absent."""
    persist_directory = persist_directory or config.CHROMA_PERSIST_DIR
    collection_name = collection_name or config.CHROMA_COLLECTION_NAME

    path = _meta_path(persist_directory, collection_name)

    if not path.exists():
        return None

    try:
        payload: Any = json.loads(
            path.read_text(encoding="utf-8")
        )

        if not isinstance(payload, dict):
            raise TypeError("metadata JSON must contain an object")

        return IndexMetadata(**payload)

    except (json.JSONDecodeError, TypeError, ValueError, KeyError) as exc:
        raise VectorStoreError(
            f"Index metadata at {path} is corrupted or invalid: {exc}"
        ) from exc


def index_exists(
    persist_directory: str | None = None,
    collection_name: str | None = None,
) -> bool:
    """Return whether valid persisted index metadata exists."""
    return (
        load_index_metadata(
            persist_directory,
            collection_name,
        )
        is not None
    )


def check_compatibility(
    embedding_provider: str,
    embedding_model: str,
    persist_directory: str | None = None,
    collection_name: str | None = None,
) -> tuple[bool, str | None]:
    """Validate current configuration against an existing index."""
    existing = load_index_metadata(
        persist_directory,
        collection_name,
    )

    if existing is None:
        return False, "No existing index found at this location."

    current_collection = (
        collection_name or config.CHROMA_COLLECTION_NAME
    )

    if existing.collection_name != current_collection:
        return False, (
            f"This index belongs to collection "
            f"'{existing.collection_name}', but the current setting is "
            f"'{current_collection}'. Rebuild the knowledge base."
        )

    if (
        existing.embedding_provider != embedding_provider
        or existing.embedding_model != embedding_model
    ):
        return False, (
            f"This index was built with embeddings "
            f"'{existing.embedding_provider}/{existing.embedding_model}', "
            f"but the current setting is "
            f"'{embedding_provider}/{embedding_model}'. "
            "These are different embedding spaces. "
            "Rebuild the knowledge base."
        )

    if existing.chunk_size != config.CHUNK_SIZE:
        return False, (
            f"This index was built with chunk_size="
            f"{existing.chunk_size}, but the current setting is "
            f"{config.CHUNK_SIZE}. Rebuild the knowledge base."
        )

    if existing.chunk_overlap != config.CHUNK_OVERLAP:
        return False, (
            f"This index was built with chunk_overlap="
            f"{existing.chunk_overlap}, but the current setting is "
            f"{config.CHUNK_OVERLAP}. Rebuild the knowledge base."
        )

    if existing.index_version != config.INDEX_VERSION:
        return False, (
            f"This index was built with index_version="
            f"{existing.index_version}, but the app now expects "
            f"index_version={config.INDEX_VERSION}. "
            "Rebuild the knowledge base."
        )

    return True, None


class _UnusedEmbeddings(Embeddings):
    """Minimal embedding implementation used only for deletion."""

    def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        return [[0.0] for _ in texts]

    def embed_query(
        self,
        text: str,
    ) -> list[float]:
        return [0.0]


def reset_collection(
    persist_directory: str | None = None,
    collection_name: str | None = None,
) -> None:
    """Delete the configured Chroma collection and metadata sidecar."""
    persist_directory = persist_directory or config.CHROMA_PERSIST_DIR
    collection_name = collection_name or config.CHROMA_COLLECTION_NAME

    try:
        vectorstore = Chroma(
            embedding_function=_UnusedEmbeddings(),
            collection_name=collection_name,
            persist_directory=persist_directory,
        )
        vectorstore.delete_collection()

    except Exception as exc:
        message = str(exc).lower()

        # A missing collection is already reset, keeping initial and replacement builds idempotent.
        missing_markers = (
            "does not exist",
            "not found",
            "no such collection",
        )

        if not any(
            marker in message
            for marker in missing_markers
        ):
            raise VectorStoreError(
                f"Failed to reset collection '{collection_name}': {exc}"
            ) from exc

    meta_path = _meta_path(
        persist_directory,
        collection_name,
    )

    try:
        meta_path.unlink(missing_ok=True)
    except OSError as exc:
        raise VectorStoreError(
            f"Failed to remove index metadata at {meta_path}: {exc}"
        ) from exc


def build_vectorstore(
    chunks: list[Document],
    embeddings: Embeddings,
    embedding_provider: str,
    embedding_model: str,
    persist_directory: str | None = None,
    collection_name: str | None = None,
) -> Chroma:
    """Replace the existing persisted collection with a fresh index."""
    persist_directory = persist_directory or config.CHROMA_PERSIST_DIR
    collection_name = collection_name or config.CHROMA_COLLECTION_NAME

    if not chunks:
        raise VectorStoreError(
            "No chunks to index — ingestion produced zero usable chunks."
        )

    reset_collection(
        persist_directory,
        collection_name,
    )

    try:
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name=collection_name,
            persist_directory=persist_directory,
            # Scores are normalized as 1 - distance during relevance retrieval.
            collection_metadata={"hnsw:space": "cosine"},
        )
    except Exception as exc:
        raise VectorStoreError(
            f"Failed to build the vector store: {exc}"
        ) from exc

    metadata = IndexMetadata(
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        collection_name=collection_name,
        index_version=config.INDEX_VERSION,
    )

    try:
        _save_index_metadata(
            persist_directory,
            metadata,
        )
    # Roll back the collection rather than leaving it without compatibility metadata.
    except VectorStoreError:
        try:
            reset_collection(
                persist_directory,
                collection_name,
            )
        except VectorStoreError:
            pass
        raise

    return vectorstore


def load_vectorstore(
    embeddings: Embeddings,
    embedding_provider: str,
    embedding_model: str,
    persist_directory: str | None = None,
    collection_name: str | None = None,
) -> Chroma:
    """Reload an existing persisted Chroma collection."""
    persist_directory = persist_directory or config.CHROMA_PERSIST_DIR
    collection_name = collection_name or config.CHROMA_COLLECTION_NAME

    compatible, reason = check_compatibility(
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
        persist_directory=persist_directory,
        collection_name=collection_name,
    )

    if not compatible:
        raise VectorStoreError(
            reason or "Existing index is incompatible."
        )

    try:
        vectorstore = Chroma(
            embedding_function=embeddings,
            collection_name=collection_name,
            persist_directory=persist_directory,
        )

        vectorstore._collection.count()

        return vectorstore

    except Exception as exc:
        raise VectorStoreError(
            f"Failed to load the existing vector store: {exc}"
        ) from exc


def add_to_vectorstore(
    vectorstore: Chroma,
    chunks: list[Document],
) -> None:
    """Add additional chunks to an existing compatible vector store."""
    if not chunks:
        return

    try:
        vectorstore.add_documents(chunks)
    except Exception as exc:
        raise VectorStoreError(
            f"Failed to add documents to the vector store: {exc}"
        ) from exc


def get_retriever(
    vectorstore: Chroma,
    k: int | None = None,
):
    """Return the standard Chroma similarity retriever."""
    search_k = (
        config.RETRIEVER_K
        if k is None
        else k
    )

    if search_k <= 0:
        raise VectorStoreError(
            f"Retriever k must be positive, got {search_k}."
        )

    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": search_k},
    )


def get_distinct_sources(
    vectorstore: Chroma,
) -> list[str]:
    """Return distinct source values stored in document metadata."""
    try:
        result = vectorstore._collection.get(
            include=["metadatas"]
        )
    except Exception as exc:
        raise VectorStoreError(
            f"Failed to read collection metadata: {exc}"
        ) from exc

    sources: set[str] = set()

    for metadata in result.get("metadatas", []) or []:
        if not isinstance(metadata, dict):
            continue

        source = metadata.get("source")

        if source is not None:
            sources.add(str(source))

    return sorted(sources)


def similarity_search_with_relevance(
    vectorstore: Chroma,
    query: str,
    k: int | None = None,
) -> list[tuple[Document, float]]:
    """Return Chroma results with cosine-similarity relevance scores."""
    query = query.strip()

    if not query:
        raise VectorStoreError(
            "Cannot perform scored retrieval with an empty query."
        )

    search_k = (
        config.RETRIEVER_K
        if k is None
        else k
    )

    if search_k <= 0:
        raise VectorStoreError(
            f"Retriever k must be positive, got {search_k}."
        )

    try:
        results = vectorstore.similarity_search_with_score(
            query,
            k=search_k,
        )
    except Exception as exc:
        raise VectorStoreError(
            f"Scored retrieval failed: {exc}"
        ) from exc

    scored_results: list[tuple[Document, float]] = []

    for document, distance in results:
        relevance = 1.0 - float(distance)
        relevance = max(-1.0, min(1.0, relevance))
        scored_results.append(
            (document, relevance)
        )

    return scored_results


def _document_key(document: Document) -> str:
    """Create a stable key for hybrid-result deduplication."""
    document_id = document.metadata.get("document_id")
    chunk_id = document.metadata.get("chunk_id")

    if document_id is not None and chunk_id is not None:
        return f"{document_id}:{chunk_id}"

    source = str(
        document.metadata.get("source", "unknown")
    )

    digest = str(abs(hash(document.page_content)))
    return f"{source}:{digest}"


def _load_all_documents(
    vectorstore: Chroma,
) -> list[Document]:
    """Load indexed text and metadata from Chroma."""
    try:
        # Hybrid BM25 retrieval needs the complete corpus, which Chroma's retriever does not expose.
        result = vectorstore._collection.get(
            include=["documents", "metadatas"]
        )
    except Exception as exc:
        raise VectorStoreError(
            f"Failed to load documents for hybrid retrieval: {exc}"
        ) from exc

    documents = result.get("documents") or []
    metadatas = result.get("metadatas") or []

    if len(documents) != len(metadatas):
        raise VectorStoreError(
            "Chroma returned inconsistent documents and metadata counts."
        )

    indexed_documents: list[Document] = []

    for content, metadata in zip(
        documents,
        metadatas,
    ):
        if not content:
            continue

        indexed_documents.append(
            Document(
                page_content=str(content),
                metadata=metadata or {},
            )
        )

    if not indexed_documents:
        raise VectorStoreError(
            "The vector store contains no documents for hybrid retrieval."
        )

    return indexed_documents


def _tokenize_for_bm25(text: str) -> list[str]:
    """Tokenize text into lowercase lexical terms."""
    return re.findall(
        r"[a-zA-Z0-9_]+",
        text.lower(),
    )


def _reciprocal_rank_fusion(
    ranked_lists: list[list[str]],
    rrf_k: int = RRF_K,
) -> dict[str, float]:
    """Fuse ranked result lists without comparing incompatible scores."""
    if rrf_k <= 0:
        raise VectorStoreError(
            f"rrf_k must be positive, got {rrf_k}."
        )

    scores: dict[str, float] = {}

    for ranked_list in ranked_lists:
        for rank, key in enumerate(
            ranked_list,
            start=1,
        ):
            scores[key] = scores.get(key, 0.0) + (
                1.0 / (rrf_k + rank)
            )

    return scores


def _expand_neighbors(
    selected_documents: list[Document],
    all_documents: list[Document],
    window: int = NEIGHBOR_WINDOW,
) -> list[Document]:
    """Add nearby chunks from the same source document."""
    if window <= 0:
        return selected_documents

    indexed_by_document: dict[str, dict[int, Document]] = {}

    for document in all_documents:
        document_id = document.metadata.get("document_id")

        if document_id is None:
            continue

        raw_chunk_id = document.metadata.get("chunk_id")

        try:
            chunk_id = int(raw_chunk_id)
        except (TypeError, ValueError):
            continue

        indexed_by_document.setdefault(
            str(document_id),
            {},
        )[chunk_id] = document

    expanded: list[Document] = []
    seen: set[str] = set()

    for document in selected_documents:
        document_key = _document_key(document)

        if document_key not in seen:
            expanded.append(document)
            seen.add(document_key)

        document_id = document.metadata.get("document_id")
        raw_chunk_id = document.metadata.get("chunk_id")

        if document_id is None:
            continue

        try:
            chunk_id = int(raw_chunk_id)
        except (TypeError, ValueError):
            continue

        neighbors = indexed_by_document.get(
            str(document_id),
            {},
        )

        for offset in range(
            -window,
            window + 1,
        ):
            if offset == 0:
                continue

            neighbor = neighbors.get(
                chunk_id + offset
            )

            if neighbor is None:
                continue

            neighbor_key = _document_key(neighbor)

            if neighbor_key not in seen:
                expanded.append(neighbor)
                seen.add(neighbor_key)

    return expanded


def hybrid_search_with_relevance(
    vectorstore: Chroma,
    query: str,
    k: int | None = None,
    semantic_k: int = SEMANTIC_CANDIDATE_K,
    lexical_k: int = LEXICAL_CANDIDATE_K,
    neighbor_window: int = NEIGHBOR_WINDOW,
) -> list[tuple[Document, float, float]]:
    """
    Combine Chroma semantic retrieval and BM25 using RRF.

    Returns:
        (document, rrf_score, raw_semantic_score)
    """
    query = query.strip()

    if not query:
        raise VectorStoreError(
            "Cannot perform hybrid retrieval with an empty query."
        )

    final_k = (
        config.RETRIEVER_K
        if k is None
        else k
    )

    if final_k <= 0:
        raise VectorStoreError(
            f"Retriever k must be positive, got {final_k}."
        )

    if semantic_k <= 0 or lexical_k <= 0:
        raise VectorStoreError(
            "semantic_k and lexical_k must both be positive."
        )

    if neighbor_window < 0:
        raise VectorStoreError(
            "neighbor_window cannot be negative."
        )

    semantic_results = similarity_search_with_relevance(
        vectorstore,
        query,
        k=semantic_k,
    )

    all_documents = _load_all_documents(
        vectorstore
    )

    tokenized_documents = [
        _tokenize_for_bm25(
            document.page_content
        )
        for document in all_documents
    ]

    try:
        bm25 = BM25Okapi(
            tokenized_documents
        )
    except Exception as exc:
        raise VectorStoreError(
            f"Failed to build BM25 index: {exc}"
        ) from exc

    query_tokens = _tokenize_for_bm25(query)

    if not query_tokens:
        raise VectorStoreError(
            "Hybrid retrieval query contains no searchable terms."
        )

    try:
        lexical_scores = bm25.get_scores(
            query_tokens
        )
    except Exception as exc:
        raise VectorStoreError(
            f"BM25 retrieval failed: {exc}"
        ) from exc

    lexical_ranked = sorted(
        zip(
            lexical_scores,
            all_documents,
        ),
        key=lambda item: item[0],
        reverse=True,
    )[:lexical_k]

    semantic_ranked_keys: list[str] = []
    semantic_scores_by_key: dict[str, float] = {}
    documents_by_key: dict[str, Document] = {}

    for document, score in semantic_results:
        key = _document_key(document)
        semantic_ranked_keys.append(key)
        semantic_scores_by_key[key] = float(score)
        documents_by_key[key] = document

    lexical_ranked_keys: list[str] = []

    for _, document in lexical_ranked:
        key = _document_key(document)
        lexical_ranked_keys.append(key)
        documents_by_key[key] = document

    rrf_scores = _reciprocal_rank_fusion(
        [
            semantic_ranked_keys,
            lexical_ranked_keys,
        ]
    )

    ranked_candidates = sorted(
        rrf_scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    seed_documents: list[Document] = []

    for key, _ in ranked_candidates[:final_k]:
        document = documents_by_key.get(key)

        if document is not None:
            seed_documents.append(document)

    # Restore adjacent chunks after ranking because context can span chunk boundaries.
    expanded_documents = _expand_neighbors(
        seed_documents,
        all_documents,
        window=neighbor_window,
    )

    selected_documents: list[
        tuple[Document, float, float]
    ] = []

    selected_keys: set[str] = set()

    ranked_scores = dict(ranked_candidates)

    for document in expanded_documents:
        key = _document_key(document)

        if key in selected_keys:
            continue

        selected_keys.add(key)

        rrf_score = ranked_scores.get(key, 0.0)

        if key in semantic_scores_by_key:
            semantic_score = semantic_scores_by_key[key]
        else:
            semantic_score = 0.0

        selected_documents.append(
            (
                document,
                float(rrf_score),
                float(semantic_score),
            )
        )

    return selected_documents