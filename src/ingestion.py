from __future__ import annotations

import hashlib
import re
from pathlib import Path
from urllib.parse import urlparse

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config
from src.exceptions import DocumentLoadError

SUPPORTED_FILE_EXTENSIONS = {".pdf", ".txt", ".md"}

_ALLOWED_URL_SCHEMES = {"http", "https"}
_URL_FETCH_TIMEOUT_SECONDS = 15


def _clean_text(text: str) -> str:
    """Normalize whitespace produced by document/web extraction."""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _document_id(identifier: str, content_bytes: bytes) -> str:
    """Create a stable ID from a readable identifier and content hash."""
    digest = hashlib.sha256(content_bytes).hexdigest()[:10]
    stem = re.sub(
        r"[^a-zA-Z0-9_-]",
        "_",
        Path(identifier).stem or identifier,
    )[:40].strip("_")
    return f"{stem or 'doc'}_{digest}"


# Validation

def validate_file(file_path: str | Path) -> None:
    """Validate a single local file before loading."""
    file_path = Path(file_path)

    if not file_path.exists():
        raise DocumentLoadError(f"File not found: {file_path}")

    if not file_path.is_file():
        raise DocumentLoadError(f"Path is not a file: {file_path}")

    suffix = file_path.suffix.lower()

    if suffix not in SUPPORTED_FILE_EXTENSIONS:
        raise DocumentLoadError(
            f"Unsupported file type '{suffix}' for {file_path.name}. "
            f"Supported types: {sorted(SUPPORTED_FILE_EXTENSIONS)}"
        )

    try:
        size_bytes = file_path.stat().st_size
    except OSError as exc:
        raise DocumentLoadError(
            f"Unable to inspect {file_path.name}: {exc}"
        ) from exc

    if size_bytes == 0:
        raise DocumentLoadError(f"{file_path.name} is empty.")

    max_bytes = config.MAX_UPLOAD_SIZE_MB * 1024 * 1024

    if size_bytes > max_bytes:
        raise DocumentLoadError(
            f"{file_path.name} is {size_bytes / (1024 * 1024):.1f} MB, "
            f"which exceeds the {config.MAX_UPLOAD_SIZE_MB:.0f} MB limit "
            "(set via MAX_UPLOAD_SIZE_MB)."
        )


def validate_file_count(file_paths: list[str | Path]) -> None:
    """Validate the number of files in one ingestion batch."""
    if len(file_paths) > config.MAX_UPLOAD_FILES:
        raise DocumentLoadError(
            f"{len(file_paths)} files were provided, which exceeds the "
            f"{config.MAX_UPLOAD_FILES} file limit "
            "(set via MAX_UPLOAD_FILES)."
        )


def validate_url(url: str) -> None:
    """Validate that a URL is an absolute HTTP(S) URL."""
    if not isinstance(url, str) or not url.strip():
        raise DocumentLoadError("URL cannot be empty.")

    url = url.strip()
    parsed = urlparse(url)

    if parsed.scheme.lower() not in _ALLOWED_URL_SCHEMES:
        raise DocumentLoadError(
            f"'{url}' is not a valid http(s) URL "
            f"(scheme was '{parsed.scheme or 'missing'}')."
        )

    if not parsed.netloc:
        raise DocumentLoadError(
            f"'{url}' doesn't look like a valid URL (no host)."
        )

    if parsed.username or parsed.password:
        raise DocumentLoadError(
            "URLs containing embedded credentials are not allowed."
        )


# Loading

def load_file(file_path: str | Path) -> list[Document]:
    """
    Load a single local PDF, TXT, or Markdown file.

    PDFs are loaded page-by-page so page metadata remains available
    for source display.
    """
    file_path = Path(file_path)
    validate_file(file_path)

    suffix = file_path.suffix.lower()

    try:
        content_bytes = file_path.read_bytes()
    except OSError as exc:
        raise DocumentLoadError(
            f"Failed to read {file_path.name}: {exc}"
        ) from exc

    document_id = _document_id(file_path.name, content_bytes)

    try:
        if suffix == ".pdf":
            docs = _load_pdf_with_pymupdf(file_path)
            source_type = "pdf"
        else:
            from langchain_community.document_loaders import TextLoader

            docs = TextLoader(
                str(file_path),
                encoding="utf-8",
            ).load()
            source_type = "text"

    except DocumentLoadError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise DocumentLoadError(
            f"Failed to read {file_path.name}: {exc}"
        ) from exc

    cleaned_docs: list[Document] = []

    for doc in docs:
        content = _clean_text(doc.page_content)

        if not content:
            continue

        doc.page_content = content
        doc.metadata["source"] = file_path.name
        doc.metadata["source_type"] = source_type
        doc.metadata["document_id"] = document_id

        cleaned_docs.append(doc)

    if not cleaned_docs:
        raise DocumentLoadError(
            f"No extractable text found in {file_path.name}."
        )

    return cleaned_docs


def _load_pdf_with_pymupdf(file_path: Path) -> list[Document]:
    """Load a PDF page-by-page using PyMuPDF."""
    try:
        import pymupdf
    except ImportError as exc:
        raise DocumentLoadError(
            "PyMuPDF is required for PDF ingestion."
        ) from exc

    docs: list[Document] = []

    try:
        with pymupdf.open(str(file_path)) as pdf:
            for page_number, page in enumerate(pdf):
                docs.append(
                    Document(
                        page_content=page.get_text(),
                        metadata={
                            "source": file_path.name,
                            "page": page_number,
                        },
                    )
                )
    except Exception as exc:  # noqa: BLE001
        raise DocumentLoadError(
            f"Failed to extract PDF {file_path.name}: {exc}"
        ) from exc

    return docs


def load_files(
    file_paths: list[str | Path],
) -> list[Document]:
    """Load and concatenate multiple local files."""
    validate_file_count(file_paths)

    all_docs: list[Document] = []

    for file_path in file_paths:
        all_docs.extend(load_file(file_path))

    return all_docs


def load_urls(urls: list[str]) -> list[Document]:
    """Load and clean one or more web pages."""
    urls = [url.strip() for url in urls if url and url.strip()]

    if not urls:
        return []

    for url in urls:
        validate_url(url)

    try:
        from langchain_community.document_loaders import WebBaseLoader

        loader = WebBaseLoader(urls)
        loader.requests_kwargs = {
            "timeout": _URL_FETCH_TIMEOUT_SECONDS,
        }

        docs = loader.load()

    except Exception as exc:  # noqa: BLE001
        raise DocumentLoadError(
            f"Failed to fetch one or more URLs: {exc}"
        ) from exc

    cleaned_docs: list[Document] = []

    for doc in docs:
        content = _clean_text(doc.page_content)

        if not content:
            continue

        source = (
            doc.metadata.get("source")
            or doc.metadata.get("title")
            or "web"
        )

        doc.page_content = content
        doc.metadata["source"] = str(source)
        doc.metadata["source_type"] = "web"
        doc.metadata["document_id"] = _document_id(
            str(source),
            content.encode("utf-8"),
        )

        cleaned_docs.append(doc)

    return cleaned_docs


# Chunking

def split_documents(
    documents: list[Document],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Document]:
    """Split documents into overlapping retrieval chunks."""
    effective_chunk_size = (
        config.CHUNK_SIZE
        if chunk_size is None
        else chunk_size
    )

    effective_chunk_overlap = (
        config.CHUNK_OVERLAP
        if chunk_overlap is None
        else chunk_overlap
    )

    if effective_chunk_size <= 0:
        raise DocumentLoadError(
            f"chunk_size must be positive, got {effective_chunk_size}."
        )

    if effective_chunk_overlap < 0:
        raise DocumentLoadError(
            f"chunk_overlap cannot be negative, got {effective_chunk_overlap}."
        )

    if effective_chunk_overlap >= effective_chunk_size:
        raise DocumentLoadError(
            "chunk_overlap must be smaller than chunk_size."
        )

    if not documents:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=effective_chunk_size,
        chunk_overlap=effective_chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(documents)

    chunk_counters: dict[str, int] = {}

    for chunk in chunks:
        document_id = chunk.metadata.get(
            "document_id",
            "unknown",
        )

        chunk_index = chunk_counters.get(document_id, 0)

        chunk.metadata["chunk_id"] = chunk_index
        chunk_counters[document_id] = chunk_index + 1

        source = chunk.metadata.get("source", "unknown")
        page = chunk.metadata.get("page")

        if isinstance(page, int):
            chunk.metadata["display_source"] = (
                f"{source} (page {page + 1})"
            )
        else:
            chunk.metadata["display_source"] = str(source)

    return chunks


# Full ingestion pipeline

def ingest(
    file_paths: list[str | Path] | None = None,
    urls: list[str] | None = None,
) -> list[Document]:
    """Validate, load, clean, and chunk all supplied sources."""
    documents: list[Document] = []

    if file_paths:
        documents.extend(
            load_files(list(file_paths))
        )

    if urls:
        documents.extend(
            load_urls(list(urls))
        )

    if not documents:
        return []

    return split_documents(documents)