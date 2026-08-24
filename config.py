import os

from dotenv import load_dotenv

from src.exceptions import ConfigurationError

load_dotenv()

VALID_LLM_PROVIDERS = {"groq", "openai", "ollama"}
VALID_EMBEDDING_PROVIDERS = {"huggingface", "openai", "ollama"}

ALLOWED_URL_SCHEMES = {"http", "https"}


def _get_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)

    if value is None:
        return default

    normalized = value.strip().lower()

    if normalized in {"1", "true", "yes", "on"}:
        return True

    if normalized in {"0", "false", "no", "off"}:
        return False

    raise ConfigurationError(
        f"{name} must be a boolean value "
        f"(true/false, yes/no, 1/0, on/off); got '{value}'"
    )


def _get_int(name: str, default: int) -> int:
    value = os.getenv(name)

    if value is None or value.strip() == "":
        return default

    try:
        return int(value)
    except ValueError as exc:
        raise ConfigurationError(
            f"{name} must be an integer; got '{value}'"
        ) from exc


def _get_float(name: str, default: float) -> float:
    value = os.getenv(name)

    if value is None or value.strip() == "":
        return default

    try:
        return float(value)
    except ValueError as exc:
        raise ConfigurationError(
            f"{name} must be a number; got '{value}'"
        ) from exc


def _get_optional_float(name: str) -> float | None:
    value = os.getenv(name)

    if value is None or value.strip() == "":
        return None

    try:
        return float(value)
    except ValueError as exc:
        raise ConfigurationError(
            f"{name} must be a number or blank; got '{value}'"
        ) from exc


LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").strip().lower()

GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-120b",
).strip()

OPENAI_MODEL = os.getenv(
    "OPENAI_MODEL",
    "gpt-5.6-terra",
).strip()

OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "llama3.1",
).strip()

OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://localhost:11434",
).strip()

LLM_TEMPERATURE = _get_float("LLM_TEMPERATURE", 0.2)

EMBEDDING_PROVIDER = os.getenv(
    "EMBEDDING_PROVIDER",
    "huggingface",
).strip().lower()

HUGGINGFACE_EMBEDDING_MODEL = os.getenv(
    "HUGGINGFACE_EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
).strip()

OPENAI_EMBEDDING_MODEL = os.getenv(
    "OPENAI_EMBEDDING_MODEL",
    "text-embedding-3-small",
).strip()

OLLAMA_EMBEDDING_MODEL = os.getenv(
    "OLLAMA_EMBEDDING_MODEL",
    "nomic-embed-text",
).strip()

CHUNK_SIZE = _get_int("CHUNK_SIZE", 1000)
CHUNK_OVERLAP = _get_int("CHUNK_OVERLAP", 150)

MAX_UPLOAD_SIZE_MB = _get_float("MAX_UPLOAD_SIZE_MB", 20.0)
MAX_UPLOAD_FILES = _get_int("MAX_UPLOAD_FILES", 10)

CHROMA_PERSIST_DIR = os.getenv(
    "CHROMA_PERSIST_DIR",
    "./chroma_db",
).strip()

CHROMA_COLLECTION_NAME = os.getenv(
    "CHROMA_COLLECTION_NAME",
    "rag_chatbot",
).strip()

INDEX_VERSION = os.getenv(
    "INDEX_VERSION",
    "1",
).strip()

RETRIEVER_K = _get_int("RETRIEVER_K", 4)

if os.getenv("RETRIEVAL_SCORE_THRESHOLD") is None:
    RETRIEVAL_SCORE_THRESHOLD = 0.2
else:
    RETRIEVAL_SCORE_THRESHOLD = _get_optional_float(
        "RETRIEVAL_SCORE_THRESHOLD"
    )

MAX_HISTORY_TURNS = _get_int("MAX_HISTORY_TURNS", 6)

ENABLE_RETRIEVAL_DIAGNOSTICS = _get_bool(
    "ENABLE_RETRIEVAL_DIAGNOSTICS",
    False,
)

SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    "You are a careful assistant that answers questions using ONLY the "
    "provided context from the user's documents. "
    "If the answer is not contained in the context, say you don't have "
    "enough information rather than guessing. "
    "Cite sources inline using the bracketed numbers shown in the context, "
    "e.g. [1]. Keep answers concise and well-structured.",
).strip()


def validate() -> None:
    errors: list[str] = []

    if not LLM_PROVIDER:
        errors.append("LLM_PROVIDER must not be empty")
    elif LLM_PROVIDER not in VALID_LLM_PROVIDERS:
        errors.append(
            f"LLM_PROVIDER must be one of "
            f"{sorted(VALID_LLM_PROVIDERS)} "
            f"(got '{LLM_PROVIDER}')"
        )

    if not EMBEDDING_PROVIDER:
        errors.append("EMBEDDING_PROVIDER must not be empty")
    elif EMBEDDING_PROVIDER not in VALID_EMBEDDING_PROVIDERS:
        errors.append(
            f"EMBEDDING_PROVIDER must be one of "
            f"{sorted(VALID_EMBEDDING_PROVIDERS)} "
            f"(got '{EMBEDDING_PROVIDER}')"
        )

    if not GROQ_MODEL:
        errors.append("GROQ_MODEL must not be empty")

    if not OPENAI_MODEL:
        errors.append("OPENAI_MODEL must not be empty")

    if not OLLAMA_MODEL:
        errors.append("OLLAMA_MODEL must not be empty")

    if not OLLAMA_BASE_URL:
        errors.append("OLLAMA_BASE_URL must not be empty")

    if not HUGGINGFACE_EMBEDDING_MODEL:
        errors.append("HUGGINGFACE_EMBEDDING_MODEL must not be empty")

    if not OPENAI_EMBEDDING_MODEL:
        errors.append("OPENAI_EMBEDDING_MODEL must not be empty")

    if not OLLAMA_EMBEDDING_MODEL:
        errors.append("OLLAMA_EMBEDDING_MODEL must not be empty")

    if CHUNK_SIZE <= 0:
        errors.append(
            f"CHUNK_SIZE must be > 0 (got {CHUNK_SIZE})"
        )

    if CHUNK_OVERLAP < 0:
        errors.append(
            f"CHUNK_OVERLAP must be >= 0 (got {CHUNK_OVERLAP})"
        )

    if CHUNK_OVERLAP >= CHUNK_SIZE:
        errors.append(
            f"CHUNK_OVERLAP ({CHUNK_OVERLAP}) must be smaller than "
            f"CHUNK_SIZE ({CHUNK_SIZE})"
        )

    if RETRIEVER_K <= 0:
        errors.append(
            f"RETRIEVER_K must be > 0 (got {RETRIEVER_K})"
        )

    if MAX_HISTORY_TURNS < 0:
        errors.append(
            f"MAX_HISTORY_TURNS must be >= 0 (got {MAX_HISTORY_TURNS})"
        )

    if LLM_TEMPERATURE < 0:
        errors.append(
            f"LLM_TEMPERATURE must be >= 0 (got {LLM_TEMPERATURE})"
        )

    if MAX_UPLOAD_SIZE_MB <= 0:
        errors.append(
            f"MAX_UPLOAD_SIZE_MB must be > 0 "
            f"(got {MAX_UPLOAD_SIZE_MB})"
        )

    if MAX_UPLOAD_FILES <= 0:
        errors.append(
            f"MAX_UPLOAD_FILES must be > 0 "
            f"(got {MAX_UPLOAD_FILES})"
        )

    if RETRIEVAL_SCORE_THRESHOLD is not None and not (
        -1.0 <= RETRIEVAL_SCORE_THRESHOLD <= 1.0
    ):
        errors.append(
            "RETRIEVAL_SCORE_THRESHOLD must be between -1 and 1, "
            f"or blank to disable "
            f"(got {RETRIEVAL_SCORE_THRESHOLD})"
        )

    if not CHROMA_PERSIST_DIR:
        errors.append("CHROMA_PERSIST_DIR must not be empty")

    if not CHROMA_COLLECTION_NAME:
        errors.append("CHROMA_COLLECTION_NAME must not be empty")

    if not INDEX_VERSION:
        errors.append("INDEX_VERSION must not be empty")

    if not SYSTEM_PROMPT:
        errors.append("SYSTEM_PROMPT must not be empty")

    if errors:
        raise ConfigurationError(
            "Invalid configuration in config.py / .env:\n  - "
            + "\n  - ".join(errors)
        )


validate()