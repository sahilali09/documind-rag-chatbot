class RAGChatbotError(Exception):
    """Base exception for all application-specific errors."""


class ConfigurationError(RAGChatbotError):
    """Raised when application configuration is invalid."""


class DocumentLoadError(RAGChatbotError):
    """Raised when a document or URL cannot be loaded or validated."""


class VectorStoreError(RAGChatbotError):
    """Raised when vector-store operations or compatibility checks fail."""


class ProviderConfigurationError(RAGChatbotError):
    """Raised when an LLM or embedding provider is misconfigured."""


class RetrievalError(RAGChatbotError):
    """Raised when the retrieval operation itself fails."""