class QdrantODMError(Exception):
    """Base exception for qdrant-odm."""


class ModelDefinitionError(QdrantODMError):
    """Raised when a model declaration is invalid."""


class SchemaConflictError(QdrantODMError):
    """Raised when live schema conflicts with model schema."""


class SerializationError(QdrantODMError):
    """Raised when model serialization or deserialization fails."""


class QueryCompileError(QdrantODMError):
    """Raised when filter/query DSL cannot be compiled."""


class RepositoryError(QdrantODMError):
    """Raised when repository operations fail."""


class NotFoundError(QdrantODMError):
    """Raised when a requested resource is not found."""
