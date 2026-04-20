class QdrantODMError(Exception):
    """
    Base exception for all qdrant-odm errors.

    All package-specific exceptions inherit from this type so callers can catch
    ODM-related failures in a single place if needed.
    """


class ModelDefinitionError(QdrantODMError):
    """
    Raised when a model declaration is invalid.

    Typical examples include:
    - missing `__collection__`,
    - missing id field,
    - duplicated vector names,
    - otherwise invalid model metadata configuration.
    """


class SchemaConflictError(QdrantODMError):
    """
    Raised when the live Qdrant schema conflicts with the declared model schema.

    This usually indicates a mismatch that cannot be reconciled automatically,
    such as incompatible vector definitions or payload index type mismatches.
    """


class SerializationError(QdrantODMError):
    """
    Raised when model serialization or deserialization fails.

    This covers failures while converting between ODM model instances and
    Qdrant payload/point representations.
    """


class QueryCompileError(QdrantODMError):
    """
    Raised when the filter or query DSL cannot be compiled.

    This typically indicates an unsupported expression type or operator in the
    internal query expression tree.
    """


class RepositoryError(QdrantODMError):
    """
    Raised when a repository operation fails.

    This exception is intended for higher-level repository failures that are
    not better represented by more specific ODM exception types.
    """


class NotFoundError(QdrantODMError):
    """
    Raised when a requested ODM resource cannot be found.

    This can be used for lookup operations where the caller expects the target
    object to exist.
    """