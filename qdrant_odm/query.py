from dataclasses import dataclass
from typing import Any, Sequence

from pydantic import BaseModel, Field
from qdrant_client.http import models

from qdrant_odm.exceptions import QueryCompileError


class Expr:
    def __and__(self, other: "Expr") -> "LogicalExpr":
        return LogicalExpr(operator="and", values=[self, other])

    def __or__(self, other: "Expr") -> "LogicalExpr":
        return LogicalExpr(operator="or", values=[self, other])

    def __invert__(self) -> "NotExpr":
        return NotExpr(value=self)


@dataclass(slots=True)
class FieldExpr:
    field_name: str

    def __eq__(self, other: Any) -> "ComparisonExpr":  # type: ignore[override]
        return ComparisonExpr(operator="eq", field_name=self.field_name, value=other)

    def __ne__(self, other: Any) -> "ComparisonExpr":  # type: ignore[override]
        return ComparisonExpr(operator="ne", field_name=self.field_name, value=other)

    def __gt__(self, other: Any) -> "ComparisonExpr":
        return ComparisonExpr(operator="gt", field_name=self.field_name, value=other)

    def __ge__(self, other: Any) -> "ComparisonExpr":
        return ComparisonExpr(operator="gte", field_name=self.field_name, value=other)

    def __lt__(self, other: Any) -> "ComparisonExpr":
        return ComparisonExpr(operator="lt", field_name=self.field_name, value=other)

    def __le__(self, other: Any) -> "ComparisonExpr":
        return ComparisonExpr(operator="lte", field_name=self.field_name, value=other)

    def in_(self, values: Sequence[Any]) -> "ComparisonExpr":
        return ComparisonExpr(operator="in", field_name=self.field_name, value=list(values))

    def not_in(self, values: Sequence[Any]) -> "ComparisonExpr":
        return ComparisonExpr(operator="not_in", field_name=self.field_name, value=list(values))

    def is_null(self) -> "ComparisonExpr":
        return ComparisonExpr(operator="is_null", field_name=self.field_name, value=None)

    def is_not_null(self) -> "ComparisonExpr":
        return ComparisonExpr(operator="is_not_null", field_name=self.field_name, value=None)


@dataclass(slots=True)
class ComparisonExpr(Expr):
    operator: str
    field_name: str
    value: Any


@dataclass(slots=True)
class LogicalExpr(Expr):
    operator: str
    values: list[Expr]


@dataclass(slots=True)
class NotExpr(Expr):
    value: Expr


FilterExpression = Expr


class SparseVectorInput(BaseModel):
    indices: list[int] = Field(default_factory=list)
    values: list[float] = Field(default_factory=list)

    def to_qdrant(self) -> models.SparseVector:
        return models.SparseVector(indices=self.indices, values=self.values)


class SearchQuery(BaseModel):
    using: str
    vector: list[float] | SparseVectorInput
    filter: FilterExpression | None = None
    limit: int = 10
    offset: int | None = None
    with_payload: bool = True
    with_vectors: bool = False
    score_threshold: float | None = None

    model_config = {"arbitrary_types_allowed": True}


class HybridSearchQuery(BaseModel):
    dense_using: str
    dense_vector: list[float]
    sparse_using: str
    sparse_vector: SparseVectorInput
    filter: FilterExpression | None = None
    limit: int = 10
    with_payload: bool = True
    with_vectors: bool = False
    score_threshold: float | None = None
    fusion_k: int = 60

    model_config = {"arbitrary_types_allowed": True}


class FilterCompiler:
    @classmethod
    def compile(cls, expr: FilterExpression | None) -> models.Filter | None:
        if expr is None:
            return None
        return cls._compile_expr(expr)

    @classmethod
    def _compile_expr(cls, expr: FilterExpression) -> models.Filter:
        if isinstance(expr, LogicalExpr):
            compiled_values = [cls._compile_expr(value) for value in expr.values]
            if expr.operator == "and":
                return models.Filter(must=compiled_values)
            if expr.operator == "or":
                return models.Filter(should=compiled_values)
            raise QueryCompileError(f"Unsupported logical operator: {expr.operator}")

        if isinstance(expr, NotExpr):
            compiled_value = cls._compile_expr(expr.value)
            return models.Filter(must_not=[compiled_value])

        if isinstance(expr, ComparisonExpr):
            return cls._compile_comparison(expr)

        raise QueryCompileError(f"Unsupported expression type: {type(expr)!r}")

    @classmethod
    def _compile_comparison(cls, expr: ComparisonExpr) -> models.Filter:
        field_name = expr.field_name
        value = expr.value

        if expr.operator == "eq":
            condition = models.FieldCondition(key=field_name, match=models.MatchValue(value=value))
            return models.Filter(must=[condition])

        if expr.operator == "ne":
            condition = models.FieldCondition(key=field_name, match=models.MatchValue(value=value))
            return models.Filter(must_not=[condition])

        if expr.operator in {"gt", "gte", "lt", "lte"}:
            kwargs = {expr.operator: value}
            condition = models.FieldCondition(key=field_name, range=models.Range(**kwargs))
            return models.Filter(must=[condition])

        if expr.operator == "in":
            condition = models.FieldCondition(key=field_name, match=models.MatchAny(any=value))
            return models.Filter(must=[condition])

        if expr.operator == "not_in":
            condition = models.FieldCondition(key=field_name, match=models.MatchAny(any=value))
            return models.Filter(must_not=[condition])

        if expr.operator == "is_null":
            return models.Filter(must=[models.IsNullCondition(is_null=models.PayloadField(key=field_name))])

        if expr.operator == "is_not_null":
            return models.Filter(
                must_not=[models.IsNullCondition(is_null=models.PayloadField(key=field_name))]
            )

        raise QueryCompileError(f"Unsupported comparison operator: {expr.operator}")
