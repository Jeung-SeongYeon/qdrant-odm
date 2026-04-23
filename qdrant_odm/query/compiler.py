from datetime import datetime, date, timezone
from typing import TYPE_CHECKING

from qdrant_client.http import models

from qdrant_odm.exceptions import QueryCompileError
from qdrant_odm.query.expressions import ComparisonExpr, Expr, LogicalExpr, NotExpr

if TYPE_CHECKING:
    from qdrant_odm.model.base import QdrantModel


class FilterCompiler:
    """
    Compile query expression objects into Qdrant filter models.

    This compiler transforms the internal expression tree used by the ODM query DSL
    into `qdrant_client.http.models.Filter` instances that can be sent directly to Qdrant.

    Supported expression categories include:
    - logical expressions (`and`, `or`)
    - negation (`not`)
    - comparison expressions (`eq`, `ne`, `gt`, `gte`, `lt`, `lte`, `in`, `not_in`)
    - null checks (`is_null`, `is_not_null`)
    """

    @classmethod
    def compile(cls, expr: Expr | None, *, model: type["QdrantModel"] | None = None) -> models.Filter | None:
        """
        Compile an optional expression into a Qdrant filter.

        Args:
            expr:
                The root expression to compile. If None, no filter is produced.

        Returns:
            A Qdrant filter object, or None when no expression is provided.
        """
        if expr is None:
            return None
        return cls._compile_expr(expr, model=model)

    @classmethod
    def _compile_expr(cls, expr: Expr, *, model: type["QdrantModel"] | None = None) -> models.Filter:
        """
        Compile a single expression node recursively.

        Args:
            expr:
                The expression node to compile.

        Returns:
            A Qdrant filter representing the given expression subtree.

        Raises:
            QueryCompileError:
                If the expression type or operator is not supported.
        """
        if isinstance(expr, LogicalExpr):
            compiled_values = [cls._compile_expr(value, model=model) for value in expr.values]
            if expr.operator == "and":
                return models.Filter(must=compiled_values)
            if expr.operator == "or":
                return models.Filter(should=compiled_values)
            raise QueryCompileError(f"Unsupported logical operator: {expr.operator}")

        if isinstance(expr, NotExpr):
            compiled_value = cls._compile_expr(expr.value, model=model)
            return models.Filter(must_not=[compiled_value])

        if isinstance(expr, ComparisonExpr):
            return cls._compile_comparison(expr, model=model)

        raise QueryCompileError(f"Unsupported expression type: {type(expr)!r}")

    @classmethod
    def _compile_comparison(cls, expr: ComparisonExpr, *, model: type["QdrantModel"] | None = None) -> models.Filter:
        """
        Compile a comparison expression into a Qdrant filter.

        Args:
            expr:
                A comparison expression containing the field name, operator, and value.

        Returns:
            A Qdrant filter representing the comparison.

        Raises:
            QueryCompileError:
                If the comparison operator is not supported.
        """
        field_name = expr.field_name
        value = expr.value

        if expr.operator == "eq":
            condition = models.FieldCondition(key=field_name, match=models.MatchValue(value=value))
            return models.Filter(must=[condition])

        if expr.operator == "ne":
            condition = models.FieldCondition(key=field_name, match=models.MatchValue(value=value))
            return models.Filter(must_not=[condition])

        if expr.operator in {"gt", "gte", "lt", "lte"}:
            kwargs = {expr.operator: cls._normalize_range_value(field_name, value, model=model)}

            if cls._is_datetime_field(field_name, model=model):
                condition = models.FieldCondition(
                    key=field_name,
                    range=models.DatetimeRange(**kwargs),
                )
            else:
                condition = models.FieldCondition(
                    key=field_name,
                    range=models.Range(**kwargs),
                )
            return models.Filter(must=[condition])

        if expr.operator == "in":
            condition = models.FieldCondition(key=field_name, match=models.MatchAny(any=value))
            return models.Filter(must=[condition])

        if expr.operator == "not_in":
            condition = models.FieldCondition(key=field_name, match=models.MatchAny(any=value))
            return models.Filter(must_not=[condition])

        if expr.operator == "is_null":
            return models.Filter(
                must=[models.IsNullCondition(is_null=models.PayloadField(key=field_name))]
            )

        if expr.operator == "is_not_null":
            return models.Filter(
                must_not=[models.IsNullCondition(is_null=models.PayloadField(key=field_name))]
            )

        raise QueryCompileError(f"Unsupported comparison operator: {expr.operator}")

    @classmethod
    def _is_datetime_field(
        cls,
        field_name: str,
        *,
        model: type["QdrantModel"] | None = None,
    ) -> bool:
        if model is None:
            return False

        payload_info = model.schema_definition().payload_fields.get(field_name)
        if payload_info is None:
            return False

        return payload_info.index == "datetime"

    @classmethod
    def _normalize_range_value(
        cls,
        field_name: str,
        value,
        *,
        model: type["QdrantModel"] | None = None,
    ):
        if cls._is_datetime_field(field_name, model=model):
            return cls._normalize_datetime_value(value)
        return value

    @classmethod
    def _normalize_datetime_value(cls, value):
        if isinstance(value, datetime):
            if value.tzinfo is None:
                value = value.replace(tzinfo=timezone.utc)
            return value.isoformat()

        if isinstance(value, date):
            return datetime(value.year, value.month, value.day, tzinfo=timezone.utc).isoformat()

        if isinstance(value, str):
            return value

        raise QueryCompileError(
            f"Datetime field comparisons require datetime/date/RFC3339 string values, got {type(value)!r}"
        )