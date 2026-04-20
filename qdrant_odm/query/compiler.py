from qdrant_client.http import models

from qdrant_odm.exceptions import QueryCompileError
from qdrant_odm.query.expressions import ComparisonExpr, Expr, LogicalExpr, NotExpr


class FilterCompiler:
    @classmethod
    def compile(cls, expr: Expr | None) -> models.Filter | None:
        if expr is None:
            return None
        return cls._compile_expr(expr)

    @classmethod
    def _compile_expr(cls, expr: Expr) -> models.Filter:
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
