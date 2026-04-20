from dataclasses import dataclass
from typing import Any, Sequence


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
