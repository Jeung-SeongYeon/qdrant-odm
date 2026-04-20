from dataclasses import dataclass
from typing import Any, Sequence


class Expr:
    """
    Base class for all query expression nodes.

    Expression objects form a small abstract syntax tree used by the ODM query DSL.
    Instances can be combined with:
    - `&` for logical AND
    - `|` for logical OR
    - `~` for logical NOT
    """

    def __and__(self, other: "Expr") -> "LogicalExpr":
        """
        Combine this expression with another expression using logical AND.

        Args:
            other:
                The expression to combine with this expression.

        Returns:
            A `LogicalExpr` representing an AND operation.
        """
        return LogicalExpr(operator="and", values=[self, other])

    def __or__(self, other: "Expr") -> "LogicalExpr":
        """
        Combine this expression with another expression using logical OR.

        Args:
            other:
                The expression to combine with this expression.

        Returns:
            A `LogicalExpr` representing an OR operation.
        """
        return LogicalExpr(operator="or", values=[self, other])

    def __invert__(self) -> "NotExpr":
        """
        Negate this expression using logical NOT.

        Returns:
            A `NotExpr` wrapping the current expression.
        """
        return NotExpr(value=self)


@dataclass(slots=True)
class FieldExpr:
    """
    Field reference used as the entry point for comparison expressions.

    A `FieldExpr` is typically created through model class attribute access, such as:
        Document.category
        Document.page

    It can then be used to build comparison expressions with operators like:
        ==, !=, >, >=, <, <=, in_, not_in, is_null, is_not_null
    """

    field_name: str

    def __eq__(self, other: Any) -> "ComparisonExpr":  # type: ignore[override]
        """
        Create an equality comparison expression.

        Args:
            other:
                The comparison value.

        Returns:
            A comparison expression with the `eq` operator.
        """
        return ComparisonExpr(operator="eq", field_name=self.field_name, value=other)

    def __ne__(self, other: Any) -> "ComparisonExpr":  # type: ignore[override]
        """
        Create an inequality comparison expression.

        Args:
            other:
                The comparison value.

        Returns:
            A comparison expression with the `ne` operator.
        """
        return ComparisonExpr(operator="ne", field_name=self.field_name, value=other)

    def __gt__(self, other: Any) -> "ComparisonExpr":
        """
        Create a greater-than comparison expression.

        Args:
            other:
                The comparison value.

        Returns:
            A comparison expression with the `gt` operator.
        """
        return ComparisonExpr(operator="gt", field_name=self.field_name, value=other)

    def __ge__(self, other: Any) -> "ComparisonExpr":
        """
        Create a greater-than-or-equal comparison expression.

        Args:
            other:
                The comparison value.

        Returns:
            A comparison expression with the `gte` operator.
        """
        return ComparisonExpr(operator="gte", field_name=self.field_name, value=other)

    def __lt__(self, other: Any) -> "ComparisonExpr":
        """
        Create a less-than comparison expression.

        Args:
            other:
                The comparison value.

        Returns:
            A comparison expression with the `lt` operator.
        """
        return ComparisonExpr(operator="lt", field_name=self.field_name, value=other)

    def __le__(self, other: Any) -> "ComparisonExpr":
        """
        Create a less-than-or-equal comparison expression.

        Args:
            other:
                The comparison value.

        Returns:
            A comparison expression with the `lte` operator.
        """
        return ComparisonExpr(operator="lte", field_name=self.field_name, value=other)

    def in_(self, values: Sequence[Any]) -> "ComparisonExpr":
        """
        Create an inclusion comparison expression.

        Args:
            values:
                The allowed values for membership testing.

        Returns:
            A comparison expression with the `in` operator.
        """
        return ComparisonExpr(operator="in", field_name=self.field_name, value=list(values))

    def not_in(self, values: Sequence[Any]) -> "ComparisonExpr":
        """
        Create a negated inclusion comparison expression.

        Args:
            values:
                The disallowed values for membership testing.

        Returns:
            A comparison expression with the `not_in` operator.
        """
        return ComparisonExpr(operator="not_in", field_name=self.field_name, value=list(values))

    def is_null(self) -> "ComparisonExpr":
        """
        Create a null-check comparison expression.

        Returns:
            A comparison expression with the `is_null` operator.
        """
        return ComparisonExpr(operator="is_null", field_name=self.field_name, value=None)

    def is_not_null(self) -> "ComparisonExpr":
        """
        Create a non-null-check comparison expression.

        Returns:
            A comparison expression with the `is_not_null` operator.
        """
        return ComparisonExpr(operator="is_not_null", field_name=self.field_name, value=None)


@dataclass(slots=True)
class ComparisonExpr(Expr):
    """
    Expression node representing a field comparison.

    Attributes:
        operator:
            The comparison operator name.
        field_name:
            The payload field being compared.
        value:
            The comparison value.
    """

    operator: str
    field_name: str
    value: Any


@dataclass(slots=True)
class LogicalExpr(Expr):
    """
    Expression node representing a logical combination of expressions.

    Attributes:
        operator:
            The logical operator name, such as `and` or `or`.
        values:
            Child expressions combined by the operator.
    """

    operator: str
    values: list[Expr]


@dataclass(slots=True)
class NotExpr(Expr):
    """
    Expression node representing logical negation.

    Attributes:
        value:
            The expression being negated.
    """

    value: Expr


FilterExpression = Expr