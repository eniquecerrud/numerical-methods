"""Exceptions raised by numerica."""

from __future__ import annotations

from typing import Any


class NumericaError(Exception):
    """Base class for errors raised by numerica."""


class ConvergenceError(NumericaError):
    """An iterative method did not meet the requested tolerance.

    Attributes
    ----------
    iterations :
        Number of iterations that were performed.
    residual :
        Residual of the last accepted estimate. The definition depends on the
        method and is documented on that method.
    tolerance :
        Tolerance the caller requested.
    value :
        Best estimate available when the iteration stopped. This may be a
        float or a NumPy array.
    """

    def __init__(
        self,
        message: str,
        *,
        iterations: int,
        residual: float,
        tolerance: float,
        value: Any = None,
    ) -> None:
        super().__init__(message)
        self.iterations = iterations
        self.residual = residual
        self.tolerance = tolerance
        self.value = value


class SingularMatrixError(NumericaError):
    """A linear-algebra routine hit a singular or rank-deficient matrix."""


class DimensionError(NumericaError):
    """Array shapes are incompatible with the requested operation."""
