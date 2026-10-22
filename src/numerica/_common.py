"""Validation helpers shared by the public algorithms."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np

from numerica.errors import ConvergenceError, DimensionError

ScalarFunction = Callable[[float], float]


def require_finite_scalar(value: Any, *, name: str) -> float:
    """Convert ``value`` to a finite float or raise."""
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a real number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite, got {number}")
    return number


def require_positive(value: float, *, name: str) -> float:
    """Return ``value`` when it is a positive finite float."""
    number = require_finite_scalar(value, name=name)
    if number <= 0.0:
        raise ValueError(f"{name} must be positive")
    return number


def require_positive_int(value: int, *, name: str) -> int:
    """Return ``value`` when it is an integer greater than or equal to 1."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    number = int(value)
    if number < 1:
        raise ValueError(f"{name} must be >= 1")
    return number


def as_float_array(value: Any, *, name: str) -> np.ndarray:
    """Convert ``value`` to a finite floating-point array."""
    try:
        array = np.array(value, dtype=float, copy=True)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a real array") from exc
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def column_vector(value: Any, *, name: str, length: int | None = None) -> np.ndarray:
    """Return a one-dimensional finite array, optionally checking its length."""
    array = as_float_array(value, name=name)
    if array.ndim != 1:
        raise DimensionError(f"{name} must be one-dimensional, got shape {array.shape}")
    if length is not None and array.shape[0] != length:
        raise DimensionError(f"{name} must have length {length}, got {array.shape[0]}")
    return array


def square_matrix(value: Any, *, name: str = "A") -> np.ndarray:
    """Return a non-empty square matrix."""
    array = as_float_array(value, name=name)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise DimensionError(f"{name} must be square, got shape {array.shape}")
    if array.shape[0] == 0:
        raise DimensionError(f"{name} must be non-empty")
    return array


def linear_right_hand_side(value: Any, n: int) -> np.ndarray:
    """Validate a right-hand side of length ``n`` or shape ``(n, n_rhs)``."""
    array = as_float_array(value, name="b")
    if array.ndim == 1:
        if array.shape[0] != n:
            raise DimensionError(f"b must have length {n}, got {array.shape[0]}")
        return array
    if array.ndim == 2:
        if array.shape[0] != n:
            raise DimensionError(f"b must have shape ({n},) or ({n}, n_rhs), got {array.shape}")
        if array.shape[1] == 0:
            raise DimensionError("b must have at least one right-hand side")
        return array
    raise DimensionError(f"b must be one- or two-dimensional, got shape {array.shape}")


def scalar_or_array(
    result: np.ndarray, *, scalar: bool, shape: tuple[int, ...]
) -> float | np.ndarray:
    """Match the rank of the caller's evaluation points."""
    if scalar:
        return float(np.asarray(result).reshape(()))
    return np.asarray(result, dtype=float).reshape(shape)


def ensure_converged(
    *,
    method: str,
    converged: bool,
    iterations: int,
    residual: float,
    tolerance: float,
    value: Any,
    raise_on_failure: bool,
) -> None:
    """Raise :class:`ConvergenceError` when a strict iteration failed."""
    if converged or not raise_on_failure:
        return
    raise ConvergenceError(
        (
            f"{method} did not converge after {iterations} iterations "
            f"(residual={residual:.3e}, tolerance={tolerance:.3e})."
        ),
        iterations=iterations,
        residual=residual,
        tolerance=tolerance,
        value=value,
    )


def increasing_nodes(x: Any, y: Any, *, minimum: int) -> tuple[np.ndarray, np.ndarray]:
    """Return finite nodes that are strictly increasing, with matching values."""
    nodes = column_vector(x, name="x")
    values = column_vector(y, name="y", length=nodes.size)
    if nodes.size < minimum:
        raise ValueError(f"At least {minimum} nodes are required, got {nodes.size}")
    if np.any(np.diff(nodes) <= 0.0):
        raise ValueError("x must be strictly increasing")
    return nodes, values
