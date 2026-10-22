"""Finite-difference derivatives."""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from numerica._common import ScalarFunction, require_finite_scalar, require_positive

DifferenceMethod = Literal["forward", "backward", "central"]


def derivative(
    f: ScalarFunction,
    x: ArrayLike,
    *,
    order: int = 1,
    method: DifferenceMethod = "central",
    step: float | None = None,
) -> float | np.ndarray:
    """Approximate the derivative of ``f`` by finite differences.

    The actual increment at a point is ``step * max(1, abs(x))``, so ``step``
    is a relative spacing. Defaults are ``1e-6`` for a first derivative and
    ``1e-4`` for a second derivative.

    Parameters
    ----------
    f :
        Scalar function.
    x :
        Point or array of points.
    order :
        ``1`` or ``2``.
    method :
        ``"forward"``, ``"backward"``, or ``"central"``. Central differences
        are second-order; one-sided differences are first-order.
    step :
        Relative spacing. Must be positive.

    Returns
    -------
    float or ndarray
        A float when ``x`` is a scalar, otherwise an array of the same shape.

    Raises
    ------
    ValueError
        If ``order``, ``method``, or ``step`` is invalid.
    """
    if order not in (1, 2):
        raise ValueError("order must be 1 or 2")
    if method not in ("forward", "backward", "central"):
        raise ValueError("method must be 'forward', 'backward', or 'central'")
    spacing = 1e-6 if order == 1 else 1e-4
    if step is not None:
        spacing = require_positive(step, name="step")

    x_arr = np.asarray(x, dtype=float)
    if not np.all(np.isfinite(x_arr)):
        raise ValueError("x must be finite")
    increment = spacing * np.maximum(1.0, np.abs(x_arr))

    def values(points: np.ndarray) -> np.ndarray:
        flat = np.reshape(points, -1)
        samples = np.array(
            [require_finite_scalar(f(float(point)), name="f(x)") for point in flat],
            dtype=float,
        )
        return samples.reshape(points.shape)

    if order == 1 and method == "forward":
        estimate = (values(x_arr + increment) - values(x_arr)) / increment
    elif order == 1 and method == "backward":
        estimate = (values(x_arr) - values(x_arr - increment)) / increment
    elif order == 1:
        estimate = (values(x_arr + increment) - values(x_arr - increment)) / (2.0 * increment)
    elif method == "forward":
        estimate = (
            values(x_arr) - 2.0 * values(x_arr + increment) + values(x_arr + 2.0 * increment)
        ) / increment**2
    elif method == "backward":
        estimate = (
            values(x_arr) - 2.0 * values(x_arr - increment) + values(x_arr - 2.0 * increment)
        ) / increment**2
    else:
        estimate = (
            values(x_arr - increment) - 2.0 * values(x_arr) + values(x_arr + increment)
        ) / increment**2

    if np.ndim(x) == 0:
        return float(estimate)
    return estimate
