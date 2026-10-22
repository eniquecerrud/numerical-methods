"""Newton-Cotes rules, Gauss-Legendre quadrature, and Romberg integration."""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from numerica._common import (
    ScalarFunction,
    ensure_converged,
    require_finite_scalar,
    require_positive,
    require_positive_int,
)
from numerica.results import IntegralResult


def _interval(a: float, b: float) -> tuple[float, float]:
    left = require_finite_scalar(a, name="a")
    right = require_finite_scalar(b, name="b")
    return left, right


def _samples(f: ScalarFunction, points: np.ndarray) -> np.ndarray:
    return np.array(
        [require_finite_scalar(f(float(point)), name="f(x)") for point in points],
        dtype=float,
    )


def _fixed_result(value: float, *, method: str, intervals: int, evaluations: int) -> IntegralResult:
    return IntegralResult(
        value=value,
        method=method,
        intervals=intervals,
        evaluations=evaluations,
    )


def trapezoidal(f: ScalarFunction, a: float, b: float, *, n: int = 128) -> IntegralResult:
    """Composite trapezoidal rule on ``n`` subintervals.

    The rule is exact for polynomials of degree at most 1. ``n`` is the number
    of panels, so the integrand is sampled at ``n + 1`` points. If ``a > b``
    the integral changes sign. A zero-length interval returns ``0``.
    """
    panels = require_positive_int(n, name="n")
    left, right = _interval(a, b)
    if left == right:
        return _fixed_result(0.0, method="trapezoidal", intervals=0, evaluations=0)
    points = np.linspace(left, right, panels + 1)
    values = _samples(f, points)
    step = (right - left) / panels
    total = step * (0.5 * values[0] + float(np.sum(values[1:-1])) + 0.5 * values[-1])
    return _fixed_result(total, method="trapezoidal", intervals=panels, evaluations=panels + 1)


def simpson(f: ScalarFunction, a: float, b: float, *, n: int = 128) -> IntegralResult:
    """Composite Simpson's 1/3 rule on ``n`` subintervals.

    ``n`` must be even. The rule is exact for polynomials of degree at most 3.
    """
    panels = require_positive_int(n, name="n")
    if panels % 2 != 0:
        raise ValueError("Simpson's 1/3 rule requires an even number of subintervals")
    left, right = _interval(a, b)
    if left == right:
        return _fixed_result(0.0, method="simpson", intervals=0, evaluations=0)
    points = np.linspace(left, right, panels + 1)
    values = _samples(f, points)
    weights = np.ones(panels + 1)
    weights[1:-1:2] = 4.0
    weights[2:-1:2] = 2.0
    step = (right - left) / panels
    total = step / 3.0 * float(np.dot(weights, values))
    return _fixed_result(total, method="simpson", intervals=panels, evaluations=panels + 1)


def simpson_38(f: ScalarFunction, a: float, b: float, *, n: int = 129) -> IntegralResult:
    """Composite Simpson's 3/8 rule on ``n`` subintervals.

    ``n`` must be a multiple of 3. The rule is exact for polynomials of degree
    at most 3.
    """
    panels = require_positive_int(n, name="n")
    if panels % 3 != 0:
        raise ValueError("Simpson's 3/8 rule requires a multiple of 3 subintervals")
    left, right = _interval(a, b)
    if left == right:
        return _fixed_result(0.0, method="simpson_38", intervals=0, evaluations=0)
    points = np.linspace(left, right, panels + 1)
    values = _samples(f, points)
    weights = np.ones(panels + 1)
    weights[1:-1] = 3.0
    weights[3:-1:3] = 2.0
    step = (right - left) / panels
    total = 3.0 * step / 8.0 * float(np.dot(weights, values))
    return _fixed_result(total, method="simpson_38", intervals=panels, evaluations=panels + 1)


@lru_cache(maxsize=64)
def _legendre_rule(n: int) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Gauss-Legendre nodes and weights on ``[-1, 1]`` via Golub-Welsch."""
    if n == 1:
        return (0.0,), (2.0,)
    index = np.arange(1, n, dtype=float)
    beta = index / np.sqrt(4.0 * index * index - 1.0)
    # Symmetric tridiagonal Jacobi matrix for the Legendre polynomials.
    jacobi = np.diag(beta, k=1) + np.diag(beta, k=-1)
    nodes, vectors = np.linalg.eigh(jacobi)
    weights = 2.0 * vectors[0, :] ** 2
    return tuple(float(node) for node in nodes), tuple(float(weight) for weight in weights)


def legendre_nodes(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Return Gauss-Legendre nodes and weights on ``[-1, 1]``.

    ``n`` points integrate polynomials of degree up to ``2 n - 1`` exactly.
    Nodes and weights are computed by the Golub-Welsch eigenvalue method, so
    the only external linear-algebra primitive is a symmetric eigensolve.

    Raises
    ------
    ValueError
        If ``n`` is outside ``1 .. 256``.
    """
    order = require_positive_int(n, name="n")
    if order > 256:
        raise ValueError("n must be <= 256")
    nodes, weights = _legendre_rule(order)
    return np.array(nodes), np.array(weights)


def gauss_legendre(f: ScalarFunction, a: float, b: float, *, n: int = 8) -> IntegralResult:
    """Gauss-Legendre quadrature with ``n`` nodes, mapped onto ``[a, b]``.

    The rule is exact for polynomials of degree at most ``2 n - 1``.
    """
    order = require_positive_int(n, name="n")
    left, right = _interval(a, b)
    if left == right:
        return _fixed_result(0.0, method="gauss_legendre", intervals=order, evaluations=0)
    nodes, weights = legendre_nodes(order)
    half = 0.5 * (right - left)
    center = 0.5 * (right + left)
    points = center + half * nodes
    total = half * float(np.dot(weights, _samples(f, points)))
    return _fixed_result(total, method="gauss_legendre", intervals=order, evaluations=order)


def romberg(
    f: ScalarFunction,
    a: float,
    b: float,
    *,
    tol: float = 1e-10,
    max_level: int = 12,
    raise_on_failure: bool = True,
) -> IntegralResult:
    """Romberg integration until successive diagonal entries differ by ``tol``.

    Level ``k`` uses the composite trapezoidal rule with ``2**k`` panels and
    Richardson extrapolation. ``max_level`` is the number of trapezoidal
    rows, so the finest grid has ``2**(max_level - 1)`` panels. ``residual``
    is the absolute difference between the last two diagonal entries.

    Raises
    ------
    ConvergenceError
        If the tolerance is not met within ``max_level`` rows and
        ``raise_on_failure`` is ``True``.
    """
    tolerance = require_positive(tol, name="tol")
    levels = require_positive_int(max_level, name="max_level")
    if levels < 2:
        raise ValueError("max_level must be at least 2")
    left, right = _interval(a, b)
    if left == right:
        return IntegralResult(
            value=0.0,
            method="romberg",
            intervals=0,
            evaluations=0,
            iterations=0,
            converged=True,
            residual=0.0,
            tolerance=tolerance,
        )

    step = right - left
    total = (
        0.5
        * step
        * (
            require_finite_scalar(f(left), name="f(x)")
            + require_finite_scalar(f(right), name="f(x)")
        )
    )
    evaluations = 2
    previous = [total]
    estimate = total
    residual = float("inf")
    converged = False
    iterations = 0
    intervals = 1

    for level in range(1, levels):
        step *= 0.5
        new_points = 2 ** (level - 1)
        points = left + step * (2.0 * np.arange(new_points) + 1.0)
        total = 0.5 * total + step * float(np.sum(_samples(f, points)))
        evaluations += new_points
        row = [total]
        for column in range(1, level + 1):
            factor = 4.0**column
            extrapolated = (factor * row[column - 1] - previous[column - 1]) / (factor - 1.0)
            row.append(extrapolated)
        estimate = row[-1]
        residual = abs(row[-1] - previous[-1])
        iterations = level
        intervals = 2**level
        previous = row
        if residual <= tolerance:
            converged = True
            break

    ensure_converged(
        method="romberg",
        converged=converged,
        iterations=iterations,
        residual=residual,
        tolerance=tolerance,
        value=estimate,
        raise_on_failure=raise_on_failure,
    )
    return IntegralResult(
        value=estimate,
        method="romberg",
        intervals=intervals,
        evaluations=evaluations,
        iterations=iterations,
        converged=converged,
        residual=residual,
        tolerance=tolerance,
    )
