"""Polynomial interpolation and cubic splines."""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from numerica._common import increasing_nodes, require_finite_scalar, scalar_or_array
from numerica.errors import NumericaError

BoundaryCondition = Literal["natural", "clamped"]


def _prepare_query(x_eval: ArrayLike) -> tuple[np.ndarray, bool, tuple[int, ...]]:
    array = np.asarray(x_eval, dtype=float)
    if not np.all(np.isfinite(array)):
        raise ValueError("x_eval must be finite")
    return array, np.ndim(x_eval) == 0, np.shape(x_eval)


def divided_difference_table(x: ArrayLike, y: ArrayLike) -> np.ndarray:
    """Build the Newton divided-difference table.

    Entry ``table[i, j]`` is the divided difference ``f[x_i, ..., x_{i+j}]``.
    The Newton coefficients are the first row, ``table[0, :]``.

    Parameters
    ----------
    x, y :
        Strictly increasing nodes and the corresponding values. At least one
        node is required.

    Returns
    -------
    ndarray
        Lower-triangular table stored in a square array. Entries that are not
        part of the table are zero.
    """
    nodes, values = increasing_nodes(x, y, minimum=1)
    n = nodes.size
    table = np.zeros((n, n))
    table[:, 0] = values
    for order in range(1, n):
        numerator = table[: n - order, order - 1]
        shifted = table[1 : n - order + 1, order - 1]
        denominator = nodes[order:] - nodes[: n - order]
        table[: n - order, order] = (shifted - numerator) / denominator
    return table


def _newton_coefficients(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return divided_difference_table(x, y)[0].copy()


def _evaluate_newton(nodes: np.ndarray, coefficients: np.ndarray, query: np.ndarray) -> np.ndarray:
    value = np.empty(query.shape, dtype=float)
    value[...] = coefficients[-1]
    for index in range(coefficients.size - 2, -1, -1):
        value = coefficients[index] + (query - nodes[index]) * value
    return value


class LagrangeInterpolant:
    """Lagrange interpolant stored at its nodes.

    The polynomial is evaluated from the direct product formula. For high
    degree or clustered nodes the Newton form or a cubic spline is more stable.
    """

    def __init__(self, x: ArrayLike, y: ArrayLike) -> None:
        self.x, self.y = increasing_nodes(x, y, minimum=1)

    def __call__(self, x_eval: ArrayLike) -> float | np.ndarray:
        """Evaluate the interpolant."""
        return lagrange(self.x, self.y, x_eval)


class NewtonInterpolant:
    """Newton divided-difference interpolant."""

    def __init__(self, x: ArrayLike, y: ArrayLike) -> None:
        self.x, self.y = increasing_nodes(x, y, minimum=1)
        self.table = divided_difference_table(self.x, self.y)
        self.coefficients = self.table[0].copy()

    def __call__(self, x_eval: ArrayLike) -> float | np.ndarray:
        """Evaluate the Newton form by nested multiplication."""
        query, scalar, shape = _prepare_query(x_eval)
        values = _evaluate_newton(self.x, self.coefficients, np.atleast_1d(query).astype(float))
        return scalar_or_array(values, scalar=scalar, shape=shape)


class CubicSpline:
    """C2 cubic spline with natural or clamped boundary conditions.

    On each interval ``[x_i, x_{i+1}]`` the spline is

    ``a + b (x - x_i) + c (x - x_i)**2 + d (x - x_i)**3``.

    ``coefficients`` has shape ``(n_intervals, 4)`` and stores those
    ``a, b, c, d`` rows. The construction follows the tridiagonal algorithms
    in Burden and Faires.
    """

    def __init__(
        self,
        x: ArrayLike,
        y: ArrayLike,
        *,
        bc: BoundaryCondition = "natural",
        dydx: tuple[float, float] | None = None,
    ) -> None:
        nodes, values = increasing_nodes(x, y, minimum=2)
        if bc not in ("natural", "clamped"):
            raise ValueError("bc must be 'natural' or 'clamped'")
        if bc == "clamped" and dydx is None:
            raise ValueError("clamped splines require dydx=(left, right)")
        if bc == "natural" and dydx is not None:
            raise ValueError("dydx is only used when bc='clamped'")

        self.x = nodes
        self.y = values
        self.bc = bc
        self.dydx = dydx
        if bc == "natural":
            self.coefficients = _natural_coefficients(nodes, values)
        else:
            assert dydx is not None
            left = require_finite_scalar(dydx[0], name="dydx[0]")
            right = require_finite_scalar(dydx[1], name="dydx[1]")
            self.dydx = (left, right)
            self.coefficients = _clamped_coefficients(nodes, values, left, right)

    def __call__(self, x_eval: ArrayLike) -> float | np.ndarray:
        """Evaluate the spline. Points outside the nodes raise ``ValueError``."""
        return self._evaluate(x_eval, order=0)

    def derivative(self, x_eval: ArrayLike, *, order: int = 1) -> float | np.ndarray:
        """Evaluate an exact piecewise derivative of order 1, 2, or 3."""
        if order not in (1, 2, 3):
            raise ValueError("order must be 1, 2, or 3")
        return self._evaluate(x_eval, order=order)

    def _evaluate(self, x_eval: ArrayLike, *, order: int) -> float | np.ndarray:
        query, scalar, shape = _prepare_query(x_eval)
        flat = np.atleast_1d(query).astype(float, copy=False)
        left, right = float(self.x[0]), float(self.x[-1])
        pad = 1e-12 * max(1.0, abs(right - left))
        if np.any(flat < left - pad) or np.any(flat > right + pad):
            raise ValueError(f"x_eval must lie in [{left}, {right}]")
        flat = np.clip(flat, left, right)
        index = np.searchsorted(self.x, flat, side="right") - 1
        index = np.clip(index, 0, self.x.size - 2)
        dx = flat - self.x[index]
        coeff = self.coefficients[index]
        a, b, c, d = coeff[:, 0], coeff[:, 1], coeff[:, 2], coeff[:, 3]
        if order == 0:
            values = a + dx * (b + dx * (c + dx * d))
        elif order == 1:
            values = b + dx * (2.0 * c + dx * 3.0 * d)
        elif order == 2:
            values = 2.0 * c + 6.0 * d * dx
        else:
            values = np.full(flat.shape, 6.0) * d
        return scalar_or_array(values, scalar=scalar, shape=shape)


def lagrange(x: ArrayLike, y: ArrayLike, x_eval: ArrayLike) -> float | np.ndarray:
    """Evaluate the Lagrange interpolant of ``(x, y)`` at ``x_eval``.

    Parameters
    ----------
    x, y :
        Strictly increasing nodes and values.
    x_eval :
        Scalar or array of evaluation points. A scalar input returns a float.

    Returns
    -------
    float or ndarray
        Interpolated values, with the same shape as ``x_eval``.
    """
    nodes, values = increasing_nodes(x, y, minimum=1)
    query, scalar, shape = _prepare_query(x_eval)
    flat = np.atleast_1d(query).astype(float)
    result = np.zeros(flat.shape, dtype=float)
    for j in range(nodes.size):
        basis = np.ones(flat.shape, dtype=float)
        for i in range(nodes.size):
            if i == j:
                continue
            basis *= (flat - nodes[i]) / (nodes[j] - nodes[i])
        result += values[j] * basis
    return scalar_or_array(result, scalar=scalar, shape=shape)


def newton_divided_difference(x: ArrayLike, y: ArrayLike, x_eval: ArrayLike) -> float | np.ndarray:
    """Evaluate the Newton divided-difference interpolant of ``(x, y)``."""
    return NewtonInterpolant(x, y)(x_eval)


def cubic_spline(
    x: ArrayLike,
    y: ArrayLike,
    x_eval: ArrayLike,
    *,
    bc: BoundaryCondition = "natural",
    dydx: tuple[float, float] | None = None,
) -> float | np.ndarray:
    """Evaluate a cubic spline of ``(x, y)`` at ``x_eval``.

    Parameters
    ----------
    x, y :
        Strictly increasing nodes and values. At least two nodes are required.
    x_eval :
        Points inside ``[x[0], x[-1]]``.
    bc :
        ``"natural"`` (second derivative zero at the ends) or ``"clamped"``.
    dydx :
        Endpoint derivatives ``(S'(x[0]), S'(x[-1]))``, required when
        ``bc="clamped"``.
    """
    return CubicSpline(x, y, bc=bc, dydx=dydx)(x_eval)


def _natural_coefficients(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Natural cubic spline coefficients (Burden and Faires)."""
    n = x.size - 1
    h = np.diff(x)
    alpha = np.zeros(n + 1)
    for i in range(1, n):
        alpha[i] = 3.0 / h[i] * (y[i + 1] - y[i]) - 3.0 / h[i - 1] * (y[i] - y[i - 1])

    ell = np.ones(n + 1)
    mu = np.zeros(n + 1)
    z = np.zeros(n + 1)
    for i in range(1, n):
        ell[i] = 2.0 * (h[i - 1] + h[i]) - h[i - 1] * mu[i - 1]
        if ell[i] == 0.0:
            raise NumericaError("Cubic spline tridiagonal system is singular")
        mu[i] = h[i] / ell[i]
        z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / ell[i]

    c = np.zeros(n + 1)
    b = np.zeros(n)
    d = np.zeros(n)
    for j in range(n - 1, -1, -1):
        c[j] = z[j] - mu[j] * c[j + 1]
        b[j] = (y[j + 1] - y[j]) / h[j] - h[j] * (c[j + 1] + 2.0 * c[j]) / 3.0
        d[j] = (c[j + 1] - c[j]) / (3.0 * h[j])
    return np.column_stack((y[:-1], b, c[:-1], d))


def _clamped_coefficients(
    x: np.ndarray,
    y: np.ndarray,
    fp_left: float,
    fp_right: float,
) -> np.ndarray:
    """Clamped cubic spline coefficients (Burden and Faires)."""
    n = x.size - 1
    h = np.diff(x)
    alpha = np.zeros(n + 1)
    alpha[0] = 3.0 * (y[1] - y[0]) / h[0] - 3.0 * fp_left
    alpha[n] = 3.0 * fp_right - 3.0 * (y[n] - y[n - 1]) / h[n - 1]
    for i in range(1, n):
        alpha[i] = 3.0 / h[i] * (y[i + 1] - y[i]) - 3.0 / h[i - 1] * (y[i] - y[i - 1])

    ell = np.zeros(n + 1)
    mu = np.zeros(n + 1)
    z = np.zeros(n + 1)
    ell[0] = 2.0 * h[0]
    mu[0] = 0.5
    z[0] = alpha[0] / ell[0]
    for i in range(1, n):
        ell[i] = 2.0 * (h[i - 1] + h[i]) - h[i - 1] * mu[i - 1]
        if ell[i] == 0.0:
            raise NumericaError("Cubic spline tridiagonal system is singular")
        mu[i] = h[i] / ell[i]
        z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / ell[i]
    ell[n] = h[n - 1] * (2.0 - mu[n - 1])
    if ell[n] == 0.0:
        raise NumericaError("Cubic spline tridiagonal system is singular")
    z[n] = (alpha[n] - h[n - 1] * z[n - 1]) / ell[n]

    c = np.zeros(n + 1)
    b = np.zeros(n)
    d = np.zeros(n)
    c[n] = z[n]
    for j in range(n - 1, -1, -1):
        c[j] = z[j] - mu[j] * c[j + 1]
        b[j] = (y[j + 1] - y[j]) / h[j] - h[j] * (c[j + 1] + 2.0 * c[j]) / 3.0
        d[j] = (c[j + 1] - c[j]) / (3.0 * h[j])
    return np.column_stack((y[:-1], b, c[:-1], d))
