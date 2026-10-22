"""Linear and polynomial least squares via Householder QR."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike

from numerica._common import as_float_array, column_vector
from numerica.errors import DimensionError, SingularMatrixError
from numerica.linear import back_substitute
from numerica.results import FitResult


def least_squares(a: ArrayLike, b: ArrayLike) -> np.ndarray:
    """Solve the overdetermined system ``A x ≈ b`` in the least-squares sense.

    The factorization is Householder QR, written out here rather than taken
    from a solver routine. ``A`` must have at least as many rows as columns.
    The returned vector minimizes ``||A x - b||_2``.

    Raises
    ------
    DimensionError
        If ``A`` and ``b`` have incompatible shapes or the system is
        underdetermined.
    SingularMatrixError
        If ``A`` is rank deficient.
    """
    matrix = as_float_array(a, name="A")
    if matrix.ndim != 2:
        raise DimensionError(f"A must be two-dimensional, got shape {matrix.shape}")
    rows, cols = matrix.shape
    if rows == 0 or cols == 0:
        raise DimensionError("A must be non-empty")
    if rows < cols:
        raise DimensionError(
            f"least squares requires at least as many rows as columns, got shape {matrix.shape}"
        )
    rhs = column_vector(b, name="b", length=rows)

    triangular = np.array(matrix, copy=True)
    projected = rhs.copy()
    for column in range(cols):
        reflector_source = triangular[column:, column]
        sigma = float(np.linalg.norm(reflector_source))
        if sigma == 0.0:
            raise SingularMatrixError("Rank-deficient matrix")
        sign = 1.0 if reflector_source[0] >= 0.0 else -1.0
        reflector = reflector_source.copy()
        reflector[0] += sign * sigma
        norm_sq = float(np.dot(reflector, reflector))
        beta = 2.0 / norm_sq
        block = triangular[column:, column:]
        triangular[column:, column:] = block - beta * np.outer(reflector, reflector @ block)
        tail = projected[column:]
        projected[column:] = tail - beta * reflector * float(np.dot(reflector, tail))

    diagonal = np.abs(np.diag(triangular[:cols, :cols]))
    scale = float(np.max(diagonal))
    threshold = float(np.finfo(float).eps * max(rows, cols) * max(scale, 1.0))
    if np.any(diagonal <= threshold):
        raise SingularMatrixError("Rank-deficient matrix")
    return back_substitute(triangular[:cols, :cols], projected[:cols], tol=threshold)


def polynomial_fit(x: ArrayLike, y: ArrayLike, degree: int) -> FitResult:
    """Fit ``y ≈ c0 + c1 x + ... + c_degree x**degree`` by least squares.

    Coefficients are returned from degree 0 upward. With ``degree + 1``
    distinct nodes the fit interpolates; with more nodes it is the discrete
    least-squares polynomial. ``r_squared`` is the coefficient of determination.

    Parameters
    ----------
    x, y :
        Sample locations and observations, with the same length.
    degree :
        Nonnegative polynomial degree. Requires at least ``degree + 1`` points.

    Returns
    -------
    FitResult
        Callable polynomial plus the Euclidean residual norm.
    """
    if isinstance(degree, bool) or not isinstance(degree, (int, np.integer)):
        raise TypeError("degree must be an integer")
    order = int(degree)
    if order < 0:
        raise ValueError("degree must be nonnegative")
    nodes = column_vector(x, name="x")
    values = column_vector(y, name="y", length=nodes.size)
    if nodes.size < order + 1:
        raise ValueError(f"Need at least {order + 1} points to fit a degree-{order} polynomial")

    design = np.empty((nodes.size, order + 1))
    design[:, 0] = 1.0
    for power in range(1, order + 1):
        design[:, power] = design[:, power - 1] * nodes
    coefficients = least_squares(design, values)
    residual = values - design @ coefficients
    ss_res = float(np.dot(residual, residual))
    centered = values - float(np.mean(values))
    ss_tot = float(np.dot(centered, centered))
    if ss_tot == 0.0 and ss_res <= 1e-24:
        r_squared = 1.0
    elif ss_tot == 0.0:
        r_squared = 0.0
    else:
        r_squared = 1.0 - ss_res / ss_tot
    return FitResult(
        coefficients=coefficients,
        residual_norm=math.sqrt(ss_res),
        r_squared=r_squared,
        degree=order,
    )
