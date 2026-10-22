"""Direct and iterative solvers for linear systems."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from numerica._common import (
    column_vector,
    ensure_converged,
    linear_right_hand_side,
    require_positive,
    require_positive_int,
    square_matrix,
)
from numerica.errors import SingularMatrixError
from numerica.results import LinearResult


def _pivot_tolerance(matrix: np.ndarray, tol: float | None) -> float:
    if tol is None:
        scale = float(np.max(np.abs(matrix)))
        return float(np.finfo(float).eps * max(matrix.shape) * max(scale, 1.0))
    tolerance = float(tol)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("tol must be a nonnegative finite number")
    return tolerance


def forward_substitute(lower: np.ndarray, rhs: np.ndarray, *, tol: float) -> np.ndarray:
    """Solve a lower-triangular system ``L y = rhs``."""
    n = lower.shape[0]
    single = rhs.ndim == 1
    right = rhs.reshape(n, 1) if single else np.array(rhs, dtype=float, copy=True)
    solution = np.zeros_like(right)
    width = right.shape[1]
    for i in range(n):
        diagonal = lower[i, i]
        if abs(diagonal) <= tol:
            raise SingularMatrixError(f"Zero diagonal in L at index {i}")
        summed = np.zeros(width) if i == 0 else lower[i, :i] @ solution[:i, :]
        solution[i, :] = (right[i, :] - np.reshape(summed, -1)) / diagonal
    return solution[:, 0] if single else solution


def back_substitute(upper: np.ndarray, rhs: np.ndarray, *, tol: float) -> np.ndarray:
    """Solve an upper-triangular system ``U x = rhs``."""
    n = upper.shape[0]
    single = rhs.ndim == 1
    right = rhs.reshape(n, 1) if single else np.array(rhs, dtype=float, copy=True)
    solution = np.zeros_like(right)
    width = right.shape[1]
    for i in range(n - 1, -1, -1):
        diagonal = upper[i, i]
        if abs(diagonal) <= tol:
            raise SingularMatrixError(f"Zero diagonal in U at index {i}")
        tail = upper[i, i + 1 :]
        summed = np.zeros(width) if tail.size == 0 else tail @ solution[i + 1 :, :]
        solution[i, :] = (right[i, :] - np.reshape(summed, -1)) / diagonal
    return solution[:, 0] if single else solution


@dataclass(frozen=True, slots=True, eq=False)
class LUFactorization:
    """PA = LU factorization with partial pivoting.

    ``p`` is the permutation matrix, ``lower`` is the unit lower-triangular
    factor L, and ``u`` is upper triangular.
    """

    p: np.ndarray
    lower: np.ndarray
    u: np.ndarray
    tol: float

    def solve(self, b: ArrayLike) -> np.ndarray:
        """Solve ``A x = b`` for one or several right-hand sides."""
        n = self.u.shape[0]
        rhs = linear_right_hand_side(b, n)
        permuted = self.p @ rhs
        middle = forward_substitute(self.lower, permuted, tol=self.tol)
        return back_substitute(self.u, middle, tol=self.tol)


def lu_factor(a: ArrayLike, *, tol: float | None = None) -> LUFactorization:
    """Factor ``A`` as ``P A = L U`` with partial pivoting.

    Parameters
    ----------
    a :
        Square coefficient matrix.
    tol :
        Pivot threshold. The default scales machine epsilon by the size of
        ``A``. A pivot whose magnitude is at most ``tol`` is treated as zero.

    Returns
    -------
    LUFactorization

    Raises
    ------
    SingularMatrixError
        If a pivot falls below ``tol``.
    DimensionError
        If ``A`` is not square.
    """
    matrix = square_matrix(a)
    n = matrix.shape[0]
    threshold = _pivot_tolerance(matrix, tol)
    upper = np.array(matrix, copy=True)
    lower = np.eye(n)
    permutation = np.eye(n)
    for column in range(n - 1):
        pivot = column + int(np.argmax(np.abs(upper[column:, column])))
        if abs(upper[pivot, column]) <= threshold:
            raise SingularMatrixError(f"Zero pivot in column {column}")
        if pivot != column:
            upper[[column, pivot]] = upper[[pivot, column]]
            permutation[[column, pivot]] = permutation[[pivot, column]]
            if column > 0:
                lower[[column, pivot], :column] = lower[[pivot, column], :column]
        for row in range(column + 1, n):
            multiplier = upper[row, column] / upper[column, column]
            lower[row, column] = multiplier
            upper[row, column:] -= multiplier * upper[column, column:]
            upper[row, column] = 0.0
    if abs(upper[n - 1, n - 1]) <= threshold:
        raise SingularMatrixError(f"Zero pivot in column {n - 1}")
    return LUFactorization(p=permutation, lower=lower, u=upper, tol=threshold)


def lu_solve(a: ArrayLike, b: ArrayLike, *, tol: float | None = None) -> np.ndarray:
    """Solve ``A x = b`` by LU factorization with partial pivoting.

    ``b`` may be a vector of length ``n`` or a matrix of shape ``(n, n_rhs)``.
    """
    return lu_factor(a, tol=tol).solve(b)


def gaussian_elimination(a: ArrayLike, b: ArrayLike, *, tol: float | None = None) -> np.ndarray:
    """Solve ``A x = b`` by Gaussian elimination with partial pivoting.

    The elimination is written out explicitly (the same arithmetic as LU, kept
    here so the algorithm can be read on its own). ``b`` may contain several
    right-hand sides.

    Raises
    ------
    SingularMatrixError
        If a pivot falls below ``tol``.
    DimensionError
        If the shapes of ``A`` and ``b`` do not match.
    """
    matrix = square_matrix(a)
    n = matrix.shape[0]
    rhs = linear_right_hand_side(b, n)
    threshold = _pivot_tolerance(matrix, tol)
    upper = np.array(matrix, copy=True)
    right = np.array(rhs, copy=True)
    for column in range(n - 1):
        pivot = column + int(np.argmax(np.abs(upper[column:, column])))
        if abs(upper[pivot, column]) <= threshold:
            raise SingularMatrixError(f"Zero pivot in column {column}")
        if pivot != column:
            upper[[column, pivot]] = upper[[pivot, column]]
            right[[column, pivot]] = right[[pivot, column]]
        for row in range(column + 1, n):
            multiplier = upper[row, column] / upper[column, column]
            upper[row, column:] -= multiplier * upper[column, column:]
            right[row] -= multiplier * right[column]
    return back_substitute(upper, right, tol=threshold)


def _infinity_residual(matrix: np.ndarray, solution: np.ndarray, rhs: np.ndarray) -> float:
    return float(np.linalg.norm(matrix @ solution - rhs, ord=np.inf))


def _iterative_result(
    *,
    method: str,
    solution: np.ndarray,
    iterations: int,
    converged: bool,
    residual: float,
    tolerance: float,
    history: list[np.ndarray] | None,
    raise_on_failure: bool,
) -> LinearResult:
    ensure_converged(
        method=method,
        converged=converged,
        iterations=iterations,
        residual=residual,
        tolerance=tolerance,
        value=solution,
        raise_on_failure=raise_on_failure,
    )
    stored = tuple(history) if history is not None else None
    return LinearResult(
        solution=solution,
        iterations=iterations,
        converged=converged,
        residual=residual,
        tolerance=tolerance,
        method=method,
        history=stored,
    )


def _prepare_iteration(
    a: ArrayLike,
    b: ArrayLike,
    x0: ArrayLike | None,
    tol: float,
    max_iter: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, int]:
    matrix = square_matrix(a)
    rhs = column_vector(b, name="b", length=matrix.shape[0])
    tolerance = require_positive(tol, name="tol")
    limit = require_positive_int(max_iter, name="max_iter")
    if x0 is None:
        guess = np.zeros(matrix.shape[0])
    else:
        guess = column_vector(x0, name="x0", length=matrix.shape[0])
    diagonal = np.diag(matrix)
    if np.any(np.abs(diagonal) <= np.finfo(float).eps * max(1.0, float(np.max(np.abs(matrix))))):
        raise SingularMatrixError("Iterative method requires a nonzero diagonal")
    return matrix, rhs, guess, tolerance, limit


def jacobi(
    a: ArrayLike,
    b: ArrayLike,
    *,
    x0: ArrayLike | None = None,
    tol: float = 1e-10,
    max_iter: int = 500,
    record_history: bool = False,
    raise_on_failure: bool = True,
) -> LinearResult:
    """Solve ``A x = b`` by Jacobi iteration.

    Component ``i`` is updated from the previous vector:

    ``x_i = (b_i - sum_{j != i} a_ij x_j) / a_ii``.

    The residual is the infinity norm ``||A x - b||_inf``. Convergence is
    guaranteed for strictly diagonally dominant matrices, which this routine
    does not check.

    Parameters
    ----------
    a, b :
        Square matrix and right-hand side.
    x0 :
        Initial guess. Defaults to the zero vector.
    tol :
        Positive residual tolerance.
    max_iter :
        Maximum number of updates.
    record_history :
        Store every iterate, including ``x0``, on the result.
    raise_on_failure :
        Raise :class:`~numerica.ConvergenceError` if ``tol`` is not met.
    """
    matrix, rhs, guess, tolerance, limit = _prepare_iteration(a, b, x0, tol, max_iter)
    diagonal = np.diag(matrix).copy()
    off_diagonal = np.array(matrix, copy=True)
    np.fill_diagonal(off_diagonal, 0.0)
    current = guess
    history = [current.copy()] if record_history else None
    residual = _infinity_residual(matrix, current, rhs)
    converged = residual <= tolerance
    iterations = 0
    if not converged:
        for step in range(1, limit + 1):
            iterations = step
            current = (rhs - off_diagonal @ current) / diagonal
            residual = _infinity_residual(matrix, current, rhs)
            if history is not None:
                history.append(current.copy())
            if residual <= tolerance:
                converged = True
                break
    return _iterative_result(
        method="jacobi",
        solution=current.copy(),
        iterations=iterations,
        converged=converged,
        residual=residual,
        tolerance=tolerance,
        history=history,
        raise_on_failure=raise_on_failure,
    )


def gauss_seidel(
    a: ArrayLike,
    b: ArrayLike,
    *,
    x0: ArrayLike | None = None,
    tol: float = 1e-10,
    max_iter: int = 500,
    record_history: bool = False,
    raise_on_failure: bool = True,
) -> LinearResult:
    """Solve ``A x = b`` by Gauss-Seidel iteration.

    Updates are used as soon as they are computed, sweeping forward through
    the unknowns. The residual is ``||A x - b||_inf``. Symmetric positive
    definite or strictly diagonally dominant matrices converge; that hypothesis
    is left to the caller.

    Parameters
    ----------
    a, b, x0, tol, max_iter, record_history, raise_on_failure :
        Same meaning as in :func:`jacobi`.
    """
    matrix, rhs, guess, tolerance, limit = _prepare_iteration(a, b, x0, tol, max_iter)
    current = guess.copy()
    history = [current.copy()] if record_history else None
    residual = _infinity_residual(matrix, current, rhs)
    converged = residual <= tolerance
    iterations = 0
    n = matrix.shape[0]
    if not converged:
        for step in range(1, limit + 1):
            iterations = step
            for i in range(n):
                left = float(matrix[i, :i] @ current[:i])
                right = float(matrix[i, i + 1 :] @ current[i + 1 :])
                current[i] = (rhs[i] - left - right) / matrix[i, i]
            residual = _infinity_residual(matrix, current, rhs)
            if history is not None:
                history.append(current.copy())
            if residual <= tolerance:
                converged = True
                break
    return _iterative_result(
        method="gauss_seidel",
        solution=current.copy(),
        iterations=iterations,
        converged=converged,
        residual=residual,
        tolerance=tolerance,
        history=history,
        raise_on_failure=raise_on_failure,
    )
