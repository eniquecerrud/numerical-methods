"""Direct and iterative linear solvers against manufactured solutions."""

from __future__ import annotations

import numpy as np
import pytest

from numerica import (
    ConvergenceError,
    DimensionError,
    SingularMatrixError,
    gauss_seidel,
    gaussian_elimination,
    jacobi,
    lu_factor,
    lu_solve,
)


def test_direct_solvers_recover_a_manufactured_solution() -> None:
    matrix = np.array([[3.0, 2.0, -1.0], [2.0, -2.0, 4.0], [-1.0, 0.5, -1.0]])
    expected = np.array([1.0, -2.0, 0.5])
    rhs = matrix @ expected
    assert gaussian_elimination(matrix, rhs) == pytest.approx(expected)
    assert lu_solve(matrix, rhs) == pytest.approx(expected)


def test_multiple_right_hand_sides() -> None:
    matrix = np.array([[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]])
    expected = np.array([[1.0, 0.0], [2.0, -1.0], [-1.0, 1.0]])
    rhs = matrix @ expected
    assert gaussian_elimination(matrix, rhs) == pytest.approx(expected)
    assert lu_solve(matrix, rhs) == pytest.approx(expected, abs=1e-12)


def test_partial_pivoting_handles_a_tiny_leading_entry() -> None:
    matrix = np.array([[1e-20, 1.0], [1.0, 1.0]])
    expected = np.array([1.0, 1.0])
    solution = gaussian_elimination(matrix, matrix @ expected)
    assert solution == pytest.approx(expected, abs=1e-12)


def test_lu_reconstructs_the_permuted_matrix() -> None:
    matrix = np.array([[0.0, 2.0, 3.0], [1.0, 1.0, -1.0], [2.0, 0.0, 1.0]])
    factors = lu_factor(matrix)
    assert factors.p @ matrix == pytest.approx(factors.lower @ factors.u)
    expected = np.array([1.0, -1.0, 2.0])
    assert factors.solve(matrix @ expected) == pytest.approx(expected)


def test_singular_and_malformed_systems() -> None:
    singular = np.array([[1.0, 2.0], [2.0, 4.0]])
    with pytest.raises(SingularMatrixError):
        gaussian_elimination(singular, [1.0, 2.0])
    with pytest.raises(SingularMatrixError):
        lu_factor(singular)
    with pytest.raises(DimensionError):
        lu_solve(np.eye(2), [1.0, 2.0, 3.0])
    with pytest.raises(DimensionError):
        gaussian_elimination([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [1.0, 2.0])


def test_iterative_solvers_on_a_diagonally_dominant_system() -> None:
    matrix = np.array([[10.0, -1.0, 2.0], [-1.0, 11.0, -1.0], [2.0, -1.0, 10.0]])
    expected = np.array([1.0, 2.0, -1.0])
    rhs = matrix @ expected
    jacobi_result = jacobi(matrix, rhs, tol=1e-12, record_history=True)
    seidel_result = gauss_seidel(matrix, rhs, tol=1e-12, record_history=True)
    assert jacobi_result.solution == pytest.approx(expected, abs=1e-10)
    assert seidel_result.solution == pytest.approx(expected, abs=1e-10)
    assert jacobi_result.residual <= jacobi_result.tolerance
    assert seidel_result.history is not None
    assert seidel_result.history[0] == pytest.approx(np.zeros(3))
    assert seidel_result.iterations <= jacobi_result.iterations


def test_iterative_solver_can_report_nonconvergence() -> None:
    matrix = np.array([[4.0, 1.0], [1.0, 3.0]])
    rhs = matrix @ np.array([1.0, 1.0])
    result = jacobi(matrix, rhs, tol=1e-30, max_iter=1, raise_on_failure=False)
    assert result.converged is False
    assert result.iterations == 1
    with pytest.raises(ConvergenceError):
        gauss_seidel(matrix, rhs, tol=1e-30, max_iter=1)
    with pytest.raises(SingularMatrixError):
        jacobi([[0.0, 1.0], [1.0, 1.0]], [1.0, 2.0])
