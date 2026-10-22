"""Power iterations against known spectra."""

from __future__ import annotations

import math

import numpy as np
import pytest

from numerica import ConvergenceError, SingularMatrixError, inverse_power_method, power_method


def test_power_method_on_a_symmetric_matrix() -> None:
    matrix = np.array([[2.0, 1.0], [1.0, 2.0]])
    result = power_method(matrix, tol=1e-12)
    assert result.converged
    assert result.eigenvalue == pytest.approx(3.0)
    assert result.eigenvector == pytest.approx(np.ones(2) / math.sqrt(2.0))
    assert result.residual <= result.tolerance
    assert result.history[-1] == pytest.approx(3.0)


def test_power_method_keeps_the_sign_of_a_negative_eigenvalue() -> None:
    result = power_method(np.diag([2.0, -5.0]), tol=1e-10)
    assert result.eigenvalue == pytest.approx(-5.0)
    assert result.eigenvector == pytest.approx([0.0, 1.0], abs=1e-8)


def test_inverse_power_finds_the_closest_eigenvalue() -> None:
    matrix = np.diag([1.0, 3.0, 10.0])
    result = inverse_power_method(matrix, shift=2.4, tol=1e-12)
    assert result.eigenvalue == pytest.approx(3.0, abs=1e-10)
    assert abs(result.eigenvector[1]) == pytest.approx(1.0)
    assert result.eigenvector[0] == pytest.approx(0.0, abs=1e-8)
    assert result.eigenvector[2] == pytest.approx(0.0, abs=1e-8)


def test_a_rotation_does_not_converge_in_real_arithmetic() -> None:
    rotation = np.array([[0.0, -1.0], [1.0, 0.0]])
    with pytest.raises(ConvergenceError):
        power_method(rotation, max_iter=15, tol=1e-8)
    failed = power_method(rotation, max_iter=10, tol=1e-8, raise_on_failure=False)
    assert failed.converged is False


def test_shift_equal_to_an_eigenvalue_is_singular() -> None:
    with pytest.raises(SingularMatrixError):
        inverse_power_method(np.diag([1.0, 4.0]), shift=4.0)
