"""Least squares against exact polynomials and normal equations."""

from __future__ import annotations

import numpy as np
import pytest

from numerica import DimensionError, SingularMatrixError, least_squares, polynomial_fit


def test_polynomial_fit_recovers_a_quadratic() -> None:
    nodes = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    observations = 3.0 + 2.0 * nodes + nodes**2
    fit = polynomial_fit(nodes, observations, 2)
    assert fit.coefficients == pytest.approx([3.0, 2.0, 1.0])
    assert fit.r_squared == pytest.approx(1.0)
    assert fit.residual_norm == pytest.approx(0.0, abs=1e-10)
    assert fit(2.5) == pytest.approx(3.0 + 2.0 * 2.5 + 2.5**2)
    assert str(fit)


def test_degree_zero_is_the_sample_mean() -> None:
    varying = polynomial_fit([0.0, 1.0, 2.0], [2.0, 4.0, 6.0], 0)
    constant = polynomial_fit([0.0, 1.0], [5.0, 5.0], 0)
    assert varying.coefficients[0] == pytest.approx(4.0)
    assert varying.r_squared == pytest.approx(0.0)
    assert constant.r_squared == pytest.approx(1.0)


def test_least_squares_mean_and_line() -> None:
    mean = least_squares([[1.0], [1.0], [1.0]], [1.0, 2.0, 3.0])
    assert mean == pytest.approx([2.0])
    design = np.column_stack((np.ones(4), np.arange(4.0)))
    observations = 1.0 + 2.0 * np.arange(4.0)
    assert least_squares(design, observations) == pytest.approx([1.0, 2.0])


def test_rank_and_shape_errors() -> None:
    with pytest.raises(SingularMatrixError):
        polynomial_fit([1.0, 1.0, 1.0], [1.0, 2.0, 3.0], 1)
    with pytest.raises(ValueError, match="at least"):
        polynomial_fit([0.0, 1.0], [0.0, 1.0], 2)
    with pytest.raises(DimensionError):
        least_squares([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], [1.0, 2.0])
    with pytest.raises(TypeError):
        polynomial_fit([0.0, 1.0], [0.0, 1.0], 1.5)  # type: ignore[arg-type]
