"""Quadrature against exact integrals."""

from __future__ import annotations

import math

import numpy as np
import pytest

from numerica import (
    ConvergenceError,
    gauss_legendre,
    legendre_nodes,
    romberg,
    simpson,
    simpson_38,
    trapezoidal,
)


def test_newton_cotes_are_exact_on_their_polynomial_degree() -> None:
    assert trapezoidal(lambda x: 2.0 * x, 0.0, 1.0, n=1).value == pytest.approx(1.0)
    assert simpson(lambda x: x**3, 0.0, 1.0, n=2).value == pytest.approx(0.25)
    assert simpson_38(lambda x: x**3, 0.0, 1.0, n=3).value == pytest.approx(0.25)
    assert trapezoidal(lambda x: 1.0, 3.0, 1.0, n=4).value == pytest.approx(-2.0)


def test_composite_rules_integrate_sine() -> None:
    assert simpson(math.sin, 0.0, math.pi, n=100).value == pytest.approx(2.0, abs=1e-6)
    assert simpson_38(math.sin, 0.0, math.pi, n=99).value == pytest.approx(2.0, abs=1e-4)
    assert trapezoidal(math.sin, 0.0, math.pi, n=200).value == pytest.approx(2.0, abs=1e-4)


def test_panel_counts_are_validated() -> None:
    with pytest.raises(ValueError, match="even"):
        simpson(math.sin, 0.0, 1.0, n=3)
    with pytest.raises(ValueError, match="multiple of 3"):
        simpson_38(math.sin, 0.0, 1.0, n=2)
    with pytest.raises(TypeError):
        trapezoidal(math.sin, 0.0, 1.0, n=True)  # type: ignore[arg-type]


def test_zero_interval_is_zero() -> None:
    result = simpson(math.exp, 2.0, 2.0, n=4)
    assert result.value == 0.0
    assert result.evaluations == 0


def test_gauss_legendre_rule_and_exactness() -> None:
    nodes, weights = legendre_nodes(2)
    assert nodes == pytest.approx([-1.0 / math.sqrt(3.0), 1.0 / math.sqrt(3.0)])
    assert weights == pytest.approx([1.0, 1.0])
    assert float(np.sum(legendre_nodes(7)[1])) == pytest.approx(2.0)
    assert gauss_legendre(lambda x: x**5, 0.0, 2.0, n=3).value == pytest.approx(32.0 / 3.0)
    assert gauss_legendre(math.exp, 0.0, 1.0, n=8).value == pytest.approx(math.e - 1.0, abs=1e-12)
    assert gauss_legendre(lambda x: 1.0, 2.0, -1.0, n=3).value == pytest.approx(-3.0)


def test_romberg_matches_the_exponential_integral() -> None:
    result = romberg(math.exp, 0.0, 1.0, tol=1e-12)
    assert result.converged
    assert result.value == pytest.approx(math.e - 1.0, abs=1e-12)
    assert result.residual is not None and result.residual <= result.tolerance
    backward = romberg(math.exp, 1.0, 0.0, tol=1e-10)
    assert backward.value == pytest.approx(-result.value, abs=1e-9)


def test_romberg_reports_failure() -> None:
    with pytest.raises(ConvergenceError) as caught:
        romberg(math.exp, 0.0, 1.0, tol=1e-16, max_level=2)
    assert caught.value.iterations == 1
    assert caught.value.value is not None
