"""Polynomial reproduction and spline smoothness."""

from __future__ import annotations

import numpy as np
import pytest

from numerica import (
    CubicSpline,
    LagrangeInterpolant,
    NewtonInterpolant,
    cubic_spline,
    divided_difference_table,
    lagrange,
    newton_divided_difference,
)


def test_lagrange_and_newton_reproduce_a_cubic() -> None:
    nodes = np.array([0.0, 1.0, 2.0, 3.0])
    values = nodes**3
    query = np.array([-0.5, 0.25, 1.5, 2.75, 3.0])
    expected = query**3
    assert lagrange(nodes, values, query) == pytest.approx(expected)
    assert newton_divided_difference(nodes, values, query) == pytest.approx(expected)
    assert LagrangeInterpolant(nodes, values)(1.5) == pytest.approx(1.5**3)
    assert NewtonInterpolant(nodes, values)(1.5) == pytest.approx(1.5**3)


def test_scalar_evaluation_returns_a_float() -> None:
    assert isinstance(lagrange([0.0, 1.0], [0.0, 1.0], 0.25), float)
    assert lagrange([0.0, 1.0], [0.0, 1.0], 0.25) == pytest.approx(0.25)


def test_divided_difference_table_of_a_quadratic() -> None:
    table = divided_difference_table([0.0, 1.0, 2.0], [0.0, 1.0, 4.0])
    assert table[0] == pytest.approx([0.0, 1.0, 1.0])


def test_nodes_must_be_strictly_increasing() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        lagrange([0.0, 0.0], [1.0, 2.0], 0.0)
    with pytest.raises(ValueError, match="strictly increasing"):
        newton_divided_difference([1.0, 0.0], [1.0, 2.0], 0.5)


def test_lagrange_and_newton_agree_on_sine() -> None:
    nodes = np.linspace(0.0, np.pi, 6)
    values = np.sin(nodes)
    query = np.linspace(0.1, 3.0, 8)
    assert lagrange(nodes, values, query) == pytest.approx(
        newton_divided_difference(nodes, values, query)
    )


def test_natural_spline_reproduces_a_line_and_hits_the_nodes() -> None:
    nodes = np.array([0.0, 0.5, 1.2, 2.0])
    values = 3.0 * nodes - 1.0
    spline = CubicSpline(nodes, values)
    query = np.linspace(0.0, 2.0, 15)
    assert spline(query) == pytest.approx(3.0 * query - 1.0)
    assert spline(nodes) == pytest.approx(values)
    assert spline.derivative(nodes[0], order=2) == pytest.approx(0.0, abs=1e-10)
    assert spline.derivative(nodes[-1], order=2) == pytest.approx(0.0, abs=1e-10)


def test_natural_spline_is_c1_at_the_knots() -> None:
    nodes = np.array([0.0, 1.0, 2.0, 3.0])
    values = np.array([0.0, 1.0, 0.0, 1.0])
    spline = CubicSpline(nodes, values)
    widths = np.diff(nodes)
    for index, width in enumerate(widths[:-1]):
        row = spline.coefficients[index]
        left_slope = row[1] + 2.0 * row[2] * width + 3.0 * row[3] * width**2
        right_slope = spline.coefficients[index + 1, 1]
        assert left_slope == pytest.approx(right_slope, abs=1e-10)


def test_clamped_spline_reproduces_a_cubic() -> None:
    def cubic(x: np.ndarray) -> np.ndarray:
        return x**3 - 2.0 * x + 0.5

    def slope(x: float) -> float:
        return 3.0 * x * x - 2.0

    nodes = np.linspace(-1.0, 2.0, 5)
    spline = CubicSpline(
        nodes,
        cubic(nodes),
        bc="clamped",
        dydx=(slope(float(nodes[0])), slope(float(nodes[-1]))),
    )
    query = np.linspace(-1.0, 2.0, 21)
    assert spline(query) == pytest.approx(cubic(query), abs=1e-10)
    assert cubic_spline(nodes, cubic(nodes), 0.3, bc="natural") == pytest.approx(
        CubicSpline(nodes, cubic(nodes))(0.3)
    )


def test_spline_rejects_queries_outside_the_interval() -> None:
    spline = CubicSpline([0.0, 1.0, 2.0], [0.0, 1.0, 0.0])
    with pytest.raises(ValueError, match="x_eval"):
        spline(2.5)
    with pytest.raises(ValueError, match="clamped"):
        CubicSpline([0.0, 1.0], [0.0, 1.0], bc="clamped")
    with pytest.raises(ValueError, match="dydx"):
        CubicSpline([0.0, 1.0], [0.0, 1.0], dydx=(0.0, 0.0))
