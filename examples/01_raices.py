"""Raíces de una ecuación escalar con cuatro métodos clásicos."""

from __future__ import annotations

import math

from numerica import bisection, fixed_point, newton, secant


def f(x: float) -> float:
    return x * x - 2.0


def main() -> None:
    print("Raíz cuadrada de 2")
    bracket = bisection(f, 0.0, 2.0)
    print(f"  bisección      {bracket.root:.12f}  ({bracket.iterations} iteraciones)")
    tangent = newton(f, 1.0, df=lambda x: 2.0 * x)
    print(f"  Newton         {tangent.root:.12f}  ({tangent.iterations} iteraciones)")
    chord = secant(f, 1.0, 2.0)
    print(f"  secante        {chord.root:.12f}  ({chord.iterations} iteraciones)")
    babylon = fixed_point(lambda x: 0.5 * (x + 2.0 / x), 1.0)
    print(f"  punto fijo     {babylon.root:.12f}  ({babylon.iterations} iteraciones)")
    print(f"  valor exacto   {math.sqrt(2.0):.12f}")

    print("\nNúmero de Dottie, punto fijo de cos")
    dottie = fixed_point(math.cos, 0.0)
    print(f"  x = cos(x) ≈ {dottie.root:.12f}  residual {dottie.residual:.2e}")


if __name__ == "__main__":
    main()
