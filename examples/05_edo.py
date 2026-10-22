"""y' = y, y(0) = 1. La solución es exp(t)."""

from __future__ import annotations

import math

from numerica import euler, heun, rk4, rk45


def main() -> None:
    exact = math.e
    print(f"exp(1) = {exact:.12f}")
    for method, step in ((euler, 0.1), (heun, 0.1), (rk4, 0.1)):
        solution = method(lambda t, y: y, (0.0, 1.0), 1.0, h=step)
        error = abs(float(solution.y[-1]) - exact)
        print(f"  {solution.method:<6} h={step:<4}  y(1)={solution.y[-1]:.12f}  error {error:.2e}")

    adaptive = rk45(lambda t, y: y, (0.0, 1.0), 1.0, rtol=1e-8, atol=1e-10)
    error = abs(float(adaptive.y[-1]) - exact)
    print(
        f"  rk45   pasos={adaptive.accepted_steps:<3} "
        f"(rechazados {adaptive.rejected_steps})  "
        f"y(1)={adaptive.y[-1]:.12f}  error {error:.2e}"
    )


if __name__ == "__main__":
    main()
