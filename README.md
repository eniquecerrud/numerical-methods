# numerica

[![CI](https://github.com/eniquecerrud/numerical-methods/actions/workflows/ci.yml/badge.svg)](https://github.com/eniquecerrud/numerical-methods/actions/workflows/ci.yml)

Biblioteca de análisis numérico en Python. Los métodos clásicos de un curso de cálculo numérico están escritos de forma explícita con NumPy: se pueden leer, probar contra resultados analíticos y reutilizar sin envolver SciPy.

El paquete se importa como `numerica`. Las docstrings del código están en inglés; esta guía está en español.

## Instalación

Python 3.10 o posterior y NumPy. Desde el repositorio:

```bash
git clone https://github.com/eniquecerrud/numerical-methods.git
cd numerical-methods
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

`pip install -e .` basta si solo se va a usar la biblioteca. El extra `[dev]` añade pytest y ruff.

## Vista rápida

```python
import math
from numerica import bisection, simpson, rk4

raiz = bisection(lambda x: x**3 - x - 1, 1.0, 2.0)
print(raiz.root, raiz.iterations)

integral = simpson(math.sin, 0.0, math.pi, n=100)
print(integral.value)  # 2

solucion = rk4(lambda t, y: -y, (0.0, 1.0), 1.0, h=0.05)
print(solucion.y[-1])  # ≈ exp(-1)
```

## Métodos

| Tema | Funciones | Notas |
| --- | --- | --- |
| Raíces | `bisection`, `newton`, `secant`, `fixed_point` | `newton_raphson` es un alias de `newton`. Si no se pasa `df`, Newton usa una diferencia central. |
| Interpolación | `lagrange`, `newton_divided_difference`, `cubic_spline` | También como objetos: `LagrangeInterpolant`, `NewtonInterpolant`, `CubicSpline` (natural o anclado). |
| Derivación | `derivative` | Diferencias finitas hacia adelante, hacia atrás y centradas, de orden 1 y 2. |
| Integración | `trapezoidal`, `simpson`, `simpson_38`, `gauss_legendre`, `romberg` | Gauss-Legendre calcula nodos y pesos por el método de Golub-Welsch. |
| Sistemas lineales | `gaussian_elimination`, `lu_factor`, `lu_solve`, `jacobi`, `gauss_seidel` | Pivoteo parcial en los métodos directos. `b` puede traer varios lados derechos. |
| EDO | `euler`, `heun`, `rk4`, `rk45` | Problema `y' = f(t, y)`. `rk45` es el par embebido de Dormand–Prince 5(4), con paso adaptativo. |
| Extras | `polynomial_fit`, `least_squares`, `power_method`, `inverse_power_method` | Mínimos cuadrados por QR de Householder. La potencia inversa reutiliza la factorización LU. |

## Ejemplos de uso

### Raíces

La bisección exige cambio de signo. Newton, la secante y el punto fijo devuelven un `RootResult` con la aproximación, las iteraciones, el residuo y el historial.

```python
from numerica import bisection, fixed_point, newton, secant

f = lambda x: x * x - 2
print(bisection(f, 0, 2))
print(newton(f, 1.0, df=lambda x: 2 * x))
print(secant(f, 1.0, 2.0))
print(fixed_point(lambda x: 0.5 * (x + 2 / x), 1.0))
```

### Interpolación

Los nodos `x` tienen que ser estrictamente crecientes. El spline natural reproduce rectas; el spline anclado, con las derivadas exactas en los extremos, reproduce polinomios cúbicos.

```python
import numpy as np
from numerica import CubicSpline, lagrange, newton_divided_difference

x = np.linspace(0, 1, 4)
y = x**3
print(lagrange(x, y, 0.4))
print(newton_divided_difference(x, y, 0.4))

spline = CubicSpline(x, y, bc="clamped", dydx=(0.0, 3.0))
print(spline(0.4), spline.derivative(0.4))
```

Fuera del intervalo el spline lanza `ValueError`. Lagrange y Newton sí evalúan el polinomio fuera de los nodos.

### Derivación e integración

`step` en `derivative` es un paso relativo: el incremento real es `step * max(1, |x|)`.

```python
import math
from numerica import derivative, gauss_legendre, romberg, simpson

print(derivative(math.sin, 0.5))          # ≈ cos(0.5)
print(simpson(lambda x: x**3, 0, 1, n=2)) # 0.25, exacto
print(gauss_legendre(math.exp, 0, 1, n=8))
print(romberg(math.exp, 0, 1, tol=1e-12))
```

`simpson` es la regla 1/3 y pide un número par de subintervalos. `simpson_38` pide un múltiplo de 3. Si `a > b` la integral cambia de signo.

### Sistemas lineales

```python
import numpy as np
from numerica import gauss_seidel, gaussian_elimination, lu_factor

A = np.array([[10.0, -1.0], [-1.0, 10.0]])
b = A @ np.array([1.0, 2.0])
print(gaussian_elimination(A, b))

factores = lu_factor(A)
print(factores.solve(b))          # PA = LU
print(gauss_seidel(A, b, tol=1e-12))
```

Jacobi y Gauss-Seidel paran cuando `||Ax - b||_∞` baja de `tol`. Convergen en matrices estrictamente diagonal dominantes; la biblioteca no comprueba esa hipótesis.

### Ecuaciones diferenciales

Un dato inicial escalar produce una solución de forma `(n,)`. Un vector produce `(n, dimensión)`. El paso `h` es una magnitud positiva y el sentido lo marca `t_span`, así que la integración hacia atrás es el mismo código.

```python
import numpy as np
from numerica import euler, rk45

oscilador = lambda t, y: np.array([y[1], -y[0]])
sol = rk45(oscilador, (0.0, 1.0), np.array([1.0, 0.0]), rtol=1e-6)
print(sol.y[-1])   # ≈ [cos(1), -sin(1)]
print(euler(lambda t, y: y, (0.0, 1.0), 1.0, h=0.1))
```

En `rk45` un paso se acepta cuando el error embebido, medido en unidades de `atol + rtol * escala`, tiene norma RMS menor o igual que 1. `rejected_steps` cuenta los intentos descartados.

### Ajuste y autovalores

Los coeficientes salen de grado 0 hacia arriba: `c[0] + c[1] x + ...`.

```python
from numerica import polynomial_fit, power_method
import numpy as np

x = np.array([0.0, 1.0, 2.0, 3.0])
fit = polynomial_fit(x, 1 + 2 * x, 1)
print(fit.coefficients, fit(10))

par = power_method([[2.0, 1.0], [1.0, 2.0]])
print(par.eigenvalue, par.eigenvector)
```

`inverse_power_method(A, shift=mu)` busca el autovalor más cercano a `mu`. El desplazamiento no puede coincidir con un autovalor, porque `A - mu I` sería singular.

## Resultados y errores

Los métodos iterativos comparten la misma información de parada: `iterations`, `residual`, `tolerance` y `converged`. Si no se cumple la tolerancia se lanza `ConvergenceError`, que guarda esa misma información y la mejor aproximación en `value`. Con `raise_on_failure=False` el resultado vuelve igualmente y `converged` queda en `False`.

| Excepción | Cuándo aparece |
| --- | --- |
| `ConvergenceError` | Se agotan las iteraciones o el paso mínimo de `rk45`. |
| `SingularMatrixError` | Pivote nulo, matriz de rango deficiente o diagonal nula en Jacobi / Gauss-Seidel. |
| `DimensionError` | Formas incompatibles: matriz no cuadrada, `b` de otra longitud, `f` que no respeta la dimensión del sistema. |
| `ValueError` | Bracket sin cambio de signo, nodos desordenados, `n` impar en Simpson, tolerancia no positiva. |
| `NumericaError` | Paso de Newton o de la secante indefinido. Es la clase base del resto. |

## Ejemplos ejecutables

```bash
python examples/01_raices.py
python examples/02_interpolacion.py
python examples/03_integracion.py
python examples/04_sistemas.py
python examples/05_edo.py
python examples/06_ajuste.py
```

`examples/recorrido.ipynb` reúne un paseo corto por raíces, Romberg y RK45.

## Desarrollo

```bash
pip install -e ".[dev]"
ruff check src tests examples
ruff format --check src tests examples
pytest
```

La integración continua (`.github/workflows/ci.yml`) hace eso en Python 3.10–3.13 y además ejecuta los scripts de `examples/`.

## Alcance

La biblioteca cubre el núcleo de un primer curso. No incluye ecuaciones en derivadas parciales, optimización no lineal general ni el método QR para el espectro completo. Quedan como continuación natural el método QR, los métodos multipaso de Adams y una cuadratura adaptativa de tipo Gauss-Kronrod.

Los algoritmos son para doble precisión real. Matrices muy mal condicionadas, como las de Hilbert de orden alto, se resienten igual que en cualquier eliminación gaussiana sin escalado adicional.

## Licencia

MIT. Véase [LICENSE](LICENSE).
