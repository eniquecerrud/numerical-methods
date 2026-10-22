"""Public surface and example scripts."""

from __future__ import annotations

import runpy
from pathlib import Path

import pytest

import numerica

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
EXAMPLE_SCRIPTS = (
    "01_raices.py",
    "02_interpolacion.py",
    "03_integracion.py",
    "04_sistemas.py",
    "05_edo.py",
    "06_ajuste.py",
)


def test_version_and_exports() -> None:
    assert numerica.__version__ == "0.1.0"
    assert numerica.__all__
    for name in numerica.__all__:
        assert hasattr(numerica, name)


@pytest.mark.parametrize("script", EXAMPLE_SCRIPTS)
def test_examples_run(script: str) -> None:
    runpy.run_path(str(EXAMPLES / script), run_name="__main__")
