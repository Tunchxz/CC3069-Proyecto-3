"""Utilidades para medir tiempos de ejecución y guardar resultados.

Ejemplo::

    from src.utils.timing import Timer, repeat, summarize
    with Timer() as t:
        work()
    print(t.elapsed)

    times = repeat(work, n=5)
    print(summarize(times))
"""

import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd


class Timer:
    """Context manager que mide el tiempo transcurrido con ``time.perf_counter``.

    Attributes:
        elapsed: Segundos transcurridos dentro del bloque ``with``.
    """

    def __init__(self) -> None:
        self.elapsed = 0.0
        self._start = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        self.elapsed = time.perf_counter() - self._start


def repeat(fn: Callable[[], object], n: int, verbose: bool = True) -> list[float]:
    """Ejecuta ``fn`` ``n`` veces y devuelve la duración de cada ejecución.

    Args:
        fn: Función sin argumentos a medir.
        n: Número de repeticiones.
        verbose: Si es ``True``, imprime la duración de cada repetición.

    Returns:
        Lista con los segundos de cada repetición.
    """
    times = []
    for i in range(n):
        with Timer() as t:
            fn()
        times.append(t.elapsed)
        if verbose:
            print(f"Repetición {i + 1}/{n}: {t.elapsed:.2f} s")
    return times


def summarize(times: list[float]) -> dict[str, float]:
    """Calcula promedio, desviación estándar, mínimo y máximo de una lista de tiempos.

    Args:
        times: Tiempos en segundos.

    Returns:
        Diccionario con las claves ``n``, ``mean``, ``std``, ``min`` y ``max``.
    """
    arr = np.asarray(times, dtype=float)
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def save_times(records: list[dict], path: Path) -> pd.DataFrame:
    """Guarda una lista de registros de tiempo en un archivo CSV.

    Args:
        records: Registros con columnas como ``phase``, ``device``, ``rep`` y ``seconds``.
        path: Ruta del archivo CSV de salida; se sobrescribe si existe.

    Returns:
        DataFrame con los registros guardados.
    """
    df = pd.DataFrame(records)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df
