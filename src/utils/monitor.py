"""Monitoreo en segundo plano del uso de CPU por núcleo y, opcionalmente, de la GPU.

Ejemplo::

    from src.utils.monitor import ResourceMonitor
    with ResourceMonitor(interval=0.5, gpu=False) as mon:
        work()
    df = mon.to_dataframe()   # columnas: t, cpu_0 ... cpu_n [, gpu_util, gpu_mem_mb]
"""

import shutil
import subprocess
import threading
import time

import pandas as pd
import psutil

_GPU_QUERY = [
    "nvidia-smi",
    "--query-gpu=utilization.gpu,memory.used",
    "--format=csv,noheader,nounits",
]


def _read_gpu() -> tuple[float, float]:
    """Devuelve ``(utilización %, memoria usada en MB)`` de la GPU 0 según ``nvidia-smi``."""
    out = subprocess.run(_GPU_QUERY, capture_output=True, text=True, check=False).stdout
    util, mem = out.strip().splitlines()[0].split(",")
    return float(util), float(mem)


class ResourceMonitor:
    """Context manager que muestrea recursos del sistema en un hilo aparte.

    Args:
        interval: Segundos entre muestras.
        gpu: Si es ``True`` y ``nvidia-smi`` existe, registra también la GPU.
    """

    def __init__(self, interval: float = 0.5, gpu: bool = False) -> None:
        self.interval = interval
        self.gpu = gpu and shutil.which("nvidia-smi") is not None
        self.samples: list[dict[str, float]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        start = time.perf_counter()
        psutil.cpu_percent(percpu=True)
        while not self._stop.wait(self.interval):
            row = {"t": time.perf_counter() - start}
            for i, pct in enumerate(psutil.cpu_percent(percpu=True)):
                row[f"cpu_{i}"] = pct
            if self.gpu:
                row["gpu_util"], row["gpu_mem_mb"] = _read_gpu()
            self.samples.append(row)

    def __enter__(self) -> "ResourceMonitor":
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join()

    def to_dataframe(self) -> pd.DataFrame:
        """Devuelve las muestras como DataFrame con una fila por instante."""
        return pd.DataFrame(self.samples)
