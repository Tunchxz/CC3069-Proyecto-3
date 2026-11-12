"""Reporte del hardware y del software disponibles para la ejecución.

Uso desde la línea de comandos::

    python -m src.utils.system_info
"""

import os
import platform
import shutil
import subprocess
import sys

import psutil


def cpu_model() -> str:
    """Devuelve el nombre del modelo del CPU según ``/proc/cpuinfo`` o ``platform``."""
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "desconocido"


def nvidia_smi() -> str:
    """Devuelve la salida de ``nvidia-smi`` o un mensaje si no está disponible."""
    if shutil.which("nvidia-smi") is None:
        return "nvidia-smi no disponible"
    result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=False)
    return result.stdout or result.stderr


def tensorflow_info() -> dict[str, str]:
    """Devuelve la versión de TensorFlow, las GPU visibles y las versiones de CUDA y cuDNN."""
    import tensorflow as tf

    build = tf.sysconfig.get_build_info()
    gpus = tf.config.list_physical_devices("GPU")
    return {
        "TensorFlow": tf.__version__,
        "Compilado con CUDA": str(build.get("cuda_version", "n/a")),
        "Compilado con cuDNN": str(build.get("cudnn_version", "n/a")),
        "GPU visibles para TF": ", ".join(g.name for g in gpus) or "ninguna",
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "(sin definir)"),
    }


def system_info() -> dict[str, str]:
    """Reúne la información del sistema operativo, CPU, RAM y TensorFlow."""
    info = {
        "Sistema operativo": platform.platform(),
        "Python": sys.version.split()[0],
        "CPU": cpu_model(),
        "Núcleos físicos": str(psutil.cpu_count(logical=False)),
        "Núcleos lógicos": str(psutil.cpu_count(logical=True)),
        "RAM total (GB)": f"{psutil.virtual_memory().total / 1024**3:.1f}",
    }
    info.update(tensorflow_info())
    return info


def print_report(include_nvidia_smi: bool = True) -> None:
    """Imprime el reporte del sistema y, opcionalmente, la salida de ``nvidia-smi``."""
    for key, value in system_info().items():
        print(f"{key:<24}{value}")
    if include_nvidia_smi:
        print()
        print(nvidia_smi())


if __name__ == "__main__":
    print_report()
