"""Preprocesamiento secuencial de imágenes: lectura, redimensionado y normalización.

Ejemplo::

    from src.features.build_features import preprocess_sequential
    X = preprocess_sequential(paths)          # (n, 64, 64, 3) float32 en [0, 1]
"""

import time
from collections.abc import Sequence

import numpy as np
from PIL import Image

from src import config


def read_image(path: str) -> Image.Image:
    """Lee y decodifica un archivo JPEG como imagen RGB.

    Args:
        path: Ruta del archivo de imagen.

    Returns:
        Imagen PIL en modo RGB (200x200 en el dataset original).
    """
    with Image.open(path) as img:
        return img.convert("RGB")


def resize_image(img: Image.Image, size: int = config.IMG_SIZE) -> np.ndarray:
    """Redimensiona una imagen a ``size`` x ``size`` con interpolación bilineal.

    Args:
        img: Imagen PIL.
        size: Lado de la imagen de salida en píxeles.

    Returns:
        Arreglo ``uint8`` con forma ``(size, size, 3)``.
    """
    return np.asarray(img.resize((size, size), Image.BILINEAR), dtype=np.uint8)


def normalize_image(arr: np.ndarray) -> np.ndarray:
    """Escala los valores de los píxeles de ``[0, 255]`` a ``[0, 1]``.

    Args:
        arr: Arreglo ``uint8`` de la imagen.

    Returns:
        Arreglo ``float32`` con la misma forma.
    """
    return arr.astype(np.float32) / 255.0


def preprocess_sequential(paths: Sequence[str], size: int = config.IMG_SIZE) -> np.ndarray:
    """Preprocesa una lista de imágenes, una por una, en un solo núcleo del CPU.

    Args:
        paths: Rutas de las imágenes.
        size: Lado de la imagen de salida en píxeles.

    Returns:
        Arreglo ``float32`` con forma ``(len(paths), size, size, 3)``.
    """
    out = np.empty((len(paths), size, size, config.N_CHANNELS), dtype=np.float32)
    for i, path in enumerate(paths):
        out[i] = normalize_image(resize_image(read_image(path), size))
    return out


def profile_stages(paths: Sequence[str], size: int = config.IMG_SIZE) -> dict[str, float]:
    """Mide el tiempo acumulado de cada etapa del preprocesamiento secuencial.

    Args:
        paths: Rutas de las imágenes.
        size: Lado de la imagen de salida en píxeles.

    Returns:
        Diccionario con los segundos totales de ``read``, ``resize`` y ``normalize``.
    """
    totals = {"read": 0.0, "resize": 0.0, "normalize": 0.0}
    for path in paths:
        t0 = time.perf_counter()
        img = read_image(path)
        t1 = time.perf_counter()
        arr = resize_image(img, size)
        t2 = time.perf_counter()
        normalize_image(arr)
        t3 = time.perf_counter()
        totals["read"] += t1 - t0
        totals["resize"] += t2 - t1
        totals["normalize"] += t3 - t2
    return totals
