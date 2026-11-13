"""Preprocesamiento de imágenes en varios núcleos del CPU con un pool de procesos.

Cada proceso del pool recibe un bloque de rutas, lee y redimensiona sus
imágenes, y devuelve un arreglo ``uint8``. El proceso principal une los
bloques y normaliza el resultado en una sola operación vectorizada.

Ejemplo::

    from src.features.parallel_features import make_pool, preprocess_parallel
    with make_pool(8) as pool:
        X = preprocess_parallel(paths, pool, n_workers=8)
"""

import multiprocessing as mp
from collections.abc import Sequence
from multiprocessing.pool import Pool

import numpy as np

from src import config
from src.features.build_features import read_image, resize_image

CHUNKS_PER_WORKER = 4


def make_pool(n_workers: int) -> Pool:
    """Crea un pool de procesos con el método de inicio ``spawn``.

    ``spawn`` inicia cada proceso desde cero, lo que evita heredar el estado
    de TensorFlow o CUDA del proceso principal.

    Args:
        n_workers: Número de procesos.

    Returns:
        Pool de ``multiprocessing`` listo para usarse como context manager.
    """
    return mp.get_context("spawn").Pool(processes=n_workers)


def split_chunks(paths: Sequence[str], n_chunks: int) -> list[list[str]]:
    """Divide ``paths`` en ``n_chunks`` bloques contiguos de tamaño similar."""
    bounds = np.linspace(0, len(paths), n_chunks + 1, dtype=int)
    return [list(paths[a:b]) for a, b in zip(bounds[:-1], bounds[1:]) if b > a]


def resize_chunk(paths: Sequence[str], size: int = config.IMG_SIZE) -> np.ndarray:
    """Lee y redimensiona un bloque de imágenes.

    Args:
        paths: Rutas de las imágenes del bloque.
        size: Lado de la imagen de salida en píxeles.

    Returns:
        Arreglo ``uint8`` con forma ``(len(paths), size, size, 3)``.
    """
    out = np.empty((len(paths), size, size, config.N_CHANNELS), dtype=np.uint8)
    for i, path in enumerate(paths):
        out[i] = resize_image(read_image(path), size)
    return out


def decode_chunk(paths: Sequence[str]) -> np.ndarray:
    """Lee y decodifica un bloque de imágenes sin redimensionarlas.

    Args:
        paths: Rutas de imágenes con el mismo tamaño.

    Returns:
        Arreglo ``uint8`` con forma ``(len(paths), alto, ancho, 3)``.
    """
    return np.stack([np.asarray(read_image(p), dtype=np.uint8) for p in paths])


def preprocess_parallel(paths: Sequence[str], pool: Pool, n_workers: int) -> np.ndarray:
    """Preprocesa las imágenes repartiendo bloques entre los procesos del pool.

    Args:
        paths: Rutas de las imágenes.
        pool: Pool creado con ``make_pool``.
        n_workers: Número de procesos del pool.

    Returns:
        Arreglo ``float32`` con forma ``(len(paths), 64, 64, 3)`` y valores en
        ``[0, 1]``, idéntico al de ``preprocess_sequential``.
    """
    chunks = split_chunks(paths, n_workers * CHUNKS_PER_WORKER)
    resized = np.concatenate(pool.map(resize_chunk, chunks))
    return resized.astype(np.float32) / 255.0


def decode_parallel(paths: Sequence[str], pool: Pool, n_workers: int) -> np.ndarray:
    """Decodifica las imágenes en paralelo sin redimensionarlas.

    Args:
        paths: Rutas de imágenes con el mismo tamaño.
        pool: Pool creado con ``make_pool``.
        n_workers: Número de procesos del pool.

    Returns:
        Arreglo ``uint8`` con forma ``(len(paths), alto, ancho, 3)``.
    """
    chunks = split_chunks(paths, n_workers * CHUNKS_PER_WORKER)
    return np.concatenate(pool.map(decode_chunk, chunks))
