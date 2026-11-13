"""Kernel de CUDA escrito con Numba para redimensionar y normalizar imágenes.

Cada hilo calcula un píxel de salida (sus 3 canales) mediante interpolación
bilineal y escala el resultado a ``[0, 1]``. Los hilos se organizan así.

- Bloque de ``16 x 16`` hilos, que cubre una región de ``16 x 16`` píxeles.
- Grid de ``(ceil(64/16), ceil(64/16), N)`` bloques, es decir ``4 x 4``
  bloques por imagen y una capa del eje ``z`` por imagen.

Ejemplo::

    from src.features.cuda_kernels import resize_normalize_gpu
    X, times = resize_normalize_gpu(batch_uint8)   # batch: (N, 200, 200, 3)
"""

import math
import time

import numpy as np
from numba import cuda, float32

from src import config

BLOCK = (16, 16, 1)


@cuda.jit
def resize_normalize_kernel(src, dst):
    """Redimensiona ``src`` (N, H, W, 3) ``uint8`` a ``dst`` (N, h, w, 3) ``float32``."""
    x, y, n = cuda.grid(3)
    out_h = dst.shape[1]
    out_w = dst.shape[2]
    if n >= dst.shape[0] or y >= out_h or x >= out_w:
        return

    in_h = src.shape[1]
    in_w = src.shape[2]
    sy = max((float32(y) + float32(0.5)) * float32(in_h) / float32(out_h) - float32(0.5), float32(0.0))
    sx = max((float32(x) + float32(0.5)) * float32(in_w) / float32(out_w) - float32(0.5), float32(0.0))
    y0 = int(sy)
    x0 = int(sx)
    y1 = min(y0 + 1, in_h - 1)
    x1 = min(x0 + 1, in_w - 1)
    dy = sy - float32(y0)
    dx = sx - float32(x0)

    for c in range(dst.shape[3]):
        top = float32(src[n, y0, x0, c]) * (float32(1.0) - dx) + float32(src[n, y0, x1, c]) * dx
        bottom = float32(src[n, y1, x0, c]) * (float32(1.0) - dx) + float32(src[n, y1, x1, c]) * dx
        dst[n, y, x, c] = (top * (float32(1.0) - dy) + bottom * dy) / float32(255.0)


def grid_dims(n_images: int, size: int = config.IMG_SIZE, block: tuple = BLOCK) -> tuple[int, int, int]:
    """Calcula las dimensiones del grid para ``n_images`` imágenes de ``size`` x ``size``."""
    return (math.ceil(size / block[0]), math.ceil(size / block[1]), n_images)


def resize_normalize_gpu(
    batch: np.ndarray, size: int = config.IMG_SIZE, block: tuple = BLOCK
) -> tuple[np.ndarray, dict[str, float]]:
    """Ejecuta el kernel sobre un lote y mide cada etapa.

    Args:
        batch: Arreglo ``uint8`` con forma ``(N, H, W, 3)`` en la memoria del CPU.
        size: Lado de la imagen de salida en píxeles.
        block: Dimensiones del bloque de hilos.

    Returns:
        Tupla ``(X, tiempos)``. ``X`` es un arreglo ``float32`` con forma
        ``(N, size, size, 3)`` y ``tiempos`` contiene los segundos de ``h2d``
        (CPU a GPU), ``kernel`` y ``d2h`` (GPU a CPU).
    """
    times = {}
    t0 = time.perf_counter()
    d_src = cuda.to_device(np.ascontiguousarray(batch))
    d_dst = cuda.device_array((batch.shape[0], size, size, batch.shape[3]), dtype=np.float32)
    cuda.synchronize()
    t1 = time.perf_counter()
    resize_normalize_kernel[grid_dims(batch.shape[0], size, block), block](d_src, d_dst)
    cuda.synchronize()
    t2 = time.perf_counter()
    out = d_dst.copy_to_host()
    t3 = time.perf_counter()
    times["h2d"] = t1 - t0
    times["kernel"] = t2 - t1
    times["d2h"] = t3 - t2
    return out, times
