"""Compara el motor de Python contra el backend en Rust.

    python native/bench.py

No es un test: no falla, solo mide. La paridad la garantiza
``test/test_backend_parity.py``.
"""

import time

import numpy as np

from vispyx import (
    _backend,
    gray_erode,
    gray_open,
    kernel_square,
    vpx_erode,
    vpx_open,
    vpx_skeletonize,
)

CASES = (
    ("vpx_erode   3x3  x1", vpx_erode, 3, 1),
    ("vpx_erode   5x5  x1", vpx_erode, 5, 1),
    ("vpx_open    3x3  x2", vpx_open, 3, 2),
    ("gray_erode  3x3  x1", gray_erode, 3, 1),
    ("gray_erode  7x7  x1", gray_erode, 7, 1),
    ("gray_erode 15x15 x1", gray_erode, 15, 1),
    ("gray_open   3x3  x1", gray_open, 3, 1),
)

SIZES = (64, 128, 256)

# Las `gray_*` no binarizan, asi que necesitan una imagen de grises de verdad.
GRAYSCALE_PREFIX = "gray_"


def timed(operation, image, kernel, iterations):
    start = time.perf_counter()
    result = operation(image, kernel, iterations)
    return time.perf_counter() - start, result


def main():
    rng = np.random.default_rng(0)

    print(f"{'caso':<20} {'tamaño':>9} {'python':>10} {'rust':>10} {'speedup':>9}")
    print("-" * 62)
    # El speedup cae con el tamano del kernel, y no es un defecto del port: en
    # Python el costo es por pixel (el overhead de numpy por ventana domina) y
    # en Rust es por celda activa. Ver docs/native_backend.md.

    for label, operation, kernel_size, iterations in CASES:
        kernel = kernel_square(kernel_size)
        for size in SIZES:
            if label.startswith(GRAYSCALE_PREFIX):
                image = rng.integers(0, 256, (size, size), dtype=np.uint8)
            else:
                image = (rng.random((size, size)) > 0.5).astype(np.uint8) * 255

            with _backend.override("python"):
                python_seconds, expected = timed(operation, image, kernel, iterations)
            with _backend.override("rust"):
                rust_seconds, actual = timed(operation, image, kernel, iterations)

            assert np.array_equal(expected, actual), "los backends divergieron"

            print(
                f"{label:<20} {size:>4}x{size:<4} "
                f"{python_seconds:>9.4f}s {rust_seconds:>9.4f}s "
                f"{python_seconds / rust_seconds:>8.0f}x"
            )

    # Zhang-Suen no toma kernel y escala con pixeles x iteraciones, y las
    # iteraciones con el grosor de los objetos: sobre ruido converge en dos o
    # tres pasadas y no mide nada. Un disco grueso es el caso que duele.
    for size in (128, 256, 512):
        yy, xx = np.mgrid[:size, :size]
        radius = size * 0.4
        image = (((yy - size / 2) ** 2 + (xx - size / 2) ** 2) < radius**2).astype(np.uint8) * 255

        with _backend.override("python"):
            start = time.perf_counter()
            expected = vpx_skeletonize(image)
            python_seconds = time.perf_counter() - start
        with _backend.override("rust"):
            start = time.perf_counter()
            actual = vpx_skeletonize(image)
            rust_seconds = time.perf_counter() - start

        assert np.array_equal(expected, actual), "los backends divergieron"

        print(
            f"{'vpx_skeletonize':<20} {size:>4}x{size:<4} "
            f"{python_seconds:>9.4f}s {rust_seconds:>9.4f}s "
            f"{python_seconds / rust_seconds:>8.0f}x"
        )


if __name__ == "__main__":
    main()
