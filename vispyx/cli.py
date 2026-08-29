import argparse
import os

import cv2
import matplotlib
import numpy as np

from vispyx.kernels import (
    _validate_size,
    kernel_cross,
    kernel_diamond,
    kernel_disk,
    kernel_square,
)
from vispyx.morphology import (
    gray_blackhat,
    gray_close,
    gray_dilate,
    gray_erode,
    gray_gradient,
    gray_open,
    gray_tophat,
    vpx_blackhat,
    vpx_boundary,
    vpx_close,
    vpx_dilate,
    vpx_erode,
    vpx_gradient,
    vpx_hitmiss,
    vpx_open,
    vpx_reconstruct,
    vpx_skeletonize,
    vpx_thin,
    vpx_tophat,
)
from vispyx.preprocessing import apply_clahe
from vispyx.segmentation import segment_otsu
from vispyx.utils import read_grayscale

matplotlib.use("TkAgg")  # Forzar backend seguro y visualizable
import matplotlib.pyplot as plt


def show_image(image, title="Resultado"):
    """Muestra una imagen usando matplotlib."""
    plt.figure(figsize=(8, 6))
    plt.imshow(image, cmap="gray")
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


KERNEL_SHAPES = ["square", "cross", "diamond", "disk"]


# Pares hit/miss del catalogo de `--pattern`. Cada uno cumple por construccion
# las tres reglas de `validate_hitmiss_kernels`: misma forma, sin solapamiento y
# con al menos un elemento activo cada uno.
_ESQUINA_HIT = np.array([[0, 0, 0], [0, 1, 1], [0, 1, 1]], dtype=np.uint8)
_ESQUINA_MISS = np.array([[1, 1, 1], [1, 0, 0], [1, 0, 0]], dtype=np.uint8)

# El par base mira al noroeste; los otros tres salen rotandolo.
_ESQUINAS = {
    "corner-nw": 0,
    "corner-ne": 1,
    "corner-se": 2,
    "corner-sw": 3,
}

PATTERNS = {
    nombre: (np.rot90(_ESQUINA_HIT, -giro), np.rot90(_ESQUINA_MISS, -giro))
    for nombre, giro in _ESQUINAS.items()
}
PATTERNS["isolated"] = (
    np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.uint8),
    np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8),
)

# `corner` no es un par: combina las cuatro orientaciones. La composicion vive
# aca y no en `morphology_binary.py`, que sigue exponiendo la operacion pura.
PATTERN_NAMES = ["corner"] + sorted(PATTERNS)


METHODS = [
    "clahe",
    "otsu",
    "vpx_erode",
    "vpx_dilate",
    "vpx_open",
    "vpx_close",
    "vpx_gradient",
    "vpx_tophat",
    "vpx_blackhat",
    "vpx_boundary",
    "vpx_hitmiss",
    "vpx_reconstruct",
    "vpx_skeletonize",
    "vpx_thin",
    "gray_erode",
    "gray_dilate",
    "gray_open",
    "gray_close",
    "gray_gradient",
    "gray_tophat",
    "gray_blackhat",
]


def _build_kernel(kernel_size, kernel_shape="square"):
    if kernel_size <= 0:
        raise ValueError("--kernel-size debe ser un entero positivo")
    if kernel_shape == "square":
        return kernel_square(kernel_size)
    elif kernel_shape == "cross":
        return kernel_cross(kernel_size)
    elif kernel_shape == "diamond":
        return kernel_diamond(kernel_size)
    elif kernel_shape == "disk":
        # kernel_disk toma radio, no lado. La paridad se valida aparte: sin
        # esto, size=4 daria radio 2 y el mismo disco 5x5 que size=5, en
        # silencio, mientras las otras tres formas rechazan el 4.
        _validate_size(kernel_size)
        return kernel_disk(kernel_size // 2)
    else:
        raise ValueError(f"Forma de kernel no reconocida: {kernel_shape}")


def run_clahe(image_path, clip_limit=2.0, grid=8):
    img = read_grayscale(image_path)
    return apply_clahe(img, clip_limit=clip_limit, tile_grid_size=(grid, grid))


def run_otsu(image_path):
    img = read_grayscale(image_path)
    return segment_otsu(img)


def run_vpx_hitmiss(image_path, pattern):
    """`vpx_hitmiss` no entra en `_run_binary_method`: toma dos kernels y no toma
    `iterations`. Por eso tiene su propia `run_*`, como `reconstruct` y `thin`.
    """
    img = read_grayscale(image_path)
    binary = (img > 0).astype(np.uint8) * 255
    if pattern == "corner":
        resultado = np.zeros_like(binary)
        for nombre in _ESQUINAS:
            hit, miss = PATTERNS[nombre]
            resultado = np.maximum(resultado, vpx_hitmiss(binary, hit, miss))
        return resultado
    if pattern not in PATTERNS:
        raise ValueError(f"Patron no reconocido: {pattern}")
    hit, miss = PATTERNS[pattern]
    return vpx_hitmiss(binary, hit, miss)


def run_vpx_reconstruct(marker_path, mask_path, kernel_size=3, max_iterations=None, kernel_shape="square"):
    marker = read_grayscale(marker_path)
    mask = read_grayscale(mask_path)
    marker = (marker > 0).astype(np.uint8) * 255
    mask = (mask > 0).astype(np.uint8) * 255
    kernel = _build_kernel(kernel_size, kernel_shape)
    return vpx_reconstruct(marker, mask, kernel=kernel, max_iterations=max_iterations)


def run_vpx_skeletonize(image_path, max_iterations=None):
    img = read_grayscale(image_path)
    binary = (img > 0).astype(np.uint8) * 255
    return vpx_skeletonize(binary, max_iterations=max_iterations)


def run_vpx_thin(image_path, iterations=1):
    img = read_grayscale(image_path)
    binary = (img > 0).astype(np.uint8) * 255
    return vpx_thin(binary, iterations=iterations)


def _run_binary_method(image_path, method, kernel_size=3, iterations=1, kernel_shape="square"):
    """Gemelo binario de ``_run_grayscale_method``: binariza antes de operar.

    Las ``vpx_*`` binarizan igual por dentro, pero el CLI lo hace explicito
    porque lee de disco: un PNG de grises entraria como mascara casi solida sin
    que nada avise. Es la unica diferencia con la version gris.
    """
    img = read_grayscale(image_path)
    binary = (img > 0).astype(np.uint8) * 255
    kernel = _build_kernel(kernel_size, kernel_shape)
    return method(binary, kernel, iterations)


def _run_grayscale_method(image_path, method, kernel_size=3, iterations=1, kernel_shape="square"):
    img = read_grayscale(image_path)
    kernel = _build_kernel(kernel_size, kernel_shape)
    return method(img, kernel, iterations)


def main():
    parser = argparse.ArgumentParser(description="CLI de procesamiento de imágenes con vispyx")
    parser.add_argument(
        "method",
        choices=METHODS,
        help="Método de procesamiento",
    )

    parser.add_argument("image_path", help="Ruta de la imagen a procesar")
    parser.add_argument("--mask", "--mask-path", dest="mask_path", help="Ruta de la mascara para reconstruccion binaria")
    parser.add_argument("--output", "-o", help="Ruta para guardar imagen procesada (opcional)", default=None)
    parser.add_argument("--show", action="store_true", help="Mostrar imagen procesada en pantalla")
    parser.add_argument("--clip", type=float, default=2.0, help="Límite de clipping para CLAHE")
    parser.add_argument("--grid", type=int, default=8, help="Tamaño de cuadrícula para CLAHE")
    parser.add_argument("--kernel-size", type=int, default=3, help="Tamaño del kernel (3, 5, 7...)")
    parser.add_argument("--kernel", dest="kernel_size", type=int, help="Alias de --kernel-size")
    parser.add_argument(
        "--kernel-shape",
        choices=KERNEL_SHAPES,
        default="square",
        help="Forma del elemento estructurante (para disk, el radio es --kernel-size // 2)",
    )
    parser.add_argument(
        "--pattern",
        choices=PATTERN_NAMES,
        default=None,
        help="Patron a detectar con vpx_hitmiss. `corner` combina las cuatro orientaciones",
    )
    parser.add_argument("--iterations", type=int, default=1, help="Número de iteraciones")
    parser.add_argument("--max-iterations", type=int, default=None, help="Máximo de iteraciones para reconstruccion binaria")

    args = parser.parse_args()

    if args.method == "clahe":
        result = run_clahe(args.image_path, clip_limit=args.clip, grid=args.grid)
    elif args.method == "otsu":
        result = run_otsu(args.image_path)
    elif args.method == "vpx_erode":
        result = _run_binary_method(args.image_path, vpx_erode, kernel_size=args.kernel_size, iterations=args.iterations, kernel_shape=args.kernel_shape)
    elif args.method == "vpx_dilate":
        result = _run_binary_method(args.image_path, vpx_dilate, kernel_size=args.kernel_size, iterations=args.iterations, kernel_shape=args.kernel_shape)
    elif args.method == "vpx_open":
        result = _run_binary_method(args.image_path, vpx_open, kernel_size=args.kernel_size, iterations=args.iterations, kernel_shape=args.kernel_shape)
    elif args.method == "vpx_close":
        result = _run_binary_method(args.image_path, vpx_close, kernel_size=args.kernel_size, iterations=args.iterations, kernel_shape=args.kernel_shape)
    elif args.method == "vpx_gradient":
        result = _run_binary_method(args.image_path, vpx_gradient, kernel_size=args.kernel_size, iterations=args.iterations, kernel_shape=args.kernel_shape)
    elif args.method == "vpx_tophat":
        result = _run_binary_method(args.image_path, vpx_tophat, kernel_size=args.kernel_size, iterations=args.iterations, kernel_shape=args.kernel_shape)
    elif args.method == "vpx_blackhat":
        result = _run_binary_method(args.image_path, vpx_blackhat, kernel_size=args.kernel_size, iterations=args.iterations, kernel_shape=args.kernel_shape)
    elif args.method == "vpx_boundary":
        result = _run_binary_method(args.image_path, vpx_boundary, kernel_size=args.kernel_size, iterations=args.iterations, kernel_shape=args.kernel_shape)
    elif args.method == "vpx_hitmiss":
        if not args.pattern:
            parser.error("--pattern es obligatorio para vpx_hitmiss")
        result = run_vpx_hitmiss(args.image_path, args.pattern)
    elif args.method == "vpx_reconstruct":
        if not args.mask_path:
            parser.error("--mask es obligatorio para vpx_reconstruct")
        result = run_vpx_reconstruct(
            args.image_path,
            args.mask_path,
            kernel_size=args.kernel_size,
            max_iterations=args.max_iterations,
            kernel_shape=args.kernel_shape,
        )
    elif args.method == "vpx_skeletonize":
        result = run_vpx_skeletonize(args.image_path, max_iterations=args.max_iterations)
    elif args.method == "vpx_thin":
        result = run_vpx_thin(args.image_path, iterations=args.iterations)
    elif args.method == "gray_erode":
        result = _run_grayscale_method(args.image_path, gray_erode, kernel_size=args.kernel_size, iterations=args.iterations, kernel_shape=args.kernel_shape)
    elif args.method == "gray_dilate":
        result = _run_grayscale_method(args.image_path, gray_dilate, kernel_size=args.kernel_size, iterations=args.iterations, kernel_shape=args.kernel_shape)
    elif args.method == "gray_open":
        result = _run_grayscale_method(args.image_path, gray_open, kernel_size=args.kernel_size, iterations=args.iterations, kernel_shape=args.kernel_shape)
    elif args.method == "gray_close":
        result = _run_grayscale_method(args.image_path, gray_close, kernel_size=args.kernel_size, iterations=args.iterations, kernel_shape=args.kernel_shape)
    elif args.method == "gray_gradient":
        result = _run_grayscale_method(args.image_path, gray_gradient, kernel_size=args.kernel_size, iterations=args.iterations, kernel_shape=args.kernel_shape)
    elif args.method == "gray_tophat":
        result = _run_grayscale_method(args.image_path, gray_tophat, kernel_size=args.kernel_size, iterations=args.iterations, kernel_shape=args.kernel_shape)
    elif args.method == "gray_blackhat":
        result = _run_grayscale_method(args.image_path, gray_blackhat, kernel_size=args.kernel_size, iterations=args.iterations, kernel_shape=args.kernel_shape)
    else:
        raise ValueError(f"Método no reconocido: {args.method}")

    if args.output:
        output_dir = os.path.dirname(args.output)
        if output_dir:
            try:
                os.makedirs(output_dir, exist_ok=True)
            except OSError as error:
                # Falla antes de escribir nada: el padre no deja crear, o no es
                # un directorio. Sin atrapar, sale como traceback de os.
                parser.error(
                    f"No se pudo crear el directorio {output_dir}: {error.strerror}"
                )
        try:
            guardada = cv2.imwrite(args.output, result)
        except cv2.error:
            # OpenCV elige el codec por la extension y lanza si no tiene ninguno.
            parser.error(
                f"No se pudo guardar la imagen en {args.output}: "
                "OpenCV no reconoce la extension"
            )
        if not guardada:
            # Devuelve False, sin lanzar, cuando no logra abrir el archivo:
            # permisos, la ruta es un directorio, el directorio no existe.
            parser.error(
                f"No se pudo guardar la imagen en {args.output}: "
                "no se pudo abrir el archivo para escritura"
            )
        print(f"Imagen guardada en: {args.output}")

    if args.show:
        show_image(result, title=args.method)

    if not args.output:
        print("Imagen procesada. No se guardó.")
