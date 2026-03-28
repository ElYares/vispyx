import argparse
import os

import cv2
import matplotlib
import numpy as np

from vispyx.morphology import (
    gray_blackhat,
    gray_close,
    gray_dilate,
    gray_erode,
    gray_gradient,
    gray_open,
    gray_tophat,
    vpx_close,
    vpx_dilate,
    vpx_erode,
    vpx_gradient,
    vpx_open,
    vpx_reconstruct,
    vpx_skeletonize,
    vpx_thin,
)
from vispyx.preprocessing import apply_clahe
from vispyx.segmentation import segment_otsu

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


def _read_grayscale(image_path):
    img = cv2.imread(image_path, 0)
    if img is None:
        raise FileNotFoundError(f"No se encontró la imagen en {image_path}")
    return img


def _build_kernel(kernel_size):
    if kernel_size <= 0:
        raise ValueError("--kernel-size debe ser un entero positivo")
    return np.ones((kernel_size, kernel_size), dtype=np.uint8)


def run_clahe(image_path, clip_limit=2.0, grid=8):
    img = _read_grayscale(image_path)
    return apply_clahe(img, clip_limit=clip_limit, tile_grid_size=(grid, grid))


def run_otsu(image_path):
    img = _read_grayscale(image_path)
    return segment_otsu(img)


def run_vpx_erode(image_path, kernel_size=3, iterations=1):
    img = _read_grayscale(image_path)
    binary = (img > 0).astype(np.uint8) * 255
    kernel = _build_kernel(kernel_size)
    return vpx_erode(binary, kernel, iterations)


def run_vpx_dilate(image_path, kernel_size=3, iterations=1):
    img = _read_grayscale(image_path)
    binary = (img > 0).astype(np.uint8) * 255
    kernel = _build_kernel(kernel_size)
    return vpx_dilate(binary, kernel, iterations)


def run_vpx_open(image_path, kernel_size=3, iterations=1):
    img = _read_grayscale(image_path)
    binary = (img > 0).astype(np.uint8) * 255
    kernel = _build_kernel(kernel_size)
    return vpx_open(binary, kernel, iterations)


def run_vpx_close(image_path, kernel_size=3, iterations=1):
    img = _read_grayscale(image_path)
    binary = (img > 0).astype(np.uint8) * 255
    kernel = _build_kernel(kernel_size)
    return vpx_close(binary, kernel, iterations)


def run_vpx_gradient(image_path, kernel_size=3, iterations=1):
    img = _read_grayscale(image_path)
    binary = (img > 0).astype(np.uint8) * 255
    kernel = _build_kernel(kernel_size)
    return vpx_gradient(binary, kernel, iterations)


def run_vpx_reconstruct(marker_path, mask_path, kernel_size=3, max_iterations=None):
    marker = _read_grayscale(marker_path)
    mask = _read_grayscale(mask_path)
    marker = (marker > 0).astype(np.uint8) * 255
    mask = (mask > 0).astype(np.uint8) * 255
    kernel = _build_kernel(kernel_size)
    return vpx_reconstruct(marker, mask, kernel=kernel, max_iterations=max_iterations)


def run_vpx_skeletonize(image_path, max_iterations=None):
    img = _read_grayscale(image_path)
    binary = (img > 0).astype(np.uint8) * 255
    return vpx_skeletonize(binary, max_iterations=max_iterations)


def run_vpx_thin(image_path, iterations=1):
    img = _read_grayscale(image_path)
    binary = (img > 0).astype(np.uint8) * 255
    return vpx_thin(binary, iterations=iterations)


def _run_grayscale_method(image_path, method, kernel_size=3, iterations=1):
    img = _read_grayscale(image_path)
    kernel = _build_kernel(kernel_size)
    return method(img, kernel, iterations)


def main():
    methods = [
        "clahe",
        "otsu",
        "vpx_erode",
        "vpx_dilate",
        "vpx_open",
        "vpx_close",
        "vpx_gradient",
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
    parser = argparse.ArgumentParser(description="CLI de procesamiento de imágenes con vispyx")
    parser.add_argument(
        "method",
        choices=methods,
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
    parser.add_argument("--iterations", type=int, default=1, help="Número de iteraciones")
    parser.add_argument("--max-iterations", type=int, default=None, help="Máximo de iteraciones para reconstruccion binaria")

    args = parser.parse_args()

    if args.method == "clahe":
        result = run_clahe(args.image_path, clip_limit=args.clip, grid=args.grid)
    elif args.method == "otsu":
        result = run_otsu(args.image_path)
    elif args.method == "vpx_erode":
        result = run_vpx_erode(args.image_path, kernel_size=args.kernel_size, iterations=args.iterations)
    elif args.method == "vpx_dilate":
        result = run_vpx_dilate(args.image_path, kernel_size=args.kernel_size, iterations=args.iterations)
    elif args.method == "vpx_open":
        result = run_vpx_open(args.image_path, kernel_size=args.kernel_size, iterations=args.iterations)
    elif args.method == "vpx_close":
        result = run_vpx_close(args.image_path, kernel_size=args.kernel_size, iterations=args.iterations)
    elif args.method == "vpx_gradient":
        result = run_vpx_gradient(args.image_path, kernel_size=args.kernel_size, iterations=args.iterations)
    elif args.method == "vpx_reconstruct":
        if not args.mask_path:
            parser.error("--mask es obligatorio para vpx_reconstruct")
        result = run_vpx_reconstruct(
            args.image_path,
            args.mask_path,
            kernel_size=args.kernel_size,
            max_iterations=args.max_iterations,
        )
    elif args.method == "vpx_skeletonize":
        result = run_vpx_skeletonize(args.image_path, max_iterations=args.max_iterations)
    elif args.method == "vpx_thin":
        result = run_vpx_thin(args.image_path, iterations=args.iterations)
    elif args.method == "gray_erode":
        result = _run_grayscale_method(args.image_path, gray_erode, kernel_size=args.kernel_size, iterations=args.iterations)
    elif args.method == "gray_dilate":
        result = _run_grayscale_method(args.image_path, gray_dilate, kernel_size=args.kernel_size, iterations=args.iterations)
    elif args.method == "gray_open":
        result = _run_grayscale_method(args.image_path, gray_open, kernel_size=args.kernel_size, iterations=args.iterations)
    elif args.method == "gray_close":
        result = _run_grayscale_method(args.image_path, gray_close, kernel_size=args.kernel_size, iterations=args.iterations)
    elif args.method == "gray_gradient":
        result = _run_grayscale_method(args.image_path, gray_gradient, kernel_size=args.kernel_size, iterations=args.iterations)
    elif args.method == "gray_tophat":
        result = _run_grayscale_method(args.image_path, gray_tophat, kernel_size=args.kernel_size, iterations=args.iterations)
    elif args.method == "gray_blackhat":
        result = _run_grayscale_method(args.image_path, gray_blackhat, kernel_size=args.kernel_size, iterations=args.iterations)
    else:
        raise ValueError(f"Método no reconocido: {args.method}")

    if args.output:
        output_dir = os.path.dirname(args.output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        cv2.imwrite(args.output, result)
        print(f"Imagen guardada en: {args.output}")

    if args.show:
        show_image(result, title=args.method)

    if not args.output:
        print("Imagen procesada. No se guardó.")
