import argparse
import os

import cv2
import matplotlib
import numpy as np

from vispyx.morphology import vpx_close, vpx_dilate, vpx_erode, vpx_gradient, vpx_open
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


def main():
    parser = argparse.ArgumentParser(description="CLI de procesamiento de imágenes con vispyx")
    parser.add_argument(
        "method",
        choices=["clahe", "otsu", "vpx_erode", "vpx_dilate", "vpx_open", "vpx_close", "vpx_gradient"],
        help="Método de procesamiento",
    )

    parser.add_argument("image_path", help="Ruta de la imagen a procesar")
    parser.add_argument("--output", "-o", help="Ruta para guardar imagen procesada (opcional)", default=None)
    parser.add_argument("--show", action="store_true", help="Mostrar imagen procesada en pantalla")
    parser.add_argument("--clip", type=float, default=2.0, help="Límite de clipping para CLAHE")
    parser.add_argument("--grid", type=int, default=8, help="Tamaño de cuadrícula para CLAHE")
    parser.add_argument("--kernel-size", type=int, default=3, help="Tamaño del kernel (3, 5, 7...)")
    parser.add_argument("--kernel", dest="kernel_size", type=int, help="Alias de --kernel-size")
    parser.add_argument("--iterations", type=int, default=1, help="Número de iteraciones")

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
