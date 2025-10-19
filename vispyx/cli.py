import argparse
import os
import cv2
import matplotlib
matplotlib.use('TkAgg')  # Forzar backend seguro y visualizable
import matplotlib.pyplot as plt
import numpy as np
import time

from vispyx.preprocessing import apply_clahe
from vispyx.segmentation import segment_otsu
from vispyx.morphology import vpx_erode, vpx_dilate


def show_image(image, title='Resultado'):
    """
    Muestra una imagen usando matplotlib
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def run_clahe(image_path):
    img = cv2.imread(image_path, 0)
    if img is None:
        raise FileNotFoundError(f"No se encontró la imagen en {image_path}")
    
    result = apply_clahe(img)
    show_image(result, title='CLAHE')
    return result

def run_otsu(image_path):
    img = cv2.imread(image_path, 0)
    if img is None:
        raise FileNotFoundError(f"No se encontró la imagen en {image_path}")
    
    result = segment_otsu(img)
    show_image(result, title='Otsu')
    return result

def run_vpx_erode(image_path, kernel_size=3, iterations=1):
    img = cv2.imread(image_path, 0)
    binary = (img > 0).astype(np.uint8) * 255
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)

    #start = time.time()

    result = vpx_erode(binary, kernel, iterations)

    #end = time.time()

    show_image(result, title=f'vpx_erode (k={kernel_size}, i={iterations})')
    #print(f"TIempo de procesamiento vpx erode: {end - start:.4f} segundos")
    return result

def run_vpx_dilate(image_path, kernel_size=3, iterations=1):
    img = cv2.imread(image_path, 0)
    binary = (img > 0).astype(np.uint8) * 255
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    result = vpx_dilate(binary, kernel, iterations)
    show_image(result, title=f'vpx_dilate (k={kernel_size}, i={iterations})')
    return result    

def main():
    parser = argparse.ArgumentParser(description="CLI de procesamiento de imágenes con vispyx")
    parser.add_argument("method", choices=["clahe", "otsu", "vpx_erode","vpx_dilate"], help="Método de procesamiento")
    parser.add_argument("image_path", help="Ruta de la imagen a procesar")
    parser.add_argument("--output", "-o", help="Ruta para guardar imagen procesada (opcional)", default=None)
    parser.add_argument("--kernel-size", type=int, default=3, help="Tamaño del kernel")
    parser.add_argument("--iterations", type=int, default=1, help="Número de iteraciones")

    args = parser.parse_args()

    if args.method == "clahe":
        result = run_clahe(args.image_path)
    elif args.method == "otsu":
        result = run_otsu(args.image_path)
    elif args.method == "vpx_erode":
        result = run_vpx_erode(args.image_path, kernel_size=args.kernel_size, iterations=args.iterations)
    elif args.method == "vpx_dilate":
        result = run_vpx_dilate(args.image_path, kernel_size=args.kernel_size, iterations=args.iterations)


    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        cv2.imwrite(args.output, result)
        print(f"Imagen guardada en {args.output}")
    else:
        print("Imagen procesada y mostrada. No se guardó.")
