import argparse
import os
import cv2
import matplotlib
matplotlib.use('TkAgg')  # Forzar backend seguro y visualizable
import matplotlib.pyplot as plt

from vispyx.preprocessing import apply_clahe
from vispyx.segmentation import segment_otsu

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

def main():
    parser = argparse.ArgumentParser(description="CLI de procesamiento de imágenes con vispyx")
    parser.add_argument("method", choices=["clahe", "otsu"], help="Método de procesamiento")
    parser.add_argument("image_path", help="Ruta de la imagen a procesar")
    parser.add_argument("--output", "-o", help="Ruta para guardar imagen procesada (opcional)", default=None)

    args = parser.parse_args()

    if args.method == "clahe":
        result = run_clahe(args.image_path)
    elif args.method == "otsu":
        result = run_otsu(args.image_path)

    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        cv2.imwrite(args.output, result)
        print(f"Imagen guardada en {args.output}")
    else:
        print("Imagen procesada y mostrada. No se guardó.")
