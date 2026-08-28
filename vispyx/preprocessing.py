import cv2
import numpy as np

from vispyx.morphology_common import validate_grayscale_image


def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8), title_grid_size=None):
    """
    Aplica la ecualización adaptativa del histograma (CLAHE) para mejorar el contraste.
    
    :param image: Imagen en escala de grises (numpy array).
    :param clip_limit: Límite de recorte para CLAHE.
    :param tile_grid_size: Tamaño de grilla de la imagen.
    :return: Imagen procesada con mayor contraste.
    """
    # Compatibilidad temporal: conservar soporte al typo histórico `title_grid_size`.
    if title_grid_size is not None:
        tile_grid_size = title_grid_size
    img = validate_grayscale_image(image)
    # CLAHE de OpenCV solo implementa CV_8UC1 y CV_16UC1. Sin este chequeo, un
    # float o un int con signo salen como `cv2.error` desde `clahe.cpp`.
    if img.dtype not in (np.uint8, np.uint16):
        raise ValueError("image must be uint8 or uint16")
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(img)
