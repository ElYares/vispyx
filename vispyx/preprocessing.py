import cv2
import numpy as np

from vispyx.morphology_common import validate_grayscale_image


def _is_positive_pair(value):
    """Two positive integers, NumPy integers included and ``bool`` excluded."""
    try:
        pair = tuple(value)
    except TypeError:
        return False
    return len(pair) == 2 and all(
        isinstance(n, (int, np.integer)) and not isinstance(n, bool) and n > 0
        for n in pair
    )


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
    # Un cero en la grilla no llega a ser `cv2.error`: OpenCV divide por el
    # tamano de la celda y el proceso muere con SIGFPE, sin traceback.
    if not _is_positive_pair(tile_grid_size):
        raise ValueError("tile_grid_size must be two positive integers")
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(img)
