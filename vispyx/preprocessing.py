import cv2

import numpy as np

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
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(image)
