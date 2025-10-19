import cv2

import numpy as np

def apply_clahe(image, clip_limit=2.0, title_grid_size=(8,8)):
    """
    Aplica la ecualización adaptativa del histograma (CLAHE) para mejorar el contraste.
    
    :param image: Imagen en escala de grises (numpy array).
    :param clip_limit: Límite de recorte para CLAHE.
    :param tile_grid_size: Tamaño de grilla de la imagen.
    :return: Imagen procesada con mayor contraste.
    """

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=title_grid_size)
    return clahe.apply(image)

