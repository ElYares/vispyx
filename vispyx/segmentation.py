import cv2
import numpy as np
from skimage.filters import threshold_otsu

def segment_otsu(image):
    """
    Segmenta una imagen en escala de grises usando el método de umbral de Otsu.

    :param image: Imagen en escala de grises.
    :return: Imagen binaria segmentada.
    """
    thresh = threshold_otsu(image)
    binary = image > thresh
    return binary.astype(np.uint8) * 255