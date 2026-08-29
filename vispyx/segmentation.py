import numpy as np
from skimage.filters import threshold_otsu

from vispyx.morphology_common import validate_grayscale_image


def segment_otsu(image):
    """
    Segmenta una imagen en escala de grises usando el método de umbral de Otsu.

    Es el puente entre los dos dominios de valores del paquete: recibe una
    imagen en grises y devuelve la máscara `{0, 255}` que las `vpx_*` esperan.

    :param image: Imagen en escala de grises (array 2D numérico).
    :return: Imagen binaria segmentada, `uint8` con valores en `{0, 255}`.
    :raises ValueError: si la imagen no es 2D o no contiene valores numéricos.
    """
    img = validate_grayscale_image(image)
    thresh = threshold_otsu(img)
    binary = img > thresh
    return binary.astype(np.uint8) * 255
