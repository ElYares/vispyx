import os

import cv2
import matplotlib.pyplot as plt


def read_grayscale(path):
    """
    Lee una imagen y la convierte a escala de grises.

    `cv2.imread` no lanza excepción cuando falla: devuelve `None` tanto si el
    archivo no existe como si existe pero no se puede decodificar. Ese `None`
    silencioso viaja hasta reventar tres funciones después, con un error que no
    dice nada del problema real, así que aquí se traduce a una excepción que
    distingue los dos casos.

    :param path: Ruta de la imagen.
    :raises FileNotFoundError: si no hay ningún archivo en `path`.
    :raises ValueError: si el archivo existe pero no es una imagen legible.
    :return: Imagen en escala de grises (numpy array, uint8).
    """
    # La existencia se comprueba antes de llamar a OpenCV: sobre una ruta que no
    # existe, `cv2.imread` escribe un WARN en stderr que ensuciaría la salida del
    # CLI justo antes de nuestro propio mensaje.
    if not os.path.exists(path):
        raise FileNotFoundError(f"No se encontró la imagen en {path}")

    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"No se pudo decodificar la imagen en {path}")
    return image


def show_image(image, title='Imagen', cmap='gray', figsize=None):
    """
    Muestra una imagen usando matplotlib.

    `figsize` crea una figura propia de ese tamaño y ajusta los márgenes con
    `tight_layout`. Omitirlo dibuja sobre la figura activa, que es el
    comportamiento histórico y el que espera un notebook. El CLI la pasa porque
    abre una ventana suelta y necesita un tamaño razonable.
    """
    if figsize is not None:
        plt.figure(figsize=figsize)
    plt.imshow(image, cmap=cmap)
    plt.title(title)
    plt.axis('off')
    if figsize is not None:
        plt.tight_layout()
    plt.show()
