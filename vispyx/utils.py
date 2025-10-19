import cv2
import numpy as np
import matplotlib.pyplot as plt

def read_grayscale(path):
    """
    Lee una imagen y la convierte a escala de grises.
    """
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)

def show_image(image, title='Imagen', cmap='gray'):
    """
    Muestra una imagen usando matplotlib.
    """
    plt.imshow(image, cmap=cmap)
    plt.title(title)
    plt.axis('off')
    plt.show()