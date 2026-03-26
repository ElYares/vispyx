import numpy as np
from scipy.ndimage import (
    binary_erosion,
    binary_dilation,
    binary_opening,
    binary_closing
)

class MorphologicalProcessor:
    def __init__(self, kernel_size=3, iterations=1):
        self.kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        self.iterations = iterations

    def binarize(self, image):
        return (image > 0).astype(np.uint8)

    def erode(self, image):
        binary = self.binarize(image)
        result = binary_erosion(binary, structure=self.kernel, iterations=self.iterations)
        return result.astype(np.uint8) * 255

    def dilate(self, image):
        binary = self.binarize(image)
        result = binary_dilation(binary, structure=self.kernel, iterations=self.iterations)
        return result.astype(np.uint8) * 255

    def open(self, image):
        binary = self.binarize(image)
        result = binary_opening(binary, structure=self.kernel, iterations=self.iterations)
        return result.astype(np.uint8) * 255

    def close(self, image):
        binary = self.binarize(image)
        result = binary_closing(binary, structure=self.kernel, iterations=self.iterations)
        return result.astype(np.uint8) * 255

    def gradient(self, image):
        dilated = self.binarize(self.dilate(image))
        eroded = self.binarize(self.erode(image))
        result = dilated - eroded
        return result.astype(np.uint8) * 255
