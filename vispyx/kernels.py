"""Structuring element generators for morphological operations."""

import numpy as np


def _validate_size(size):
    """Validate odd kernel sizes used to define a central anchor."""
    if not isinstance(size, int) or size <= 0:
        raise ValueError("size must be a positive integer")
    if size % 2 == 0:
        raise ValueError("size must be odd")


def _validate_radius(radius):
    """Validate disk radii for structuring elements."""
    if not isinstance(radius, int) or radius < 0:
        raise ValueError("radius must be a non-negative integer")


def kernel_square(size):
    """Return an odd-sized square structuring element of ones."""
    _validate_size(size)
    return np.ones((size, size), dtype=np.uint8)


def kernel_cross(size):
    """Return an odd-sized cross structuring element."""
    _validate_size(size)
    kernel = np.zeros((size, size), dtype=np.uint8)
    center = size // 2
    kernel[center, :] = 1
    kernel[:, center] = 1
    return kernel


def kernel_diamond(size):
    """Return an odd-sized diamond structuring element using Manhattan distance."""
    _validate_size(size)
    center = size // 2
    kernel = np.zeros((size, size), dtype=np.uint8)
    for i in range(size):
        for j in range(size):
            if abs(i - center) + abs(j - center) <= center:
                kernel[i, j] = 1
    return kernel


def kernel_disk(radius):
    """Return a disk-like structuring element using Euclidean distance."""
    _validate_radius(radius)
    size = radius * 2 + 1
    center = radius
    kernel = np.zeros((size, size), dtype=np.uint8)
    for i in range(size):
        for j in range(size):
            if (i - center) ** 2 + (j - center) ** 2 <= radius**2:
                kernel[i, j] = 1
    return kernel
