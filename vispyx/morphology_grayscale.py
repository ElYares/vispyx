"""Grayscale morphological operations implemented from scratch."""

import numpy as np

from vispyx.morphology_common import apply_grayscale_operation, validate_grayscale_image


def gray_erode(image, kernel=None, iterations=1):
    """Apply grayscale erosion using the minimum value under the active kernel."""
    return apply_grayscale_operation(
        image,
        kernel,
        iterations,
        reducer=lambda region: np.min(region),
    )


def gray_dilate(image, kernel=None, iterations=1):
    """Apply grayscale dilation using the maximum value under the active kernel."""
    return apply_grayscale_operation(
        image,
        kernel,
        iterations,
        reducer=lambda region: np.max(region),
    )


def gray_open(image, kernel=None, iterations=1):
    """Apply grayscale opening: erosion followed by dilation."""
    eroded = gray_erode(image, kernel, iterations)
    return gray_dilate(eroded, kernel, iterations)


def gray_close(image, kernel=None, iterations=1):
    """Apply grayscale closing: dilation followed by erosion."""
    dilated = gray_dilate(image, kernel, iterations)
    return gray_erode(dilated, kernel, iterations)


def gray_gradient(image, kernel=None, iterations=1):
    """Compute the grayscale morphological gradient (dilation - erosion)."""
    img = validate_grayscale_image(image)
    dilated = gray_dilate(img, kernel, iterations)
    eroded = gray_erode(img, kernel, iterations)
    gradient = (dilated.astype(np.int32) - eroded.astype(np.int32)).clip(min=0)
    return gradient.astype(img.dtype, copy=False)


def gray_tophat(image, kernel=None, iterations=1):
    """Compute the grayscale white top-hat transform (image - opening)."""
    img = validate_grayscale_image(image)
    opened = gray_open(img, kernel, iterations)
    top_hat = (img.astype(np.int32) - opened.astype(np.int32)).clip(min=0)
    return top_hat.astype(img.dtype, copy=False)


def gray_blackhat(image, kernel=None, iterations=1):
    """Compute the grayscale black-hat transform (closing - image)."""
    img = validate_grayscale_image(image)
    closed = gray_close(img, kernel, iterations)
    black_hat = (closed.astype(np.int32) - img.astype(np.int32)).clip(min=0)
    return black_hat.astype(img.dtype, copy=False)
