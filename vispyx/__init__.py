"""Public package interface for vispyx."""

from vispyx.kernels import kernel_cross, kernel_diamond, kernel_disk, kernel_square
from vispyx.morphology import vpx_close, vpx_dilate, vpx_erode, vpx_gradient, vpx_open
from vispyx.preprocessing import apply_clahe
from vispyx.segmentation import segment_otsu
from vispyx.utils import read_grayscale, show_image

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "apply_clahe",
    "kernel_cross",
    "kernel_diamond",
    "kernel_disk",
    "kernel_square",
    "read_grayscale",
    "segment_otsu",
    "show_image",
    "vpx_close",
    "vpx_dilate",
    "vpx_erode",
    "vpx_gradient",
    "vpx_open",
]
