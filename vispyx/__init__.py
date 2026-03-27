"""Public package interface for vispyx."""

from vispyx.kernels import kernel_cross, kernel_diamond, kernel_disk, kernel_square
from vispyx.morphology import (
    gray_close,
    gray_dilate,
    gray_erode,
    gray_open,
    vpx_blackhat,
    vpx_boundary,
    vpx_close,
    vpx_dilate,
    vpx_erode,
    vpx_gradient,
    vpx_hitmiss,
    vpx_open,
    vpx_tophat,
)
from vispyx.preprocessing import apply_clahe
from vispyx.segmentation import segment_otsu
from vispyx.utils import read_grayscale, show_image

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "apply_clahe",
    "gray_close",
    "gray_dilate",
    "gray_erode",
    "gray_open",
    "kernel_cross",
    "kernel_diamond",
    "kernel_disk",
    "kernel_square",
    "read_grayscale",
    "segment_otsu",
    "show_image",
    "vpx_blackhat",
    "vpx_boundary",
    "vpx_close",
    "vpx_dilate",
    "vpx_erode",
    "vpx_gradient",
    "vpx_hitmiss",
    "vpx_open",
    "vpx_tophat",
]
