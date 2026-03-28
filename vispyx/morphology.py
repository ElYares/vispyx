"""Compatibility facade for morphological operations.

This module preserves the historical import path `vispyx.morphology` while the
implementation now lives in dedicated binary and grayscale modules.
"""

from vispyx.morphology_binary import (
    vpx_blackhat,
    vpx_boundary,
    vpx_close,
    vpx_dilate,
    vpx_erode,
    vpx_gradient,
    vpx_hitmiss,
    vpx_open,
    vpx_reconstruct,
    vpx_skeletonize,
    vpx_thin,
    vpx_tophat,
)
from vispyx.morphology_common import pad_image as vpx_pad_image
from vispyx.morphology_grayscale import (
    gray_blackhat,
    gray_close,
    gray_dilate,
    gray_erode,
    gray_gradient,
    gray_open,
    gray_tophat,
)

__all__ = [
    "gray_blackhat",
    "gray_close",
    "gray_dilate",
    "gray_erode",
    "gray_gradient",
    "gray_open",
    "gray_tophat",
    "vpx_blackhat",
    "vpx_boundary",
    "vpx_close",
    "vpx_dilate",
    "vpx_erode",
    "vpx_gradient",
    "vpx_hitmiss",
    "vpx_open",
    "vpx_pad_image",
    "vpx_reconstruct",
    "vpx_skeletonize",
    "vpx_thin",
    "vpx_tophat",
]
