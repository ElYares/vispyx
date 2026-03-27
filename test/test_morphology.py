import numpy as np
import pytest

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


def _to_uint8(binary_matrix):
    """Convert a binary matrix of 0/1 values to the package convention 0/255."""
    return np.array(binary_matrix, dtype=np.uint8) * 255


def test_gray_erode_returns_local_minimum_under_square_kernel():
    image = np.array(
        [
            [10, 20, 30],
            [40, 50, 60],
            [70, 80, 90],
        ],
        dtype=np.uint8,
    )
    expected = np.array(
        [
            [10, 10, 20],
            [10, 10, 20],
            [40, 40, 50],
        ],
        dtype=np.uint8,
    )

    result = gray_erode(image)

    np.testing.assert_array_equal(result, expected)


def test_gray_dilate_returns_local_maximum_under_square_kernel():
    image = np.array(
        [
            [10, 20, 30],
            [40, 50, 60],
            [70, 80, 90],
        ],
        dtype=np.uint8,
    )
    expected = np.array(
        [
            [50, 60, 60],
            [80, 90, 90],
            [80, 90, 90],
        ],
        dtype=np.uint8,
    )

    result = gray_dilate(image)

    np.testing.assert_array_equal(result, expected)


def test_gray_dilate_supports_custom_cross_kernel():
    image = np.array(
        [
            [10, 20, 30],
            [40, 50, 60],
            [70, 80, 90],
        ],
        dtype=np.uint8,
    )
    kernel = np.array(
        [
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0],
        ],
        dtype=np.uint8,
    )
    expected = np.array(
        [
            [40, 50, 60],
            [70, 80, 90],
            [80, 90, 90],
        ],
        dtype=np.uint8,
    )

    result = gray_dilate(image, kernel=kernel)

    np.testing.assert_array_equal(result, expected)


def test_gray_erode_supports_multiple_iterations():
    image = np.array(
        [
            [10, 20, 30, 40, 50],
            [15, 25, 35, 45, 55],
            [20, 30, 40, 50, 60],
            [25, 35, 45, 55, 65],
            [30, 40, 50, 60, 70],
        ],
        dtype=np.uint8,
    )
    expected = np.array(
        [
            [10, 10, 10, 20, 30],
            [10, 10, 10, 20, 30],
            [10, 10, 10, 20, 30],
            [15, 15, 15, 25, 35],
            [20, 20, 20, 30, 40],
        ],
        dtype=np.uint8,
    )

    result = gray_erode(image, iterations=2)

    np.testing.assert_array_equal(result, expected)


def test_gray_erode_rejects_non_numeric_images():
    image = np.array([["a", "b"], ["c", "d"]], dtype=object)

    with pytest.raises(ValueError, match="image must contain numeric values"):
        gray_erode(image)


def test_gray_open_removes_small_bright_peak():
    image = np.array(
        [
            [10, 10, 10, 10, 10],
            [10, 10, 10, 10, 10],
            [10, 10, 80, 10, 10],
            [10, 10, 10, 10, 10],
            [10, 10, 10, 10, 10],
        ],
        dtype=np.uint8,
    )
    expected = np.full((5, 5), 10, dtype=np.uint8)

    result = gray_open(image)

    np.testing.assert_array_equal(result, expected)


def test_gray_close_fills_small_dark_hole():
    image = np.array(
        [
            [80, 80, 80, 80, 80],
            [80, 80, 80, 80, 80],
            [80, 80, 10, 80, 80],
            [80, 80, 80, 80, 80],
            [80, 80, 80, 80, 80],
        ],
        dtype=np.uint8,
    )
    expected = np.full((5, 5), 80, dtype=np.uint8)

    result = gray_close(image)

    np.testing.assert_array_equal(result, expected)


def test_vpx_erode_reduces_center_block_to_single_pixel():
    image = _to_uint8(
        [
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0],
        ]
    )
    expected = _to_uint8(
        [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]
    )

    result = vpx_erode(image)

    np.testing.assert_array_equal(result, expected)


def test_vpx_dilate_expands_single_pixel_to_center_block():
    image = _to_uint8(
        [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]
    )
    expected = _to_uint8(
        [
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0],
        ]
    )

    result = vpx_dilate(image)

    np.testing.assert_array_equal(result, expected)


def test_vpx_open_removes_isolated_noise_pixel():
    image = _to_uint8(
        [
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 1],
        ]
    )
    expected = _to_uint8(
        [
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0],
        ]
    )

    result = vpx_open(image)

    np.testing.assert_array_equal(result, expected)


def test_vpx_close_fills_single_pixel_hole_inside_region():
    image = _to_uint8(
        [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 1, 0, 0],
            [0, 0, 1, 0, 1, 0, 0],
            [0, 0, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ]
    )
    expected = _to_uint8(
        [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 1, 0, 0],
            [0, 0, 1, 1, 1, 0, 0],
            [0, 0, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ]
    )

    result = vpx_close(image)

    np.testing.assert_array_equal(result, expected)


def test_vpx_gradient_returns_boundary_of_solid_region():
    image = _to_uint8(
        [
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0],
        ]
    )
    expected = _to_uint8(
        [
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 0, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
        ]
    )

    result = vpx_gradient(image)

    np.testing.assert_array_equal(result, expected)


def test_vpx_tophat_extracts_small_bright_noise_removed_by_opening():
    image = _to_uint8(
        [
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 1],
        ]
    )
    expected = _to_uint8(
        [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1],
        ]
    )

    result = vpx_tophat(image)

    np.testing.assert_array_equal(result, expected)


def test_vpx_blackhat_extracts_small_hole_filled_by_closing():
    image = _to_uint8(
        [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 1, 0, 0],
            [0, 0, 1, 0, 1, 0, 0],
            [0, 0, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ]
    )
    expected = _to_uint8(
        [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ]
    )

    result = vpx_blackhat(image)

    np.testing.assert_array_equal(result, expected)


def test_vpx_boundary_extracts_internal_contour_of_region():
    image = _to_uint8(
        [
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0],
        ]
    )
    expected = _to_uint8(
        [
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 0, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0],
        ]
    )

    result = vpx_boundary(image)

    np.testing.assert_array_equal(result, expected)


def test_vpx_hitmiss_detects_cross_pattern_at_center():
    image = _to_uint8(
        [
            [0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0],
        ]
    )
    kernel_hit = np.array(
        [
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0],
        ],
        dtype=np.uint8,
    )
    kernel_miss = np.array(
        [
            [1, 0, 1],
            [0, 0, 0],
            [1, 0, 1],
        ],
        dtype=np.uint8,
    )
    expected = _to_uint8(
        [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]
    )

    result = vpx_hitmiss(image, kernel_hit=kernel_hit, kernel_miss=kernel_miss)

    np.testing.assert_array_equal(result, expected)


def test_vpx_hitmiss_rejects_overlapping_kernels():
    image = _to_uint8(
        [
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0],
        ]
    )
    kernel_hit = np.array(
        [
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0],
        ],
        dtype=np.uint8,
    )
    kernel_miss = np.array(
        [
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0],
        ],
        dtype=np.uint8,
    )

    with pytest.raises(ValueError, match="must not overlap"):
        vpx_hitmiss(image, kernel_hit=kernel_hit, kernel_miss=kernel_miss)


def test_vpx_erode_supports_multiple_iterations():
    image = _to_uint8(
        [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ]
    )
    expected = _to_uint8(
        [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ]
    )

    result = vpx_erode(image, iterations=2)

    np.testing.assert_array_equal(result, expected)


def test_vpx_dilate_supports_custom_cross_kernel():
    image = _to_uint8(
        [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]
    )
    kernel = np.array(
        [
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0],
        ],
        dtype=np.uint8,
    )
    expected = _to_uint8(
        [
            [0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0],
        ]
    )

    result = vpx_dilate(image, kernel=kernel)

    np.testing.assert_array_equal(result, expected)


def test_vpx_erode_rejects_even_sized_kernel():
    image = _to_uint8([[1, 1], [1, 1]])
    kernel = np.ones((2, 2), dtype=np.uint8)

    with pytest.raises(ValueError, match="kernel dimensions must be odd"):
        vpx_erode(image, kernel=kernel)


def test_vpx_erode_rejects_non_positive_iterations():
    image = _to_uint8([[1, 1], [1, 1]])

    with pytest.raises(ValueError, match="iterations must be a positive integer"):
        vpx_erode(image, iterations=0)


def test_vpx_erode_rejects_non_2d_images():
    image = np.zeros((2, 2, 2), dtype=np.uint8)

    with pytest.raises(ValueError, match="image must be a 2D array"):
        vpx_erode(image)
