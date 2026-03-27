import numpy as np
import pytest

from vispyx.morphology import (
    vpx_close,
    vpx_dilate,
    vpx_erode,
    vpx_gradient,
    vpx_open,
)


def _to_uint8(binary_matrix):
    """Convert a binary matrix of 0/1 values to the package convention 0/255."""
    return np.array(binary_matrix, dtype=np.uint8) * 255


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
