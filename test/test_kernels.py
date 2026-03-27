import numpy as np
import pytest

from vispyx.kernels import kernel_cross, kernel_diamond, kernel_disk, kernel_square


def test_kernel_square_returns_odd_square_of_ones():
    result = kernel_square(3)
    expected = np.ones((3, 3), dtype=np.uint8)
    np.testing.assert_array_equal(result, expected)


def test_kernel_cross_returns_expected_shape():
    result = kernel_cross(3)
    expected = np.array(
        [
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0],
        ],
        dtype=np.uint8,
    )
    np.testing.assert_array_equal(result, expected)


def test_kernel_diamond_returns_expected_shape():
    result = kernel_diamond(5)
    expected = np.array(
        [
            [0, 0, 1, 0, 0],
            [0, 1, 1, 1, 0],
            [1, 1, 1, 1, 1],
            [0, 1, 1, 1, 0],
            [0, 0, 1, 0, 0],
        ],
        dtype=np.uint8,
    )
    np.testing.assert_array_equal(result, expected)


def test_kernel_disk_radius_one_returns_cross_like_shape():
    result = kernel_disk(1)
    expected = np.array(
        [
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0],
        ],
        dtype=np.uint8,
    )
    np.testing.assert_array_equal(result, expected)


def test_kernel_square_rejects_even_size():
    with pytest.raises(ValueError, match="size must be odd"):
        kernel_square(4)


def test_kernel_disk_rejects_negative_radius():
    with pytest.raises(ValueError, match="radius must be a non-negative integer"):
        kernel_disk(-1)
