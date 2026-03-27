import numpy as np

import vispyx


def test_public_api_exposes_expected_symbols():
    expected_symbols = {
        "apply_clahe",
        "gray_blackhat",
        "gray_close",
        "gray_dilate",
        "gray_erode",
        "gray_gradient",
        "gray_open",
        "gray_tophat",
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
        "vpx_reconstruct",
        "vpx_skeletonize",
        "vpx_thin",
        "vpx_tophat",
    }

    assert expected_symbols.issubset(set(vispyx.__all__))
    assert vispyx.__version__ == "0.2.0"


def test_public_api_kernel_and_morphology_work_together():
    image = np.zeros((5, 5), dtype=np.uint8)
    image[2, 2] = 255

    kernel = vispyx.kernel_cross(3)
    result = vispyx.vpx_dilate(image, kernel=kernel)

    expected = np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 0, 255, 0, 0],
            [0, 255, 255, 255, 0],
            [0, 0, 255, 0, 0],
            [0, 0, 0, 0, 0],
        ],
        dtype=np.uint8,
    )

    np.testing.assert_array_equal(result, expected)


def test_public_api_exposes_grayscale_morphology():
    image = np.array(
        [
            [10, 20, 30],
            [40, 50, 60],
            [70, 80, 90],
        ],
        dtype=np.uint8,
    )

    result = vispyx.gray_dilate(image)
    expected = np.array(
        [
            [50, 60, 60],
            [80, 90, 90],
            [80, 90, 90],
        ],
        dtype=np.uint8,
    )

    np.testing.assert_array_equal(result, expected)


def test_public_api_exposes_grayscale_opening():
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

    result = vispyx.gray_open(image)

    np.testing.assert_array_equal(result, np.full((5, 5), 10, dtype=np.uint8))


def test_public_api_exposes_grayscale_gradient():
    image = np.array(
        [
            [10, 20, 30],
            [40, 50, 60],
            [70, 80, 90],
        ],
        dtype=np.uint8,
    )

    result = vispyx.gray_gradient(image)
    expected = np.array(
        [
            [40, 50, 40],
            [70, 80, 70],
            [40, 50, 40],
        ],
        dtype=np.uint8,
    )

    np.testing.assert_array_equal(result, expected)


def test_public_api_exposes_binary_reconstruction():
    marker = np.zeros((5, 5), dtype=np.uint8)
    marker[2, 2] = 255
    mask = np.zeros((5, 5), dtype=np.uint8)
    mask[1:4, 1:4] = 255

    result = vispyx.vpx_reconstruct(marker, mask)

    np.testing.assert_array_equal(result, mask)


def test_public_api_exposes_binary_skeletonization():
    image = np.zeros((5, 5), dtype=np.uint8)
    image[1:4, 1:4] = 255

    result = vispyx.vpx_skeletonize(image)

    assert result.dtype == np.uint8
    assert np.count_nonzero(result) > 0


def test_public_api_exposes_binary_thinning():
    image = np.zeros((5, 5), dtype=np.uint8)
    image[1:4, 1:4] = 255

    result = vispyx.vpx_thin(image, iterations=1)

    assert result.dtype == np.uint8
    assert np.count_nonzero(result) > 0
