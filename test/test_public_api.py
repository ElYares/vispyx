import numpy as np

import vispyx


def test_public_api_exposes_expected_symbols():
    expected_symbols = {
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
    }

    assert expected_symbols.issubset(set(vispyx.__all__))
    assert vispyx.__version__ == "0.1.0"


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
