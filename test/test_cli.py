import numpy as np
import cv2

from vispyx.cli import run_vpx_reconstruct, run_vpx_skeletonize, run_vpx_thin


def test_run_vpx_reconstruct_restores_binary_region_from_files(tmp_path):
    marker = np.zeros((5, 5), dtype=np.uint8)
    marker[2, 2] = 255

    mask = np.zeros((5, 5), dtype=np.uint8)
    mask[1:4, 1:4] = 255

    marker_path = tmp_path / "marker.pgm"
    mask_path = tmp_path / "mask.pgm"

    cv2.imwrite(str(marker_path), marker)
    cv2.imwrite(str(mask_path), mask)

    result = run_vpx_reconstruct(str(marker_path), str(mask_path), kernel_size=3)

    np.testing.assert_array_equal(result, mask)


def test_run_vpx_skeletonize_reduces_binary_region_from_file(tmp_path):
    image = np.zeros((5, 5), dtype=np.uint8)
    image[1:4, 1:4] = 255
    image_path = tmp_path / "block.pgm"
    cv2.imwrite(str(image_path), image)

    result = run_vpx_skeletonize(str(image_path))

    assert result.dtype == np.uint8
    assert 0 < np.count_nonzero(result) < np.count_nonzero(image)


def test_run_vpx_thin_supports_iterative_thinning_from_file(tmp_path):
    image = np.zeros((5, 5), dtype=np.uint8)
    image[1:4, 1:4] = 255
    image_path = tmp_path / "thin_block.pgm"
    cv2.imwrite(str(image_path), image)

    result = run_vpx_thin(str(image_path), iterations=1)

    assert result.dtype == np.uint8
    assert 0 < np.count_nonzero(result) <= np.count_nonzero(image)
