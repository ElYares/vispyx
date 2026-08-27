"""Differential tests against the SciPy reference implementation.

``vispyx`` implements morphology from scratch. ``morph_scipy.MorphologicalProcessor``
implements the same binary operations on top of ``scipy.ndimage`` with matching
kernel and iteration semantics. These tests compare one against the other, which
is the cross-validation the suite was missing.

Border handling is the one place where the two deliberately disagree:
``vispyx`` pads by reflection while ``scipy.ndimage`` treats the outside as
background. Every comparison here therefore runs on images surrounded by a
frame of background wide enough that the operation never reaches the edge, so
reflection and zero-fill see the same values. ``test_border_handling_differs``
covers the divergence itself.
"""

import numpy as np
import pytest

pytest.importorskip("scipy", reason="scipy is only needed for the reference oracle")

from morph_scipy import MorphologicalProcessor

from vispyx import vpx_close, vpx_dilate, vpx_erode, vpx_gradient, vpx_open

KERNEL_SIZES = (3, 5)
ITERATIONS = (1, 2, 3)


def _as_mask(image):
    """Normalize to 0/1 so the two conventions can be compared."""
    return (np.asarray(image) > 0).astype(np.uint8)


def _framed_noise(seed, kernel_size, iterations, size=20, density=0.5):
    """Random binary image inside a background frame the operation cannot cross.

    A composite operation reaches at most ``(kernel_size // 2) * iterations``
    pixels per stage and has two stages, so a frame twice that width keeps every
    window that touches foreground fully inside the array.
    """
    frame = (kernel_size // 2) * iterations * 2
    rng = np.random.default_rng(seed)
    framed = np.zeros((size + 2 * frame, size + 2 * frame), dtype=np.uint8)
    framed[frame:-frame, frame:-frame] = (rng.random((size, size)) > 1 - density) * 255
    return framed


def _reference(kernel_size, iterations):
    return MorphologicalProcessor(kernel_size=kernel_size, iterations=iterations)


def _kernel(kernel_size):
    return np.ones((kernel_size, kernel_size), dtype=np.uint8)


@pytest.mark.parametrize("kernel_size", KERNEL_SIZES)
@pytest.mark.parametrize("iterations", ITERATIONS)
def test_erode_matches_scipy(kernel_size, iterations):
    image = _framed_noise(101, kernel_size, iterations)
    result = vpx_erode(image, _kernel(kernel_size), iterations)
    expected = _reference(kernel_size, iterations).erode(image)
    np.testing.assert_array_equal(_as_mask(result), _as_mask(expected))


@pytest.mark.parametrize("kernel_size", KERNEL_SIZES)
@pytest.mark.parametrize("iterations", ITERATIONS)
def test_dilate_matches_scipy(kernel_size, iterations):
    image = _framed_noise(202, kernel_size, iterations)
    result = vpx_dilate(image, _kernel(kernel_size), iterations)
    expected = _reference(kernel_size, iterations).dilate(image)
    np.testing.assert_array_equal(_as_mask(result), _as_mask(expected))


@pytest.mark.parametrize("kernel_size", KERNEL_SIZES)
@pytest.mark.parametrize("iterations", ITERATIONS)
def test_open_matches_scipy(kernel_size, iterations):
    image = _framed_noise(303, kernel_size, iterations)
    result = vpx_open(image, _kernel(kernel_size), iterations)
    expected = _reference(kernel_size, iterations).open(image)
    np.testing.assert_array_equal(_as_mask(result), _as_mask(expected))


@pytest.mark.parametrize("kernel_size", KERNEL_SIZES)
@pytest.mark.parametrize("iterations", ITERATIONS)
def test_close_matches_scipy(kernel_size, iterations):
    image = _framed_noise(404, kernel_size, iterations)
    result = vpx_close(image, _kernel(kernel_size), iterations)
    expected = _reference(kernel_size, iterations).close(image)
    np.testing.assert_array_equal(_as_mask(result), _as_mask(expected))


def test_gradient_matches_scipy_once_normalized():
    """The two gradients agree on shape, not on intermediate representation.

    ``morph_scipy.gradient`` re-binarizes the already binary results of
    ``dilate``/``erode`` before subtracting, while ``vpx_gradient`` subtracts in
    ``int16`` over 0/255. For binary input the outcome is the same region, so the
    comparison is made on masks.
    """
    image = _framed_noise(505, kernel_size=3, iterations=1)
    result = vpx_gradient(image, _kernel(3), 1)
    expected = _reference(3, 1).gradient(image)

    assert set(np.unique(result)) <= {0, 255}
    np.testing.assert_array_equal(_as_mask(result), _as_mask(expected))


@pytest.mark.parametrize("density", (0.2, 0.5, 0.8))
def test_open_and_close_agree_on_sparse_and_dense_images(density):
    """Both extremes matter: sparse images stress opening, dense ones closing."""
    image = _framed_noise(606, kernel_size=3, iterations=1, size=32, density=density)
    reference = _reference(3, 1)

    np.testing.assert_array_equal(
        _as_mask(vpx_open(image, _kernel(3), 1)), _as_mask(reference.open(image))
    )
    np.testing.assert_array_equal(
        _as_mask(vpx_close(image, _kernel(3), 1)), _as_mask(reference.close(image))
    )


def test_border_handling_differs_on_purpose():
    """vispyx reflects at the border; scipy.ndimage treats the outside as background.

    A solid block touching the edge survives erosion under reflection, because the
    reflected neighbourhood is foreground too. Under zero-fill it does not. This is
    the documented divergence, not a bug in either implementation.
    """
    image = np.full((6, 6), 255, dtype=np.uint8)

    mine = vpx_erode(image, _kernel(3), 1)
    theirs = _reference(3, 1).erode(image)

    assert np.all(mine == 255), "reflection keeps the whole block alive"
    assert np.all(_as_mask(theirs)[0, :] == 0), "zero-fill erodes the outer row"
    np.testing.assert_array_equal(
        _as_mask(mine)[1:-1, 1:-1], _as_mask(theirs)[1:-1, 1:-1]
    )
