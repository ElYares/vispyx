"""Property tests for the classical guarantees of mathematical morphology.

Every other test in this suite pins a specific input to a specific output. Those
catch a wrong value, but they say nothing about whether the operations obey the
laws that define them, because a hand-picked case can satisfy a law by accident.
These tests assert the laws themselves over random images, and they do it for
each structuring element shape, so a claim like "opening is idempotent" is
checked as a general property rather than as one example.

Three families are covered, for both value domains:

- idempotence: ``open(open(x)) == open(x)``, ``close(close(x)) == close(x)``
- duality by complement: ``erode(x, k) == ¬dilate(¬x, k)`` and the same for
  opening against closing
- ordering: ``erode(x) ⊆ open(x) ⊆ x ⊆ close(x) ⊆ dilate(x)``, plus monotonicity
  (``x ⊆ y`` implies ``erode(x) ⊆ erode(y)``)

Two choices deserve an explanation.

**Kernel size is 7, never 3 or 5.** The four shapes collapse onto each other at
small radii: at 3, cross, diamond and disk are the same matrix, and at 5 diamond
and disk still are. Only at 7 do the four differ (they weigh 49, 13, 25 and 29
pixels), so only at 7 does parametrizing by shape actually exercise four code
paths instead of two.

**Images are not framed in background.** ``test_reference_scipy`` surrounds its
inputs with a margin because ``scipy.ndimage`` treats the outside as background
while ``vispyx`` pads by reflection, and the two disagree there. No margin is
needed here: these properties are compared against ``vispyx`` itself, and
reflection turns out to preserve all of them at the border as well, which is a
fact about the padding that nothing else in the suite records.

Asserting over the whole array is therefore both simpler and stricter than
framing, but its reach has a measured limit. Switching the padding to zero-fill
breaks 96 of these tests, because zero-fill is not an extension of the image and
the laws stop holding at the edge. Switching it to ``edge`` replication breaks
none of them: replication preserves every property here just as reflection does,
so these tests constrain the *class* of padding, not the choice within it. That
choice is Decision 001's, and it stays covered by ``test_reference_scipy``.
"""

import numpy as np
import pytest

from vispyx.kernels import kernel_cross, kernel_diamond, kernel_disk, kernel_square
from vispyx.morphology import (
    gray_close,
    gray_dilate,
    gray_erode,
    gray_open,
    vpx_close,
    vpx_dilate,
    vpx_erode,
    vpx_open,
)

KERNEL_SIZE = 7
SEEDS = (11, 22, 33, 44)
SHAPES = ((20, 20), (9, 25), (8, 8))

# ``kernel_disk`` takes a radius while the other three take a side, so the disk
# is built from ``KERNEL_SIZE // 2`` to land on the same 7x7 footprint. This is
# the same conversion ``cli.py`` performs for ``--kernel-shape disk``.
KERNELS = {
    "square": kernel_square(KERNEL_SIZE),
    "cross": kernel_cross(KERNEL_SIZE),
    "diamond": kernel_diamond(KERNEL_SIZE),
    "disk": kernel_disk(KERNEL_SIZE // 2),
}

MAX = 255


@pytest.fixture(params=sorted(KERNELS), ids=sorted(KERNELS))
def kernel(request):
    """Each test runs once per structuring element shape."""
    return KERNELS[request.param]


def _binary(seed, shape=(20, 20), density=0.5):
    """Random mask in the ``vpx_*`` convention of 0 and 255."""
    rng = np.random.default_rng(seed)
    return ((rng.random(shape) > 1 - density) * MAX).astype(np.uint8)


def _grayscale(seed, shape=(20, 20)):
    """Random image spanning the full uint8 range, for the ``gray_*`` domain."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, shape, dtype=np.uint8)


def _assert_contained(smaller, larger, message):
    """Assert a pixelwise ``<=`` and report where it first breaks."""
    smaller = np.asarray(smaller)
    larger = np.asarray(larger)
    violations = np.argwhere(smaller > larger)
    assert violations.size == 0, (
        f"{message}: {len(violations)} pixel(s) violate the ordering, "
        f"first at {tuple(violations[0])} "
        f"({smaller[tuple(violations[0])]} > {larger[tuple(violations[0])]})"
    )


# --------------------------------------------------------------------------
# Idempotence
# --------------------------------------------------------------------------


@pytest.mark.parametrize("seed", SEEDS)
def test_binary_opening_is_idempotent(kernel, seed):
    """A second opening with the same kernel has nothing left to remove."""
    image = _binary(seed)
    once = vpx_open(image, kernel)
    np.testing.assert_array_equal(vpx_open(once, kernel), once)


@pytest.mark.parametrize("seed", SEEDS)
def test_binary_closing_is_idempotent(kernel, seed):
    """A second closing with the same kernel has nothing left to fill."""
    image = _binary(seed)
    once = vpx_close(image, kernel)
    np.testing.assert_array_equal(vpx_close(once, kernel), once)


@pytest.mark.parametrize("seed", SEEDS)
def test_grayscale_opening_is_idempotent(kernel, seed):
    image = _grayscale(seed)
    once = gray_open(image, kernel)
    np.testing.assert_array_equal(gray_open(once, kernel), once)


@pytest.mark.parametrize("seed", SEEDS)
def test_grayscale_closing_is_idempotent(kernel, seed):
    image = _grayscale(seed)
    once = gray_close(image, kernel)
    np.testing.assert_array_equal(gray_close(once, kernel), once)


# --------------------------------------------------------------------------
# Duality by complement
# --------------------------------------------------------------------------


@pytest.mark.parametrize("seed", SEEDS)
def test_binary_erosion_is_dual_to_dilation(kernel, seed):
    """Eroding is dilating the complement and complementing the result back."""
    image = _binary(seed)
    np.testing.assert_array_equal(
        vpx_erode(image, kernel), MAX - vpx_dilate(MAX - image, kernel)
    )


@pytest.mark.parametrize("seed", SEEDS)
def test_binary_opening_is_dual_to_closing(kernel, seed):
    """The same duality one level up, on the composite operations."""
    image = _binary(seed)
    np.testing.assert_array_equal(
        vpx_open(image, kernel), MAX - vpx_close(MAX - image, kernel)
    )


@pytest.mark.parametrize("seed", SEEDS)
def test_grayscale_erosion_is_dual_to_dilation(kernel, seed):
    """In grayscale the complement is the photographic negative."""
    image = _grayscale(seed)
    np.testing.assert_array_equal(
        gray_erode(image, kernel), MAX - gray_dilate(MAX - image, kernel)
    )


# --------------------------------------------------------------------------
# Ordering: extensivity, anti-extensivity and monotonicity
# --------------------------------------------------------------------------


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("shape", SHAPES)
def test_binary_operations_preserve_the_expected_ordering(kernel, seed, shape):
    """``erode ⊆ open ⊆ x ⊆ close ⊆ dilate``, on non-square images too."""
    image = _binary(seed, shape)
    eroded = vpx_erode(image, kernel)
    opened = vpx_open(image, kernel)
    closed = vpx_close(image, kernel)
    dilated = vpx_dilate(image, kernel)

    _assert_contained(eroded, opened, "erosion should not exceed opening")
    _assert_contained(opened, image, "opening should be anti-extensive")
    _assert_contained(image, closed, "closing should be extensive")
    _assert_contained(closed, dilated, "closing should not exceed dilation")


@pytest.mark.parametrize("seed", SEEDS)
def test_grayscale_operations_preserve_the_expected_ordering(kernel, seed):
    image = _grayscale(seed)
    _assert_contained(gray_erode(image, kernel), image, "erosion is anti-extensive")
    _assert_contained(gray_open(image, kernel), image, "opening is anti-extensive")
    _assert_contained(image, gray_close(image, kernel), "closing is extensive")
    _assert_contained(image, gray_dilate(image, kernel), "dilation is extensive")


@pytest.mark.parametrize("seed", SEEDS)
def test_binary_operations_are_monotonic(kernel, seed):
    """Growing the input can only grow the output, never shrink it."""
    smaller = _binary(seed, density=0.35)
    larger = np.maximum(smaller, _binary(seed + 1, density=0.35))

    _assert_contained(smaller, larger, "the fixture itself must be nested")
    _assert_contained(
        vpx_erode(smaller, kernel), vpx_erode(larger, kernel), "erosion is monotonic"
    )
    _assert_contained(
        vpx_dilate(smaller, kernel), vpx_dilate(larger, kernel), "dilation is monotonic"
    )
    _assert_contained(
        vpx_open(smaller, kernel), vpx_open(larger, kernel), "opening is monotonic"
    )
    _assert_contained(
        vpx_close(smaller, kernel), vpx_close(larger, kernel), "closing is monotonic"
    )


# --------------------------------------------------------------------------
# The premise the parametrization rests on
# --------------------------------------------------------------------------


def test_the_four_kernel_shapes_are_distinct_at_this_size():
    """Guard the choice of 7: at 3 or 5 the shapes collapse and the sweep lies.

    Without this, someone lowering ``KERNEL_SIZE`` would silently turn four
    parametrized runs into two distinct ones, and every test above would keep
    passing while covering less.
    """
    footprints = {name: k.tobytes() for name, k in KERNELS.items()}
    assert len(set(footprints.values())) == len(KERNELS), (
        "the four structuring elements must differ pairwise at "
        f"KERNEL_SIZE={KERNEL_SIZE}; got weights "
        f"{ {name: int(k.sum()) for name, k in KERNELS.items()} }"
    )
