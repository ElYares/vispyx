"""Binary morphological operations implemented from scratch."""

import numpy as np

from vispyx.morphology_common import (
    apply_binary_operation,
    validate_binary_image,
    validate_hitmiss_kernels,
    validate_iterations,
    validate_kernel,
)


def vpx_erode(image, kernel=None, iterations=1):
    """Apply binary erosion using a custom structuring element."""
    return apply_binary_operation(
        image,
        kernel,
        iterations,
        reducer=lambda region, active_count: int(np.sum(region) == active_count),
    )


def vpx_dilate(image, kernel=None, iterations=1):
    """Apply binary dilation using a custom structuring element."""
    return apply_binary_operation(
        image,
        kernel,
        iterations,
        reducer=lambda region, _active_count: int(np.any(region)),
    )


def vpx_open(image, kernel=None, iterations=1):
    """Apply binary opening: erosion followed by dilation."""
    eroded = vpx_erode(image, kernel, iterations)
    return vpx_dilate(eroded, kernel, iterations)


def vpx_close(image, kernel=None, iterations=1):
    """Apply binary closing: dilation followed by erosion."""
    dilated = vpx_dilate(image, kernel, iterations)
    return vpx_erode(dilated, kernel, iterations)


def vpx_gradient(image, kernel=None, iterations=1):
    """Compute the binary morphological gradient (dilation - erosion)."""
    dilated = vpx_dilate(image, kernel, iterations)
    eroded = vpx_erode(image, kernel, iterations)
    gradient = (dilated.astype(np.int16) - eroded.astype(np.int16)).clip(min=0)
    return gradient.astype(np.uint8)


def vpx_tophat(image, kernel=None, iterations=1):
    """Compute the white top-hat transform (image - opening)."""
    img = validate_binary_image(image) * 255
    opened = vpx_open(img, kernel, iterations)
    top_hat = (img.astype(np.int16) - opened.astype(np.int16)).clip(min=0)
    return top_hat.astype(np.uint8)


def vpx_blackhat(image, kernel=None, iterations=1):
    """Compute the black-hat transform (closing - image)."""
    img = validate_binary_image(image) * 255
    closed = vpx_close(img, kernel, iterations)
    black_hat = (closed.astype(np.int16) - img.astype(np.int16)).clip(min=0)
    return black_hat.astype(np.uint8)


def vpx_boundary(image, kernel=None, iterations=1):
    """Compute the internal boundary of a binary object (image - erosion)."""
    img = validate_binary_image(image) * 255
    eroded = vpx_erode(img, kernel, iterations)
    boundary = (img.astype(np.int16) - eroded.astype(np.int16)).clip(min=0)
    return boundary.astype(np.uint8)


def vpx_hitmiss(image, kernel_hit, kernel_miss):
    """Detect binary patterns using the hit-or-miss transform."""
    img = validate_binary_image(image) * 255
    kernel_hit, kernel_miss = validate_hitmiss_kernels(kernel_hit, kernel_miss)
    hit = vpx_erode(img, kernel=kernel_hit, iterations=1)
    miss = vpx_erode(255 - img, kernel=kernel_miss, iterations=1)
    return np.logical_and(hit > 0, miss > 0).astype(np.uint8) * 255


def vpx_reconstruct(marker, mask, kernel=None, max_iterations=None):
    """Reconstruct a binary marker under a binary mask using iterative dilation."""
    marker_img = validate_binary_image(marker)
    mask_img = validate_binary_image(mask)
    if marker_img.shape != mask_img.shape:
        raise ValueError("marker and mask must have the same shape")
    if np.any(marker_img > mask_img):
        raise ValueError("marker must be a subset of mask")

    if max_iterations is not None:
        validate_iterations(max_iterations)

    kernel = validate_kernel(kernel)
    current = marker_img.astype(np.uint8) * 255
    mask_uint8 = mask_img.astype(np.uint8) * 255
    steps = 0

    # Reconstruction stops when repeated constrained dilations no longer change the marker.
    while True:
        dilated = vpx_dilate(current, kernel=kernel, iterations=1)
        updated = np.minimum(dilated, mask_uint8).astype(np.uint8)
        if np.array_equal(updated, current):
            return updated
        current = updated
        steps += 1
        if max_iterations is not None and steps >= max_iterations:
            return current


def _count_transitions(neighbors):
    """Count 0-to-1 transitions in a circular 8-neighborhood."""
    circular = neighbors + neighbors[:1]
    return sum(current == 0 and next_ == 1 for current, next_ in zip(circular, circular[1:]))


def vpx_skeletonize(image, max_iterations=None):
    """Skeletonize a binary image using iterative thinning."""
    img = validate_binary_image(image)

    if max_iterations is not None:
        validate_iterations(max_iterations)

    current = img.copy()
    iterations = 0

    # Zhang-Suen thinning removes contour pixels while preserving connectivity.
    while True:
        changed = False

        for step in (0, 1):
            to_remove = []
            padded = np.pad(current, 1, mode="constant", constant_values=0)

            for i in range(1, padded.shape[0] - 1):
                for j in range(1, padded.shape[1] - 1):
                    if padded[i, j] != 1:
                        continue

                    p2 = padded[i - 1, j]
                    p3 = padded[i - 1, j + 1]
                    p4 = padded[i, j + 1]
                    p5 = padded[i + 1, j + 1]
                    p6 = padded[i + 1, j]
                    p7 = padded[i + 1, j - 1]
                    p8 = padded[i, j - 1]
                    p9 = padded[i - 1, j - 1]
                    neighbors = [p2, p3, p4, p5, p6, p7, p8, p9]

                    active_neighbors = int(sum(neighbors))
                    transitions = _count_transitions(neighbors)

                    if not (2 <= active_neighbors <= 6):
                        continue
                    if transitions != 1:
                        continue

                    if step == 0:
                        if p2 * p4 * p6 != 0:
                            continue
                        if p4 * p6 * p8 != 0:
                            continue
                    else:
                        if p2 * p4 * p8 != 0:
                            continue
                        if p2 * p6 * p8 != 0:
                            continue

                    to_remove.append((i - 1, j - 1))

            if to_remove:
                changed = True
                for i, j in to_remove:
                    current[i, j] = 0

        iterations += 1
        if not changed:
            return current.astype(np.uint8) * 255
        if max_iterations is not None and iterations >= max_iterations:
            return current.astype(np.uint8) * 255


def vpx_thin(image, iterations=1):
    """Thin a binary image for a controlled number of thinning iterations."""
    return vpx_skeletonize(image, max_iterations=iterations)
