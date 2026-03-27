import numpy as np


def _validate_image(image):
    """Validate and normalize the input image to a binary uint8 array."""
    image = np.asarray(image)
    if image.ndim != 2:
        raise ValueError("image must be a 2D array")
    return (image > 0).astype(np.uint8)


def _validate_grayscale_image(image):
    """Validate and normalize the input image for grayscale morphology."""
    image = np.asarray(image)
    if image.ndim != 2:
        raise ValueError("image must be a 2D array")
    if not np.issubdtype(image.dtype, np.number):
        raise ValueError("image must contain numeric values")
    return image


def _validate_iterations(iterations):
    """Validate the number of morphological iterations."""
    if not isinstance(iterations, int) or iterations <= 0:
        raise ValueError("iterations must be a positive integer")


def _validate_kernel(kernel):
    """Validate and normalize the structuring element."""
    if kernel is None:
        kernel = np.ones((3, 3), dtype=np.uint8)
    kernel = np.asarray(kernel)
    if kernel.ndim != 2:
        raise ValueError("kernel must be a 2D array")
    if kernel.size == 0:
        raise ValueError("kernel must not be empty")
    if kernel.shape[0] % 2 == 0 or kernel.shape[1] % 2 == 0:
        raise ValueError("kernel dimensions must be odd")
    normalized_kernel = (kernel > 0).astype(np.uint8)
    if not np.any(normalized_kernel):
        raise ValueError("kernel must contain at least one active element")
    return normalized_kernel


def _validate_hitmiss_kernels(kernel_hit, kernel_miss):
    """Validate the kernel pair used by the hit-or-miss transform."""
    kernel_hit = _validate_kernel(kernel_hit)
    kernel_miss = _validate_kernel(kernel_miss)
    if kernel_hit.shape != kernel_miss.shape:
        raise ValueError("kernel_hit and kernel_miss must have the same shape")
    if np.any((kernel_hit == 1) & (kernel_miss == 1)):
        raise ValueError("kernel_hit and kernel_miss must not overlap")
    return kernel_hit, kernel_miss


def vpx_pad_image(image, kernel):
    """Apply reflection padding based on the kernel shape."""
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    return np.pad(image, ((ph, ph), (pw, pw)), mode="reflect")


def _apply_binary_operation(image, kernel, iterations, reducer):
    """Apply a per-window binary morphological reduction."""
    img = _validate_image(image)
    _validate_iterations(iterations)
    kernel = _validate_kernel(kernel)

    kh, kw = kernel.shape
    active_mask = kernel == 1
    active_count = int(np.sum(kernel))

    for _ in range(iterations):
        padded = vpx_pad_image(img, kernel)
        output = np.zeros_like(img)

        # Evaluate the active kernel support at every pixel location.
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                region = padded[i : i + kh, j : j + kw]
                output[i, j] = reducer(region[active_mask], active_count)

        img = output

    return img * 255


def _apply_grayscale_operation(image, kernel, iterations, reducer):
    """Apply a per-window grayscale morphological reduction."""
    img = _validate_grayscale_image(image).copy()
    _validate_iterations(iterations)
    kernel = _validate_kernel(kernel)

    kh, kw = kernel.shape
    active_mask = kernel == 1

    for _ in range(iterations):
        padded = vpx_pad_image(img, kernel)
        output = np.zeros_like(img)

        # Evaluate only the active kernel support for each local neighborhood.
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                region = padded[i : i + kh, j : j + kw]
                output[i, j] = reducer(region[active_mask])

        img = output

    return img.astype(image.dtype, copy=False)


def vpx_erode(image, kernel=None, iterations=1):
    """Apply binary erosion using a custom structuring element."""
    return _apply_binary_operation(
        image,
        kernel,
        iterations,
        reducer=lambda region, active_count: int(np.sum(region) == active_count),
    )


def vpx_dilate(image, kernel=None, iterations=1):
    """Apply binary dilation using a custom structuring element."""
    return _apply_binary_operation(
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
    img = _validate_image(image) * 255
    opened = vpx_open(img, kernel, iterations)
    top_hat = (img.astype(np.int16) - opened.astype(np.int16)).clip(min=0)
    return top_hat.astype(np.uint8)


def vpx_blackhat(image, kernel=None, iterations=1):
    """Compute the black-hat transform (closing - image)."""
    img = _validate_image(image) * 255
    closed = vpx_close(img, kernel, iterations)
    black_hat = (closed.astype(np.int16) - img.astype(np.int16)).clip(min=0)
    return black_hat.astype(np.uint8)


def vpx_boundary(image, kernel=None, iterations=1):
    """Compute the internal boundary of a binary object (image - erosion)."""
    img = _validate_image(image) * 255
    eroded = vpx_erode(img, kernel, iterations)
    boundary = (img.astype(np.int16) - eroded.astype(np.int16)).clip(min=0)
    return boundary.astype(np.uint8)


def vpx_hitmiss(image, kernel_hit, kernel_miss):
    """Detect binary patterns using the hit-or-miss transform."""
    img = _validate_image(image) * 255
    kernel_hit, kernel_miss = _validate_hitmiss_kernels(kernel_hit, kernel_miss)
    hit = vpx_erode(img, kernel=kernel_hit, iterations=1)
    miss = vpx_erode(255 - img, kernel=kernel_miss, iterations=1)
    return np.logical_and(hit > 0, miss > 0).astype(np.uint8) * 255


def gray_erode(image, kernel=None, iterations=1):
    """Apply grayscale erosion using the minimum value under the active kernel."""
    return _apply_grayscale_operation(
        image,
        kernel,
        iterations,
        reducer=lambda region: np.min(region),
    )


def gray_dilate(image, kernel=None, iterations=1):
    """Apply grayscale dilation using the maximum value under the active kernel."""
    return _apply_grayscale_operation(
        image,
        kernel,
        iterations,
        reducer=lambda region: np.max(region),
    )


def gray_open(image, kernel=None, iterations=1):
    """Apply grayscale opening: erosion followed by dilation."""
    eroded = gray_erode(image, kernel, iterations)
    return gray_dilate(eroded, kernel, iterations)


def gray_close(image, kernel=None, iterations=1):
    """Apply grayscale closing: dilation followed by erosion."""
    dilated = gray_dilate(image, kernel, iterations)
    return gray_erode(dilated, kernel, iterations)
