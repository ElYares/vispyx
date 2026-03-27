import numpy as np


def _validate_image(image):
    """Validate and normalize the input image to a binary uint8 array."""
    image = np.asarray(image)
    if image.ndim != 2:
        raise ValueError("image must be a 2D array")
    return (image > 0).astype(np.uint8)


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
