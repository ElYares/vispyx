"""Shared validation and helper utilities for morphological operations."""

import numbers

import numpy as np

from vispyx import _backend


def validate_binary_image(image):
    """Validate and normalize the input image to a binary uint8 array."""
    image = np.asarray(image)
    if image.ndim != 2:
        raise ValueError("image must be a 2D array")
    if image.size == 0:
        raise ValueError("image must not be empty")
    return (image > 0).astype(np.uint8)


def validate_grayscale_image(image):
    """Validate and normalize the input image for grayscale morphology."""
    image = np.asarray(image)
    if image.ndim != 2:
        raise ValueError("image must be a 2D array")
    if image.size == 0:
        raise ValueError("image must not be empty")
    if not np.issubdtype(image.dtype, np.number):
        raise ValueError("image must contain numeric values")
    return image


def validate_iterations(iterations):
    """Validate the number of morphological iterations.

    Any integer type is accepted, NumPy integers included: ``np.int64(2)`` is a
    perfectly valid count and rejecting it only punished callers who derived the
    value from an array. Booleans are rejected on purpose, even though ``bool``
    is a subclass of ``int``, because ``iterations=True`` is a mistake worth
    reporting rather than a request for one iteration.
    """
    if isinstance(iterations, bool) or not isinstance(iterations, numbers.Integral):
        raise ValueError("iterations must be a positive integer")
    if iterations <= 0:
        raise ValueError("iterations must be a positive integer")


def validate_kernel(kernel):
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


def validate_hitmiss_kernels(kernel_hit, kernel_miss):
    """Validate the kernel pair used by the hit-or-miss transform."""
    kernel_hit = validate_kernel(kernel_hit)
    kernel_miss = validate_kernel(kernel_miss)
    if kernel_hit.shape != kernel_miss.shape:
        raise ValueError("kernel_hit and kernel_miss must have the same shape")
    if np.any((kernel_hit == 1) & (kernel_miss == 1)):
        raise ValueError("kernel_hit and kernel_miss must not overlap")
    return kernel_hit, kernel_miss


def pad_image(image, kernel):
    """Apply reflection padding based on the kernel shape."""
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    return np.pad(image, ((ph, ph), (pw, pw)), mode="reflect")


def apply_binary_operation(image, kernel, iterations, reducer, native_op=None):
    """Apply a per-window binary morphological reduction.

    ``native_op`` nombra la operación equivalente del backend opcional en Rust.
    Cuando ese backend está activo y la operación es una de las que implementa,
    el recorrido se delega; el resultado es idéntico bit a bit al del bucle de
    abajo, que sigue siendo la implementación de referencia. Las validaciones
    ya corrieron: el nativo nunca ve una entrada sin normalizar y nunca lanza
    los mensajes de error, que son contrato público de este módulo.
    """
    img = validate_binary_image(image)
    validate_iterations(iterations)
    kernel = validate_kernel(kernel)

    if native_op is not None:
        backend = _backend.native()
        if backend is not None:
            return backend.binary_op(img, kernel, int(iterations), native_op) * 255

    kh, kw = kernel.shape
    active_mask = kernel == 1
    active_count = int(np.sum(kernel))

    for _ in range(iterations):
        padded = pad_image(img, kernel)
        output = np.zeros_like(img)

        # Evaluate the active kernel support at every pixel location.
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                region = padded[i : i + kh, j : j + kw]
                output[i, j] = reducer(region[active_mask], active_count)

        img = output

    return img * 255


# El motor nativo solo cubre enteros. ``Ord`` en Rust es un orden total; los
# flotantes no lo tienen, y reproducir la propagacion de ``NaN`` de ``np.min``
# bit a bit no vale el riesgo. Un float cae al bucle de abajo sin avisar, que es
# el comportamiento correcto: mismo resultado, solo mas lento.
_NATIVE_GRAYSCALE_KINDS = ("i", "u")


def apply_grayscale_operation(image, kernel, iterations, reducer, native_op=None):
    """Apply a per-window grayscale morphological reduction.

    Gemelo de ``apply_binary_operation``: ``native_op`` nombra la operacion
    equivalente del backend opcional en Rust, y la delegacion va despues de las
    validaciones para que el nativo nunca vea una entrada sin normalizar.
    """
    source = validate_grayscale_image(image)
    img = source.copy()
    validate_iterations(iterations)
    kernel = validate_kernel(kernel)

    if native_op is not None and img.dtype.kind in _NATIVE_GRAYSCALE_KINDS:
        backend = _backend.native()
        if backend is not None:
            resultado = backend.grayscale_op(img, kernel, int(iterations), native_op)
            return resultado.astype(source.dtype, copy=False)

    kh, kw = kernel.shape
    active_mask = kernel == 1

    for _ in range(iterations):
        padded = pad_image(img, kernel)
        output = np.zeros_like(img)

        # Evaluate only the active kernel support for each local neighborhood.
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                region = padded[i : i + kh, j : j + kw]
                output[i, j] = reducer(region[active_mask])

        img = output

    # Cast against the validated array, not the raw argument: a nested list has
    # no ``.dtype`` and would raise AttributeError instead of a clean error.
    return img.astype(source.dtype, copy=False)
