"""Paridad entre el motor de Python y el backend opcional en Rust.

`vispyx` implementa la morfología desde cero en Python. `vispyx-native` recorre
el mismo algoritmo en Rust. Estos tests exigen que coincidan **exactamente**:
mismo valor en cada píxel y mismo dtype. El Python es el oráculo, igual que
`morph_scipy.MorphologicalProcessor` lo es en `test_reference_scipy.py`.

La diferencia con aquel archivo es deliberada: allá cada imagen va rodeada de un
marco de fondo para que la operación nunca alcance el borde, porque scipy y
vispyx tratan el exterior distinto. Acá pasa lo contrario. El borde es
justamente donde el port se rompe, así que estas pruebas lo tocan a propósito:
imágenes con foreground pegado al margen, y kernels más grandes que la imagen,
donde el reflejo de `np.pad` se pliega más de una vez.
"""

import numpy as np
import pytest

pytest.importorskip(
    "vispyx_native",
    reason="the Rust backend is optional; install it from native/",
)

from vispyx import _backend
from vispyx import (
    kernel_cross,
    kernel_diamond,
    kernel_disk,
    kernel_square,
    vpx_boundary,
    vpx_close,
    vpx_dilate,
    vpx_erode,
    vpx_gradient,
    vpx_hitmiss,
    vpx_open,
    vpx_reconstruct,
    vpx_tophat,
)

# Las cuatro formas coinciden entre sí en radios chicos, así que el tamaño 7 es
# el primero que separa cruz, diamante y disco de verdad.
#
# Los cuatro últimos no son decoración. Un kernel **sólido** no distingue el
# reflejo de la repetición de borde: en la columna 0, el reflejo muestrea
# `{img[1], img[0], img[1]}` y la repetición `{img[0], img[0], img[1]}`, que
# como conjunto son el mismo, y `min`/`max` no ven la diferencia. Lo mismo pasa
# con cruz, diamante y disco: son simétricos y contienen el centro.
#
# Para que el borde discrimine de verdad hace falta un soporte que **excluya el
# centro** y sea asimétrico. Verificado por mutación: cambiar el reflejo por
# `clamp` en el Rust deja pasar todos los kernels sólidos y solo cae en estos.
KERNELS = (
    None,
    kernel_square(3),
    kernel_cross(7),
    kernel_diamond(7),
    kernel_disk(3),
    np.ones((7, 3), dtype=np.uint8),
    np.array([[1, 0, 1]], dtype=np.uint8),
    np.array([[1, 0, 0]], dtype=np.uint8),
    np.array([[1], [0], [0]], dtype=np.uint8),
    np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=np.uint8),
)

KERNEL_IDS = (
    "default",
    "square3",
    "cross7",
    "diamond7",
    "disk3",
    "tall7x3",
    "hueco-row1x3",
    "solo-izquierda",
    "solo-arriba",
    "solo-esquina-nw",
)

ELEMENTWISE_OPERATIONS = (vpx_erode, vpx_dilate)
COMPOSED_OPERATIONS = (vpx_open, vpx_close, vpx_gradient, vpx_tophat, vpx_boundary)


def both_backends(operation, *args, **kwargs):
    """Run one operation through each backend and return the two results."""
    with _backend.override("python"):
        expected = operation(*args, **kwargs)
    with _backend.override("rust"):
        actual = operation(*args, **kwargs)
    return expected, actual


def assert_identical(expected, actual):
    """The two backends must agree on values and on dtype."""
    assert actual.dtype == expected.dtype
    assert np.array_equal(actual, expected)


def noise(shape, seed, density=0.5):
    """Random binary image with foreground reaching the borders."""
    rng = np.random.default_rng(seed)
    return ((rng.random(shape) > 1 - density).astype(np.uint8)) * 255


def test_the_backend_under_test_is_actually_the_native_one():
    """Guard against a green suite that never exercised Rust at all."""
    with _backend.override("rust"):
        assert _backend.name() == "rust"
        assert "erode" in _backend.native().supported_ops()


@pytest.mark.parametrize("operation", ELEMENTWISE_OPERATIONS, ids=lambda op: op.__name__)
@pytest.mark.parametrize("kernel", KERNELS, ids=KERNEL_IDS)
@pytest.mark.parametrize("iterations", (1, 2, 3))
def test_erode_and_dilate_match_on_noise(operation, kernel, iterations):
    image = noise((16, 13), seed=11)
    assert_identical(*both_backends(operation, image, kernel, iterations))


@pytest.mark.parametrize("operation", COMPOSED_OPERATIONS, ids=lambda op: op.__name__)
@pytest.mark.parametrize("kernel", KERNELS, ids=KERNEL_IDS)
def test_composed_operations_match(operation, kernel):
    image = noise((14, 14), seed=23)
    assert_identical(*both_backends(operation, image, kernel, 2))


@pytest.mark.parametrize(
    "shape",
    ((1, 1), (1, 9), (9, 1), (2, 2), (3, 4), (5, 5)),
    ids=("1x1", "1x9", "9x1", "2x2", "3x4", "5x5"),
)
@pytest.mark.parametrize("kernel", KERNELS, ids=KERNEL_IDS)
def test_kernels_larger_than_the_image_still_match(shape, kernel):
    """A 7x7 kernel over a 2-pixel axis folds the reflection several times."""
    image = noise(shape, seed=hash(shape) % 1000)
    assert_identical(*both_backends(vpx_erode, image, kernel, 1))
    assert_identical(*both_backends(vpx_dilate, image, kernel, 1))


@pytest.mark.parametrize("density", (0.0, 0.05, 0.95, 1.0))
def test_saturated_and_empty_images_match(density):
    """All-background and all-foreground are the two fixed points."""
    image = noise((12, 12), seed=41, density=density)
    assert_identical(*both_backends(vpx_erode, image, kernel_cross(7), 2))
    assert_identical(*both_backends(vpx_dilate, image, kernel_cross(7), 2))


@pytest.mark.parametrize("kernel", KERNELS, ids=KERNEL_IDS)
def test_single_pixel_in_a_corner_matches(kernel):
    """The corner is where reflection padding differs the most from zero-fill."""
    image = np.zeros((9, 9), dtype=np.uint8)
    image[0, 0] = 255
    assert_identical(*both_backends(vpx_dilate, image, kernel, 1))
    assert_identical(*both_backends(vpx_erode, image, kernel, 1))


@pytest.mark.parametrize("kernel", KERNELS, ids=KERNEL_IDS)
@pytest.mark.parametrize("column", (0, 1, -2, -1))
def test_every_border_column_matches(kernel, column):
    """Una franja vertical pegada a cada borde, que es donde el padding manda.

    Separado del ruido general a propósito: con una imagen aleatoria densa, un
    error de borde se tapa solo porque el vecino también estaba encendido.
    """
    image = np.zeros((11, 11), dtype=np.uint8)
    image[:, column] = 255
    image[3, :] = 255
    assert_identical(*both_backends(vpx_erode, image, kernel, 1))
    assert_identical(*both_backends(vpx_dilate, image, kernel, 1))


def test_hitmiss_matches():
    image = noise((12, 12), seed=59)
    kernel_hit = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
    kernel_miss = np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]], dtype=np.uint8)
    assert_identical(*both_backends(vpx_hitmiss, image, kernel_hit, kernel_miss))


def test_reconstruct_matches():
    """Reconstruction loops until convergence, so a drift of one pixel diverges."""
    mask = noise((14, 14), seed=67, density=0.6)
    marker = np.zeros_like(mask)
    marker[mask > 0] = 0
    seeds = np.argwhere(mask > 0)[:3]
    for row, column in seeds:
        marker[row, column] = 255
    assert_identical(*both_backends(vpx_reconstruct, marker, mask))


def test_non_contiguous_input_matches():
    """A sliced view has strides the native side must not assume away."""
    image = noise((20, 20), seed=71)[::2, ::2]
    assert not image.flags["C_CONTIGUOUS"]
    assert_identical(*both_backends(vpx_erode, image, kernel_square(3), 1))


def test_validation_errors_still_come_from_python():
    """Error messages are public contract and must not change per backend."""
    with _backend.override("rust"):
        with pytest.raises(ValueError, match="kernel dimensions must be odd"):
            vpx_erode(noise((5, 5), seed=3), np.ones((2, 2), dtype=np.uint8))
        with pytest.raises(ValueError, match="image must not be empty"):
            vpx_erode(np.zeros((0, 0), dtype=np.uint8))
        with pytest.raises(ValueError, match="iterations must be a positive integer"):
            vpx_erode(noise((5, 5), seed=3), None, True)


def test_override_restores_the_previous_backend():
    before = _backend.name()
    with _backend.override("python"):
        assert _backend.name() == "python"
    assert _backend.name() == before


def test_unknown_backend_mode_is_rejected():
    with pytest.raises(ValueError, match="VISPYX_BACKEND must be one of"):
        with _backend.override("cuda"):
            pass
