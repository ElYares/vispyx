"""Tests for ``vispyx.preprocessing.apply_clahe``.

Antes habia dos: uno comprobaba la forma y el otro que la salida fuera un
``ndarray``, los dos sobre ruido **sin semilla fija**. Pasaban aunque
``apply_clahe`` devolviera la imagen sin tocar, e ignoraban los dos parametros y
el alias historico.

Las imagenes de aca se construyen, no se sortean: no hay aleatoriedad que sembrar.
"""

import numpy as np
import pytest

from vispyx.preprocessing import apply_clahe


def _imagen_de_dos_mitades():
    """Mitad con rango completo, mitad casi plana.

    CLAHE ecualiza por regiones, asi que hace falta una region de bajo contraste
    dentro de una imagen que no lo sea. Sobre un gradiente uniforme el efecto es
    chico y el test no distinguiria gran cosa.
    """
    imagen = np.zeros((64, 64), dtype=np.uint8)
    imagen[:32, :] = np.linspace(0, 255, 64, dtype=np.uint8)
    imagen[32:, :] = np.linspace(100, 115, 64, dtype=np.uint8)
    return imagen


def test_apply_clahe_expands_local_contrast():
    """Lo que CLAHE existe para hacer, y lo que ningun test afirmaba.

    La mitad plana va de un rango de 15 niveles a uno de 215. Un
    ``apply_clahe`` que devolviera la entrada sin tocar pasaba los dos tests
    viejos y falla este.
    """
    imagen = _imagen_de_dos_mitades()
    plana = imagen[32:, :]

    resultado = apply_clahe(imagen, clip_limit=40.0, tile_grid_size=(2, 2))

    rango_antes = int(plana.max()) - int(plana.min())
    rango_despues = int(resultado[32:, :].max()) - int(resultado[32:, :].min())
    assert rango_antes == 15
    assert rango_despues > 10 * rango_antes


def test_apply_clahe_preserves_shape_and_dtype():
    imagen = _imagen_de_dos_mitades()

    resultado = apply_clahe(imagen)

    assert isinstance(resultado, np.ndarray)
    assert resultado.shape == imagen.shape
    assert resultado.dtype == np.uint8


def test_apply_clahe_is_deterministic():
    """La version vieja sorteaba ruido sin semilla: era no determinista."""
    imagen = _imagen_de_dos_mitades()

    np.testing.assert_array_equal(apply_clahe(imagen), apply_clahe(imagen))


def test_clip_limit_reaches_opencv():
    """Sin esto, ignorar el parametro pasaria desapercibido."""
    imagen = _imagen_de_dos_mitades()

    suave = apply_clahe(imagen, clip_limit=1.0)
    fuerte = apply_clahe(imagen, clip_limit=40.0)

    assert not np.array_equal(suave, fuerte)


def test_tile_grid_size_reaches_opencv():
    imagen = _imagen_de_dos_mitades()

    gruesa = apply_clahe(imagen, tile_grid_size=(2, 2))
    fina = apply_clahe(imagen, tile_grid_size=(16, 16))

    assert not np.array_equal(gruesa, fina)


@pytest.mark.parametrize("grid", [(2, 2), (16, 16)])
def test_title_grid_size_is_the_historic_typo_alias(grid):
    """``title_grid_size`` es un typo que se conserva por compatibilidad.

    Vive en `preprocessing.py:14` con un comentario que lo llama temporal. Nada
    lo fijaba, asi que borrarlo o dejar de reenviarlo no rompia ningun test —
    y es superficie publica para quien ya lo usa.
    """
    imagen = _imagen_de_dos_mitades()

    np.testing.assert_array_equal(
        apply_clahe(imagen, title_grid_size=grid),
        apply_clahe(imagen, tile_grid_size=grid),
    )


def test_apply_clahe_accepts_uint16():
    """OpenCV implementa CLAHE para ``CV_8UC1`` y ``CV_16UC1``, no solo uint8.

    Verificado contra la version instalada: uint8 y uint16 pasan; ``int16``,
    ``int32``, ``float32`` y ``float64`` fallan dentro de ``clahe.cpp``. La
    validacion tiene que dejar pasar los dos que funcionan, no solo el comun.
    """
    imagen = (_imagen_de_dos_mitades().astype(np.uint16) * 257)

    resultado = apply_clahe(imagen)

    assert resultado.dtype == np.uint16
    assert resultado.shape == imagen.shape


def test_apply_clahe_rejects_non_2d_images():
    with pytest.raises(ValueError, match="image must be a 2D array"):
        apply_clahe(np.zeros((8, 8, 3), dtype=np.uint8))


def test_apply_clahe_rejects_non_numeric_images():
    with pytest.raises(ValueError, match="image must contain numeric values"):
        apply_clahe(np.array([["a", "b"], ["c", "d"]]))


@pytest.mark.parametrize("dtype", [np.float64, np.float32, np.int16, np.int32])
def test_apply_clahe_rejects_dtypes_that_opencv_cannot_handle(dtype):
    """Antes salian como ``cv2.error`` de ``clahe.cpp``, no como ``ValueError``.

    Es la misma fuga que se cerro en el CLI con ``cv2.imwrite``: una excepcion
    de la dependencia llegando cruda a quien llamo a ``vispyx``.
    """
    with pytest.raises(ValueError, match="image must be uint8 or uint16"):
        apply_clahe(np.zeros((8, 8), dtype=dtype))


def test_apply_clahe_accepts_nested_lists():
    """``validate_grayscale_image`` normaliza con ``np.asarray`` antes de mirar.

    Una lista anidada de enteros llega como ``int64`` y se rechaza por dtype, no
    con el ``AttributeError`` o el ``cv2.error`` que salia antes.
    """
    with pytest.raises(ValueError, match="image must be uint8 or uint16"):
        apply_clahe([[0, 1], [1, 0]])
