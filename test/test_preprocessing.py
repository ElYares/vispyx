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
