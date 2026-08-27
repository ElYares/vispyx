"""Tests for the I/O helpers in ``vispyx.utils``.

``read_grayscale`` is the single entry point for reading images: the CLI uses
this same function, so the error a user sees is identical whether they came in
through Python or through the command line.
"""

import cv2
import matplotlib
import numpy as np
import pytest

from vispyx import read_grayscale, show_image


def _escribe_imagen(ruta, valor=128, tamano=(8, 8)):
    cv2.imwrite(str(ruta), np.full(tamano, valor, dtype=np.uint8))
    return str(ruta)


def test_read_grayscale_devuelve_un_array_2d_uint8(tmp_path):
    ruta = _escribe_imagen(tmp_path / "imagen.pgm")

    imagen = read_grayscale(ruta)

    assert isinstance(imagen, np.ndarray)
    assert imagen.ndim == 2
    assert imagen.dtype == np.uint8
    assert imagen.shape == (8, 8)


def test_read_grayscale_convierte_color_a_un_solo_canal(tmp_path):
    ruta = str(tmp_path / "color.png")
    cv2.imwrite(ruta, np.zeros((8, 8, 3), dtype=np.uint8))

    assert read_grayscale(ruta).ndim == 2


def test_read_grayscale_falla_ruidosamente_si_no_existe(tmp_path):
    ruta = str(tmp_path / "no-existe.pgm")

    with pytest.raises(FileNotFoundError, match="No se encontró la imagen"):
        read_grayscale(ruta)


def test_el_error_incluye_la_ruta(tmp_path):
    """Sin la ruta en el mensaje, el error no dice cuál de las lecturas falló."""
    ruta = str(tmp_path / "no-existe.pgm")

    with pytest.raises(FileNotFoundError) as excepcion:
        read_grayscale(ruta)

    assert ruta in str(excepcion.value)


def test_read_grayscale_distingue_ilegible_de_inexistente(tmp_path):
    """``cv2.imread`` devuelve None en los dos casos; el contrato los separa."""
    ruta = tmp_path / "basura.pgm"
    ruta.write_text("esto no es una imagen")

    with pytest.raises(ValueError, match="No se pudo decodificar la imagen"):
        read_grayscale(str(ruta))


def test_un_directorio_no_es_una_imagen_legible(tmp_path):
    with pytest.raises(ValueError, match="No se pudo decodificar la imagen"):
        read_grayscale(str(tmp_path))


def test_read_grayscale_nunca_devuelve_none(tmp_path):
    """El contrato que esta historia vino a cambiar: antes devolvía None callado."""
    with pytest.raises((FileNotFoundError, ValueError)):
        read_grayscale(str(tmp_path / "ausente.pgm"))


def test_el_cli_usa_el_mismo_lector(tmp_path):
    """Un mismo fallo tiene que dar el mismo error por Python y por la CLI."""
    from vispyx import cli

    assert cli.read_grayscale is read_grayscale
    assert not hasattr(cli, "_read_grayscale")

    with pytest.raises(FileNotFoundError, match="No se encontró la imagen"):
        cli.run_vpx_skeletonize(str(tmp_path / "no-existe.pgm"))


def test_show_image_dibuja_sin_backend_interactivo(tmp_path, monkeypatch):
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mostradas = []
    monkeypatch.setattr(plt, "show", lambda *a, **k: mostradas.append(True))

    assert show_image(np.zeros((4, 4), dtype=np.uint8), title="prueba") is None
    assert mostradas == [True]
    plt.close("all")
