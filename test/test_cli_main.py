"""Tests for ``vispyx.cli.main``: the argument parsing layer.

``test_cli.py`` cubre tres funciones ``run_*``; esto cubre lo que las rodea —
el parser, las flags, el guardado y los codigos de salida — que es lo que un
usuario toca de verdad.
"""

import sys

import cv2
import numpy as np
import pytest

from vispyx import cli

METODOS = [
    "clahe",
    "otsu",
    "vpx_erode",
    "vpx_dilate",
    "vpx_open",
    "vpx_close",
    "vpx_gradient",
    "vpx_reconstruct",
    "vpx_skeletonize",
    "vpx_thin",
    "gray_erode",
    "gray_dilate",
    "gray_open",
    "gray_close",
    "gray_gradient",
    "gray_tophat",
    "gray_blackhat",
]


@pytest.fixture
def imagen(tmp_path):
    """Un bloque claro con una mota de ruido, escrito a disco."""
    datos = np.zeros((16, 16), dtype=np.uint8)
    datos[4:12, 4:12] = 200
    datos[1, 1] = 255
    ruta = tmp_path / "entrada.pgm"
    cv2.imwrite(str(ruta), datos)
    return str(ruta)


@pytest.fixture
def mascara(tmp_path):
    """Una mascara que contiene al marcador, para ``vpx_reconstruct``."""
    datos = np.zeros((16, 16), dtype=np.uint8)
    datos[4:12, 4:12] = 255
    ruta = tmp_path / "mascara.pgm"
    cv2.imwrite(str(ruta), datos)
    return str(ruta)


def _correr(monkeypatch, argumentos):
    monkeypatch.setattr(sys, "argv", ["vispyx"] + argumentos)
    cli.main()


def _argumentos(metodo, imagen, mascara, salida):
    """Los argumentos minimos para que un metodo corra."""
    if metodo == "vpx_reconstruct":
        # el posicional es el marker: un pixel dentro de la mascara
        return [metodo, imagen, "--mask", mascara, "-o", salida]
    return [metodo, imagen, "-o", salida]


@pytest.mark.parametrize("metodo", METODOS)
def test_cada_metodo_corre_y_guarda(metodo, imagen, mascara, tmp_path, monkeypatch, capsys):
    """Los 17 metodos del parser tienen que estar realmente conectados."""
    salida = str(tmp_path / f"{metodo}.pgm")

    if metodo == "vpx_reconstruct":
        marcador = np.zeros((16, 16), dtype=np.uint8)
        marcador[8, 8] = 255
        imagen = str(tmp_path / "marcador.pgm")
        cv2.imwrite(imagen, marcador)

    _correr(monkeypatch, _argumentos(metodo, imagen, mascara, salida))

    assert f"Imagen guardada en: {salida}" in capsys.readouterr().out
    resultado = cv2.imread(salida, cv2.IMREAD_GRAYSCALE)
    assert resultado is not None
    assert resultado.shape == (16, 16)


def test_output_crea_los_directorios_que_faltan(imagen, tmp_path, monkeypatch):
    salida = tmp_path / "sub" / "otro" / "resultado.pgm"
    assert not salida.parent.exists()

    _correr(monkeypatch, ["vpx_open", imagen, "-o", str(salida)])

    assert salida.exists()


def test_sin_output_no_escribe_nada_y_lo_dice(imagen, tmp_path, monkeypatch, capsys):
    antes = set(tmp_path.iterdir())

    _correr(monkeypatch, ["vpx_erode", imagen])

    assert "Imagen procesada. No se guardó." in capsys.readouterr().out
    assert set(tmp_path.iterdir()) == antes


def test_reconstruct_sin_mask_sale_con_codigo_2(imagen, monkeypatch, capsys):
    with pytest.raises(SystemExit) as salida:
        _correr(monkeypatch, ["vpx_reconstruct", imagen])

    assert salida.value.code == 2
    assert "--mask es obligatorio para vpx_reconstruct" in capsys.readouterr().err


def test_metodo_desconocido_sale_con_codigo_2(imagen, monkeypatch, capsys):
    with pytest.raises(SystemExit) as salida:
        _correr(monkeypatch, ["no_existe", imagen])

    assert salida.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


def test_kernel_par_es_rechazado(imagen, monkeypatch):
    """El CLI no valida la paridad: el error viene de la capa morfologica."""
    with pytest.raises(ValueError, match="kernel dimensions must be odd"):
        _correr(monkeypatch, ["vpx_erode", imagen, "--kernel-size", "4"])


def test_kernel_size_no_positivo_es_rechazado(imagen, monkeypatch):
    with pytest.raises(ValueError, match="--kernel-size debe ser un entero positivo"):
        _correr(monkeypatch, ["vpx_erode", imagen, "--kernel-size", "0"])


def test_iterations_cero_es_rechazado(imagen, monkeypatch):
    with pytest.raises(ValueError, match="iterations must be a positive integer"):
        _correr(monkeypatch, ["vpx_erode", imagen, "--iterations", "0"])


def test_kernel_es_alias_de_kernel_size(imagen, tmp_path, monkeypatch):
    """Comparten ``dest``; el default efectivo sigue siendo 3."""
    con_alias = str(tmp_path / "alias.pgm")
    con_largo = str(tmp_path / "largo.pgm")

    _correr(monkeypatch, ["vpx_erode", imagen, "--kernel", "5", "-o", con_alias])
    _correr(monkeypatch, ["vpx_erode", imagen, "--kernel-size", "5", "-o", con_largo])

    np.testing.assert_array_equal(
        cv2.imread(con_alias, cv2.IMREAD_GRAYSCALE),
        cv2.imread(con_largo, cv2.IMREAD_GRAYSCALE),
    )


def test_imagen_inexistente_falla_con_el_error_del_paquete(tmp_path, monkeypatch):
    """El CLI usa ``read_grayscale``, asi que el error es el mismo que en Python."""
    with pytest.raises(FileNotFoundError, match="No se encontró la imagen"):
        _correr(monkeypatch, ["clahe", str(tmp_path / "no-existe.pgm")])


def test_imagen_ilegible_se_distingue_de_inexistente(tmp_path, monkeypatch):
    ruta = tmp_path / "basura.pgm"
    ruta.write_text("esto no es una imagen")

    with pytest.raises(ValueError, match="No se pudo decodificar la imagen"):
        _correr(monkeypatch, ["clahe", str(ruta)])


def test_clahe_respeta_clip_y_grid(imagen, tmp_path, monkeypatch):
    """Dos configuraciones distintas de CLAHE no pueden dar lo mismo."""
    suave = str(tmp_path / "suave.pgm")
    fuerte = str(tmp_path / "fuerte.pgm")

    _correr(monkeypatch, ["clahe", imagen, "--clip", "1.0", "--grid", "2", "-o", suave])
    _correr(monkeypatch, ["clahe", imagen, "--clip", "40.0", "--grid", "8", "-o", fuerte])

    assert not np.array_equal(
        cv2.imread(suave, cv2.IMREAD_GRAYSCALE),
        cv2.imread(fuerte, cv2.IMREAD_GRAYSCALE),
    )


def test_vpx_binariza_la_entrada(imagen, tmp_path, monkeypatch):
    """Los metodos ``vpx_*`` devuelven 0/255 aunque la entrada sea de grises."""
    salida = str(tmp_path / "binario.pgm")

    _correr(monkeypatch, ["vpx_dilate", imagen, "-o", salida])

    resultado = cv2.imread(salida, cv2.IMREAD_GRAYSCALE)
    assert set(np.unique(resultado)) <= {0, 255}


def test_gray_no_binariza_la_entrada(imagen, tmp_path, monkeypatch):
    """Los ``gray_*`` conservan los valores intermedios."""
    salida = str(tmp_path / "gris.pgm")

    _correr(monkeypatch, ["gray_erode", imagen, "-o", salida])

    resultado = cv2.imread(salida, cv2.IMREAD_GRAYSCALE)
    assert 200 in np.unique(resultado)
