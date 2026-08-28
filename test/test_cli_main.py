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
from vispyx.kernels import kernel_cross, kernel_diamond, kernel_disk, kernel_square

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


def test_directorio_de_salida_que_no_se_puede_crear(imagen, tmp_path, monkeypatch, capsys):
    """``os.makedirs`` corre antes de escribir y tambien puede fallar.

    Sin atraparlo, un ``--output`` bajo un padre que no es directorio salia como
    traceback de ``os``, no como error del CLI. Se usa un archivo como padre y
    no un permiso porque ``chmod`` no frena a root, y la suite puede correr ahi.
    """
    padre = tmp_path / "soy_un_archivo"
    padre.write_text("no soy un directorio")
    salida = padre / "sub" / "resultado.pgm"

    with pytest.raises(SystemExit) as fallo:
        _correr(monkeypatch, ["vpx_erode", imagen, "-o", str(salida)])

    assert fallo.value.code == 2
    capturado = capsys.readouterr()
    assert f"No se pudo crear el directorio {salida.parent}" in capturado.err
    assert "Imagen guardada" not in capturado.out


def test_guardado_fallido_no_dice_que_guardo(imagen, tmp_path, monkeypatch, capsys):
    """``cv2.imwrite`` devuelve ``False`` en vez de lanzar, y eso se ignoraba.

    El CLI imprimia "Imagen guardada en: ..." y salia con codigo 0 sin que
    existiera el archivo. Se fuerza el ``False`` directamente porque cuando
    OpenCV lo devuelve depende de la version: en 5.0 una extension desconocida
    lanza, y en 4.x devolvia ``False``. El contrato del CLI no debe depender de
    cual de las dos ocurra.
    """
    salida = tmp_path / "resultado.pgm"
    monkeypatch.setattr(cli.cv2, "imwrite", lambda *_: False)

    with pytest.raises(SystemExit) as fallo:
        _correr(monkeypatch, ["vpx_erode", imagen, "-o", str(salida)])

    assert fallo.value.code == 2
    capturado = capsys.readouterr()
    assert f"No se pudo guardar la imagen en {salida}" in capturado.err
    assert "Imagen guardada" not in capturado.out


def test_output_que_es_un_directorio_falla(imagen, tmp_path, monkeypatch, capsys):
    """El camino real que devuelve ``False``: la ruta existe y no es un archivo."""
    salida = tmp_path / "ya_es_un_directorio.pgm"
    salida.mkdir()

    with pytest.raises(SystemExit) as fallo:
        _correr(monkeypatch, ["vpx_erode", imagen, "-o", str(salida)])

    assert fallo.value.code == 2
    capturado = capsys.readouterr()
    assert "no se pudo abrir el archivo para escritura" in capturado.err
    assert "Imagen guardada" not in capturado.out


def test_extension_desconocida_no_escupe_traceback(imagen, tmp_path, monkeypatch, capsys):
    """OpenCV elige el codec por la extension y lanza ``cv2.error`` si no tiene.

    Sin atrapar eso, el usuario recibia el traceback de ``loadsave.cpp`` y un
    codigo de salida 1.
    """
    salida = tmp_path / "resultado.txt"

    with pytest.raises(SystemExit) as fallo:
        _correr(monkeypatch, ["vpx_erode", imagen, "-o", str(salida)])

    assert fallo.value.code == 2
    assert "OpenCV no reconoce la extension" in capsys.readouterr().err
    assert not salida.exists()


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
    """Desde ``--kernel-shape`` la paridad la valida ``kernels.py``.

    Antes el error salia de ``validate_kernel``, ya dentro de la operacion y
    despues de leer la imagen: ``kernel dimensions must be odd``. Ahora
    ``_build_kernel`` delega en los generadores y falla antes de leer nada.
    """
    with pytest.raises(ValueError, match="size must be odd"):
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


# --- forma del kernel (--kernel-shape) ---------------------------------------
#
# Las formas coinciden entre si para radios chicos, asi que el tamano del test
# decide que puede distinguir:
#
#   size=3  cross == diamond == disk  ->  no distingue nada
#   size=5  diamond == disk           ->  separa cross, no separa diamond/disk
#   size=7  las cuatro difieren       ->  el unico que las separa a todas
#
# Por eso el caso base va con 5 y la separacion completa con 7.


def _kernel_usado(monkeypatch, argumentos):
    """Corre el CLI y devuelve el kernel que realmente llego a la operacion."""
    capturado = {}

    def espia(imagen, kernel, iterations):
        capturado["kernel"] = kernel
        return imagen

    monkeypatch.setattr(cli, "vpx_erode", espia)
    _correr(monkeypatch, argumentos)
    return capturado["kernel"]


def test_kernel_shape_omitido_construye_el_cuadrado_de_siempre(imagen, monkeypatch):
    kernel = _kernel_usado(monkeypatch, ["vpx_erode", imagen, "--kernel-size", "5"])

    np.testing.assert_array_equal(kernel, np.ones((5, 5), dtype=np.uint8))


@pytest.mark.parametrize(
    "forma, esperado",
    [
        ("square", kernel_square(5)),
        ("cross", kernel_cross(5)),
        ("diamond", kernel_diamond(5)),
        ("disk", kernel_disk(2)),
    ],
)
def test_cada_forma_construye_su_generador(forma, esperado, imagen, monkeypatch):
    kernel = _kernel_usado(
        monkeypatch,
        ["vpx_erode", imagen, "--kernel-size", "5", "--kernel-shape", forma],
    )

    np.testing.assert_array_equal(kernel, esperado)


def test_diamante_y_disco_no_son_la_misma_forma(imagen, monkeypatch):
    """Con ``size=5`` coinciden; recien en 7 se separan.

    Sin este caso, cambiar la rama ``diamond`` por ``kernel_disk(size // 2)``
    pasa el resto de la suite sin una sola falla.
    """
    diamante = _kernel_usado(
        monkeypatch,
        ["vpx_erode", imagen, "--kernel-size", "7", "--kernel-shape", "diamond"],
    )
    disco = _kernel_usado(
        monkeypatch,
        ["vpx_erode", imagen, "--kernel-size", "7", "--kernel-shape", "disk"],
    )

    np.testing.assert_array_equal(diamante, kernel_diamond(7))
    np.testing.assert_array_equal(disco, kernel_disk(3))
    assert not np.array_equal(diamante, disco)


def test_el_disco_deriva_el_radio_del_tamano(imagen, monkeypatch):
    """``--kernel-size 5`` es ``kernel_disk(2)``: radio = size // 2, lado 5."""
    kernel = _kernel_usado(
        monkeypatch,
        ["vpx_erode", imagen, "--kernel-size", "5", "--kernel-shape", "disk"],
    )

    assert kernel.shape == (5, 5)


def test_la_forma_llega_a_los_gray(tmp_path, monkeypatch):
    """La flag no es solo de los ``vpx_*``: los ``gray_*`` tambien la reciben.

    El fixture ``imagen`` no sirve aca: sobre un bloque cuadrado solido las
    cuatro formas erosionan identico. Con un unico pixel oscuro, el cuadrado
    lo propaga a sus 25 vecinos y la cruz solo a 9.
    """
    datos = np.full((16, 16), 200, dtype=np.uint8)
    datos[8, 8] = 0
    entrada = str(tmp_path / "punto.pgm")
    cv2.imwrite(entrada, datos)

    oscuros = {}
    for forma in ("square", "cross"):
        salida = str(tmp_path / f"{forma}.pgm")
        _correr(
            monkeypatch,
            ["gray_erode", entrada, "--kernel-size", "5", "--kernel-shape", forma, "-o", salida],
        )
        oscuros[forma] = int((cv2.imread(salida, cv2.IMREAD_GRAYSCALE) == 0).sum())

    assert oscuros["square"] == 25
    assert oscuros["cross"] == 9


@pytest.mark.parametrize("forma", ["square", "cross", "diamond", "disk"])
def test_ninguna_forma_acepta_un_tamano_par(forma, imagen, monkeypatch):
    """El disco tambien: sin la validacion, size=4 daria el mismo disco que 5."""
    with pytest.raises(ValueError, match="size must be odd"):
        _correr(
            monkeypatch,
            ["vpx_erode", imagen, "--kernel-size", "4", "--kernel-shape", forma],
        )


def test_forma_desconocida_sale_con_codigo_2(imagen, monkeypatch, capsys):
    with pytest.raises(SystemExit) as salida:
        _correr(
            monkeypatch,
            ["vpx_erode", imagen, "--kernel-shape", "triangulo"],
        )

    assert salida.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


def test_build_kernel_rechaza_una_forma_que_el_parser_no_filtro():
    """``_build_kernel`` es API interna: no puede confiar en el ``choices``."""
    with pytest.raises(ValueError, match="Forma de kernel no reconocida: triangulo"):
        cli._build_kernel(5, "triangulo")
