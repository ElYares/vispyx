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
from vispyx import vpx_blackhat, vpx_boundary, vpx_tophat
from vispyx.cli import METHODS, PATTERN_NAMES
from vispyx.kernels import kernel_cross, kernel_diamond, kernel_disk, kernel_square

METODOS = [
    "clahe",
    "otsu",
    "vpx_erode",
    "vpx_dilate",
    "vpx_open",
    "vpx_close",
    "vpx_gradient",
    "vpx_tophat",
    "vpx_blackhat",
    "vpx_boundary",
    "vpx_hitmiss",
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
    if metodo == "vpx_hitmiss":
        return [metodo, imagen, "--pattern", "corner", "-o", salida]
    return [metodo, imagen, "-o", salida]


@pytest.mark.parametrize("metodo", METODOS)
def test_cada_metodo_corre_y_guarda(metodo, imagen, mascara, tmp_path, monkeypatch, capsys):
    """Los 21 metodos del parser tienen que estar realmente conectados."""
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


@pytest.mark.parametrize(
    "metodo,operacion",
    [("vpx_tophat", vpx_tophat), ("vpx_blackhat", vpx_blackhat), ("vpx_boundary", vpx_boundary)],
)
def test_las_binarias_nuevas_despachan_a_su_operacion(
    metodo, operacion, imagen, tmp_path, monkeypatch
):
    """Que el metodo corra no prueba que llame a la operacion correcta.

    ``test_cada_metodo_corre_y_guarda`` pasaria igual con las tres ramas de
    despacho permutadas entre si. Esto compara contra la funcion de la libreria
    sobre la misma entrada binarizada, asi que solo pasa la rama que corresponde.
    Sobre el fixture las tres dan resultados distintos — 1, 3 y 29 pixeles
    activos — que es lo que hace la comparacion capaz de distinguirlas.
    """
    salida = tmp_path / f"{metodo}.pgm"

    _correr(monkeypatch, [metodo, imagen, "--kernel-size", "3", "-o", str(salida)])

    entrada = cv2.imread(imagen, cv2.IMREAD_GRAYSCALE)
    binaria = (entrada > 0).astype(np.uint8) * 255
    esperado = operacion(binaria, kernel_square(3), 1)

    np.testing.assert_array_equal(
        cv2.imread(str(salida), cv2.IMREAD_GRAYSCALE), esperado
    )


def test_la_lista_de_metodos_del_cli_esta_toda_cubierta():
    """Sin esto, agregar un metodo al CLI lo deja sin cobertura en silencio.

    ``METODOS`` es una lista escrita a mano y ``cli.METHODS`` es la real. Nada
    las comparaba: al agregar ``vpx_hitmiss`` la suite siguio verde con el
    metodo nuevo sin un solo test. La parametrizacion es tan buena como esta
    lista.
    """
    assert sorted(METODOS) == sorted(METHODS)


PATRONES_ESPERADOS = {
    "corner": [(3, 3), (3, 7), (7, 3), (7, 7)],
    "corner-nw": [(3, 3)],
    "corner-ne": [(3, 7)],
    "corner-se": [(7, 7)],
    "corner-sw": [(7, 3)],
    "isolated": [(10, 10)],
}


@pytest.fixture
def imagen_con_esquinas(tmp_path):
    """Bloque solido, un pixel suelto de verdad, y un par que se toca en diagonal.

    **El par diagonal no es decorado.** Sin el, recortar el `miss` de `isolated`
    de los 8 vecinos a los 4 ortogonales **pasa la suite**: un pixel sin ningun
    vecino se detecta igual con las dos versiones. Con el par en (11,1) y (12,2)
    la mutacion los reporta como aislados, y ahi falla.

    Medido: el par no agrega detecciones a ninguno de los patrones de esquina.
    """
    datos = np.zeros((14, 14), dtype=np.uint8)
    datos[3:8, 3:8] = 255
    datos[10, 10] = 255
    datos[11, 1] = 255
    datos[12, 2] = 255
    ruta = tmp_path / "esquinas.pgm"
    cv2.imwrite(str(ruta), datos)
    return str(ruta)


@pytest.mark.parametrize("patron", sorted(PATRONES_ESPERADOS))
def test_cada_patron_detecta_exactamente_lo_suyo(
    patron, imagen_con_esquinas, tmp_path, monkeypatch
):
    """Coordenadas exactas, no conteos.

    Un test que solo contara pixeles activos dejaria pasar `corner-ne` y
    `corner-sw` intercambiados: los dos detectan un pixel. Y intercambiar el
    `hit` con el `miss` de un patron tampoco da una imagen vacia — solo la
    posicion los separa.
    """
    salida = tmp_path / f"{patron}.pgm"

    _correr(monkeypatch, ["vpx_hitmiss", imagen_con_esquinas, "--pattern", patron, "-o", str(salida)])

    resultado = cv2.imread(str(salida), cv2.IMREAD_GRAYSCALE)
    ys, xs = np.nonzero(resultado)
    assert sorted(zip(ys.tolist(), xs.tolist())) == PATRONES_ESPERADOS[patron]
    assert resultado.dtype == np.uint8
    assert set(np.unique(resultado).tolist()) <= {0, 255}


def test_corner_es_la_union_de_las_cuatro_orientaciones(
    imagen_con_esquinas, tmp_path, monkeypatch
):
    """`corner` no es un par hit/miss: compone las cuatro. Si devolviera una
    sola orientacion seguiria dando una imagen valida, con un pixel."""
    salidas = {}
    for patron in ("corner", "corner-nw", "corner-ne", "corner-se", "corner-sw"):
        ruta = tmp_path / f"u-{patron}.pgm"
        _correr(monkeypatch, ["vpx_hitmiss", imagen_con_esquinas, "--pattern", patron, "-o", str(ruta)])
        salidas[patron] = cv2.imread(str(ruta), cv2.IMREAD_GRAYSCALE)

    union = np.zeros_like(salidas["corner"])
    for patron in ("corner-nw", "corner-ne", "corner-se", "corner-sw"):
        union = np.maximum(union, salidas[patron])

    np.testing.assert_array_equal(salidas["corner"], union)


def test_hitmiss_sin_pattern_sale_con_codigo_2(imagen, monkeypatch, capsys):
    with pytest.raises(SystemExit) as fallo:
        _correr(monkeypatch, ["vpx_hitmiss", imagen])

    assert fallo.value.code == 2
    assert "--pattern es obligatorio para vpx_hitmiss" in capsys.readouterr().err


def test_pattern_desconocido_sale_con_codigo_2(imagen, monkeypatch, capsys):
    with pytest.raises(SystemExit) as fallo:
        _correr(monkeypatch, ["vpx_hitmiss", imagen, "--pattern", "triangulo"])

    assert fallo.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


def test_el_catalogo_de_patrones_es_valido_entero():
    """Un par mal escrito tiene que fallar al agregarlo, no al usarlo.

    Las tres reglas de ``validate_hitmiss_kernels``: misma forma, sin
    solapamiento, y cada uno con al menos un elemento activo.
    """
    from vispyx.cli import PATTERNS

    assert sorted(PATTERN_NAMES) == sorted(["corner"] + list(PATTERNS))
    for nombre, (hit, miss) in PATTERNS.items():
        assert hit.shape == miss.shape, nombre
        assert not np.any((hit > 0) & (miss > 0)), nombre
        assert hit.any() and miss.any(), nombre


def test_hitmiss_ignora_las_flags_de_kernel(imagen_con_esquinas, tmp_path, monkeypatch):
    """`vpx_hitmiss` no toma kernel ni iterations, igual que `vpx_skeletonize`
    ya ignora `--kernel-size`. Pasarlas no debe fallar ni cambiar el resultado."""
    con, sin = tmp_path / "con.pgm", tmp_path / "sin.pgm"

    _correr(monkeypatch, ["vpx_hitmiss", imagen_con_esquinas, "--pattern", "isolated", "-o", str(sin)])
    _correr(monkeypatch, ["vpx_hitmiss", imagen_con_esquinas, "--pattern", "isolated",
                          "--kernel-size", "7", "--kernel-shape", "disk", "--iterations", "3", "-o", str(con)])

    np.testing.assert_array_equal(
        cv2.imread(str(sin), cv2.IMREAD_GRAYSCALE), cv2.imread(str(con), cv2.IMREAD_GRAYSCALE)
    )


def test_show_llama_al_unico_show_image_del_paquete(imagen, monkeypatch):
    """``--show`` era el unico camino del CLI sin cubrir.

    No se puede ejercitar de verdad: `cli.py` fuerza el backend `TkAgg` al
    importar y hace falta display. Lo que si se puede fijar es que despache al
    `show_image` de `utils.py` — antes `cli.py` tenia **su propia copia**, que
    la cobertura mostraba con cero ejecuciones.
    """
    llamadas = []
    monkeypatch.setattr(cli, "show_image", lambda img, **kw: llamadas.append((img, kw)))

    _correr(monkeypatch, ["vpx_erode", imagen, "--show"])

    assert len(llamadas) == 1
    img, kw = llamadas[0]
    assert kw == {"title": "vpx_erode", "figsize": (8, 6)}
    assert img.shape == (16, 16)


def test_show_image_es_el_mismo_objeto_en_cli_y_en_utils():
    """La duplicacion que este test impide que vuelva."""
    from vispyx.utils import show_image as desde_utils

    assert cli.show_image is desde_utils


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


def test_run_vpx_hitmiss_rechaza_un_patron_que_el_parser_no_filtro(imagen):
    """Mismo caso que ``_build_kernel``: API interna, no puede confiar en el
    ``choices`` de argparse. Es la unica forma de alcanzar esa rama."""
    with pytest.raises(ValueError, match="Patron no reconocido: espiral"):
        cli.run_vpx_hitmiss(imagen, "espiral")


def test_un_metodo_en_la_lista_sin_rama_de_despacho_falla(imagen, monkeypatch):
    """La red que `HU-003` estuvo a punto de borrar por "codigo muerto".

    La rama ``else`` de ``main()`` es inalcanzable desde ``argparse``, pero no
    es muerta: protege contra agregar un metodo a ``METHODS`` y olvidar su rama
    de despacho. Sin ella ese olvido seria un ``NameError`` sobre ``result``,
    que no dice nada. Se alcanza inyectando el metodo en la lista, que es
    exactamente el escenario del que protege.
    """
    monkeypatch.setattr(cli, "METHODS", cli.METHODS + ["vpx_inventado"])

    with pytest.raises(ValueError, match="Método no reconocido: vpx_inventado"):
        _correr(monkeypatch, ["vpx_inventado", imagen])
