"""Tests for ``vispyx.segmentation.segment_otsu``.

Era el unico modulo del paquete **sin un solo test funcional**: `segment_otsu`
aparecia nombrada dentro de `expected_symbols` y nunca se invocaba. La cobertura
de lineas decia 100% porque el test de punta a punta del CLI corre `otsu`, pero
nada afirmaba que produjera. Medir ejecucion no es medir verificacion.

Es el puente entre los dos dominios de valores del paquete, asi que lo que fija
es justamente eso: recibe grises y devuelve la mascara `{0, 255}` que las
`vpx_*` esperan.
"""

import numpy as np
import pytest

from skimage.filters import threshold_otsu

from vispyx.segmentation import segment_otsu


def _imagen_de_tres_niveles():
    """Fondo 30, region media 90, objeto 200.

    **Tres niveles y no dos, a proposito.** Con una imagen de dos niveles que
    incluya el 0, el umbral de Otsu **da 0**, y entonces `segment_otsu` es
    indistinguible de un simple `image > 0`: el test pasaria sin probar que se
    calcule ningun umbral. Aca el umbral cae en 30, estrictamente adentro del
    rango, y los dos resultados difieren.
    """
    imagen = np.full((8, 8), 30, dtype=np.uint8)
    imagen[1:7, 1:7] = 90
    imagen[3:5, 3:5] = 200
    return imagen


def test_segment_otsu_umbrala_donde_lo_dice_otsu():
    imagen = _imagen_de_tres_niveles()
    umbral = threshold_otsu(imagen)

    resultado = segment_otsu(imagen)

    assert 0 < umbral < 200, "el umbral tiene que ser interior para que el test valga"
    np.testing.assert_array_equal(resultado, (imagen > umbral).astype(np.uint8) * 255)


def test_segment_otsu_no_es_lo_mismo_que_binarizar_con_cero():
    """La confusion que el test debe poder detectar.

    Si `segment_otsu` ignorara el umbral y binarizara con `> 0`, seria
    indistinguible sobre cualquier imagen cuyo fondo sea negro puro. Con fondo
    en 30 no lo es.
    """
    imagen = _imagen_de_tres_niveles()

    assert not np.array_equal(segment_otsu(imagen), (imagen > 0).astype(np.uint8) * 255)


def test_segment_otsu_devuelve_la_convencion_binaria_del_paquete():
    """Es el puente hacia las `vpx_*`, que exigen `uint8` en `{0, 255}`."""
    resultado = segment_otsu(_imagen_de_tres_niveles())

    assert resultado.dtype == np.uint8
    assert set(np.unique(resultado).tolist()) <= {0, 255}
    assert resultado.shape == (8, 8)


def test_segment_otsu_alimenta_a_las_vpx_sin_traduccion():
    """El puente, ejercido de punta a punta."""
    from vispyx import vpx_erode

    mascara = segment_otsu(_imagen_de_tres_niveles())
    erosionada = vpx_erode(mascara)

    assert erosionada.dtype == np.uint8
    assert set(np.unique(erosionada).tolist()) <= {0, 255}
    assert np.count_nonzero(erosionada) < np.count_nonzero(mascara)


def test_una_imagen_uniforme_da_una_mascara_vacia():
    """Trampa documentada, no bug.

    Con una imagen constante el umbral de Otsu es **esa misma constante**, y
    `image > thresh` no deja nada. Sale una mascara toda negra, sin warning ni
    error. Queda fijado para que nadie lo tome por una falla intermitente.
    """
    resultado = segment_otsu(np.full((8, 8), 128, dtype=np.uint8))

    assert np.count_nonzero(resultado) == 0


def test_segment_otsu_rechaza_imagenes_que_no_son_2d():
    """Antes **no fallaba**: aceptaba RGB, emitia un `UserWarning` de skimage que
    nadie ve, y devolvia un resultado sin sentido. Una respuesta equivocada en
    silencio es peor que una excepcion."""
    with pytest.raises(ValueError, match="image must be a 2D array"):
        segment_otsu(np.zeros((8, 8, 3), dtype=np.uint8))


def test_segment_otsu_rechaza_valores_no_numericos():
    """Antes salia como `UFuncTypeError` de numpy, no como `ValueError`."""
    with pytest.raises(ValueError, match="image must contain numeric values"):
        segment_otsu(np.array([["a", "b"], ["c", "d"]]))


def test_segment_otsu_acepta_listas_anidadas():
    """Antes lanzaba `AttributeError: 'list' object has no attribute 'ndim'`.

    Es la misma aspereza que `0.2.1` corrigio para las `gray_*`: el cast iba
    contra el argumento crudo y no contra el array validado.
    """
    resultado = segment_otsu([[0, 0, 200], [0, 200, 200], [0, 0, 0]])

    assert resultado.dtype == np.uint8
    assert set(np.unique(resultado).tolist()) <= {0, 255}
