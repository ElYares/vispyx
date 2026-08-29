"""Casos limite del motor: entradas degeneradas y caminos que nadie recorria.

`docs/testing.md` los listaba como "casos limite sin probar": imagenes vacias o
de un pixel, kernels mas grandes que la imagen, kernels no cuadrados,
`max_iterations` explicito e `iterations > 2`.

Escribirlos encontro un bug real. Ver la clase `TestImagenVacia`.
"""

import numpy as np
import pytest

from vispyx import (
    apply_clahe,
    gray_erode,
    segment_otsu,
    vpx_dilate,
    vpx_erode,
    vpx_reconstruct,
    vpx_skeletonize,
    vpx_thin,
)


class TestImagenVacia:
    """Una imagen vacia se comportaba de **cuatro maneras distintas**.

    Medido antes de arreglarlo, todo con `np.zeros((0, 0), np.uint8)`:

    | Funcion | Antes |
    |---|---|
    | las 17 `vpx_*` y `gray_*` con padding por reflejo | `ValueError`, pero con el mensaje interno de numpy: `can't extend empty axis 0 using modes other than 'constant'` |
    | `vpx_skeletonize`, `vpx_thin` | **no fallaban**: devolvian `(0, 0)`, porque Zhang-Suen usa padding de ceros y numpy si acepta eso |
    | `segment_otsu` | `IndexError` desde skimage |
    | `apply_clahe` | **devolvia `None`** con `(0, 0)`, y **se colgaba** con `(0, 5)` o `(5, 0)` |

    El cuelgue es lo mas grave: no es un valor malo, es el proceso trabado. Y el
    `None` es exactamente lo que `0.3.0` se dedico a erradicar de
    `read_grayscale`.

    `validate_kernel` ya rechazaba kernels vacios con `kernel must not be
    empty`; las imagenes no tenian el chequeo equivalente. Ahora si, en
    `validate_binary_image` y `validate_grayscale_image`.

    **Advertencia para quien mute esto**: si se quita la validacion, los tests de
    `apply_clahe` con `(0, 5)` y `(5, 0)` **no fallan, cuelgan**. Correr las
    mutaciones con timeout.
    """

    FORMAS = [(0, 0), (0, 5), (5, 0)]

    @pytest.mark.parametrize("forma", FORMAS)
    @pytest.mark.parametrize(
        "operacion", [vpx_erode, vpx_dilate, gray_erode, segment_otsu, apply_clahe]
    )
    def test_toda_operacion_rechaza_una_imagen_vacia(self, operacion, forma):
        with pytest.raises(ValueError, match="image must not be empty"):
            operacion(np.zeros(forma, dtype=np.uint8))

    @pytest.mark.parametrize("operacion", [vpx_skeletonize, vpx_thin])
    def test_zhang_suen_tambien_la_rechaza(self, operacion):
        """Estas dos **no fallaban**: usan padding de ceros, que numpy acepta
        sobre un eje vacio, asi que devolvian `(0, 0)` sin quejarse."""
        with pytest.raises(ValueError, match="image must not be empty"):
            operacion(np.zeros((0, 0), dtype=np.uint8))


class TestImagenesMinimas:
    """Una imagen de un pixel es valida, no degenerada."""

    def test_un_solo_pixel_sobrevive_a_la_erosion(self):
        """Con padding por reflejo el unico pixel se ve rodeado de si mismo."""
        resultado = vpx_erode(np.array([[255]], dtype=np.uint8))

        assert resultado.shape == (1, 1)
        assert resultado[0, 0] == 255

    def test_una_sola_fila_se_procesa_como_imagen_2d(self):
        imagen = np.array([[0, 255, 255, 0]], dtype=np.uint8)

        resultado = vpx_erode(imagen)

        assert resultado.shape == (1, 4)
        assert np.count_nonzero(resultado) == 0


class TestKernelMasGrandeQueLaImagen:
    """No es un error: el reflejo extiende la imagen tanto como haga falta."""

    @pytest.mark.parametrize("lado", [3, 7, 9])
    def test_la_erosion_borra_y_la_dilatacion_llena(self, lado):
        imagen = np.array([[0, 255], [255, 0]], dtype=np.uint8)
        kernel = np.ones((lado, lado), dtype=np.uint8)

        assert np.count_nonzero(vpx_erode(imagen, kernel=kernel)) == 0
        assert np.count_nonzero(vpx_dilate(imagen, kernel=kernel)) == 4


class TestKernelNoCuadrado:
    """La API los acepta y el CLI no puede expresarlos, asi que solo un test los
    ejercita. Estaban declarados sin cubrir en `docs/testing.md`."""

    @staticmethod
    def _cruz_gruesa():
        """Una cruz, no un bloque.

        **Sobre un bloque solido `3x5` y `5x3` dan lo mismo** — los dos lo
        borran entero — y el test no distinguiria nada. La cruz deja un
        segmento horizontal con `3x5` y uno vertical con `5x3`.
        """
        imagen = np.zeros((11, 11), dtype=np.uint8)
        imagen[4:7, 1:10] = 255
        imagen[1:10, 4:7] = 255
        return imagen

    def test_3x5_y_5x3_son_transpuestos_uno_del_otro(self):
        imagen = self._cruz_gruesa()

        ancho = vpx_erode(imagen, kernel=np.ones((3, 5), dtype=np.uint8))
        alto = vpx_erode(imagen, kernel=np.ones((5, 3), dtype=np.uint8))

        assert not np.array_equal(ancho, alto)
        # la imagen es simetrica, asi que un resultado es la transpuesta del otro
        np.testing.assert_array_equal(ancho, alto.T)

    def test_3x5_deja_un_segmento_horizontal(self):
        resultado = vpx_erode(self._cruz_gruesa(), kernel=np.ones((3, 5), dtype=np.uint8))
        filas, columnas = np.nonzero(resultado)

        assert sorted(set(filas.tolist())) == [5]
        assert sorted(set(columnas.tolist())) == [3, 4, 5, 6, 7]


class TestIteracionesYMaxIterations:
    """`iterations > 2` y el `max_iterations` explicito no se probaban: solo se
    ejercitaba el camino hasta la convergencia."""

    @pytest.mark.parametrize(
        "iteraciones,activos_esperados", [(1, 81), (2, 49), (3, 25), (5, 1)]
    )
    def test_cada_iteracion_come_un_anillo(self, iteraciones, activos_esperados):
        imagen = np.zeros((15, 15), dtype=np.uint8)
        imagen[2:13, 2:13] = 255

        resultado = vpx_erode(imagen, iterations=iteraciones)

        assert np.count_nonzero(resultado) == activos_esperados

    @pytest.mark.parametrize(
        "max_iterations,activos_esperados", [(1, 9), (2, 25), (3, 49), (None, 121)]
    )
    def test_reconstruct_se_detiene_donde_se_le_dice(self, max_iterations, activos_esperados):
        """Con `None` corre hasta converger y llena la mascara entera."""
        marcador = np.zeros((15, 15), dtype=np.uint8)
        marcador[7, 7] = 255
        mascara = np.zeros((15, 15), dtype=np.uint8)
        mascara[2:13, 2:13] = 255

        resultado = vpx_reconstruct(marcador, mascara, max_iterations=max_iterations)

        assert np.count_nonzero(resultado) == activos_esperados

    @pytest.mark.parametrize("max_iterations,activos_esperados", [(1, 24), (2, 8), (None, 1)])
    def test_skeletonize_se_detiene_donde_se_le_dice(self, max_iterations, activos_esperados):
        imagen = np.zeros((13, 13), dtype=np.uint8)
        imagen[3:10, 3:10] = 255

        resultado = vpx_skeletonize(imagen, max_iterations=max_iterations)

        assert np.count_nonzero(resultado) == activos_esperados
