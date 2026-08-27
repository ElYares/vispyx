# Testing

Estado real de la suite, qué contrato fija y dónde están los huecos.

## Correr los tests

```bash
pip install -e .[dev]
pytest -q
```

Estado verificado: **49 tests, 49 pasan, ~1.9 s** (Python 3.14.5, numpy 2.5.2,
OpenCV 5.0, scikit-image 0.26, matplotlib 3.11, pytest 9.1).

No hay `conftest.py` ni configuración de pytest más allá de declarar `pytest`
en `[project.optional-dependencies].dev`.

Si `pytest` falla con `ModuleNotFoundError`, casi siempre es el entorno y no el
código: `test_cli.py` y `test_preprocessing.py` importan `cv2` a nivel de
módulo, y `vispyx/__init__.py` arrastra `skimage` vía `segmentation.py`. Sin
esas dependencias **ningún** archivo de test llega siquiera a colectarse.

## Reparto

| Archivo | Tests | Qué fija |
|---|---|---|
| `test_morphology.py` | 30 | el núcleo algorítmico, binario y grayscale |
| `test_public_api.py` | 8 | cableado de la superficie pública y versión |
| `test_kernels.py` | 6 | forma exacta de los cuatro generadores |
| `test_cli.py` | 3 | tres funciones `run_*` con I/O real en disco |
| `test_preprocessing.py` | 2 | shape y tipo de `apply_clahe`, nada más |

## Qué se verifica de verdad

La estrategia es **valor exacto sobre matrices chicas hechas a mano**, comparadas
con `assert_array_equal` contra un resultado escrito a mano. No hay
property-based testing ni comparación contra implementaciones de referencia.

Lo que sí queda amarrado:

- **Composición operacional**: que `open = erode → dilate`, `close = dilate →
  erode`, `gradient = dilate − erode`, `tophat = image − opening`,
  `blackhat = closing − image`, `boundary = image − erosion` producen los
  números correctos en casos concretos, en ambos dominios.
- **Efecto del kernel**: dilatar con cruz da un rombo y con cuadrado da un
  cuadrado; confirma que `active_mask` restringe la vecindad de verdad.
- **Iteraciones**: `iterations=2` equivale a aplicar la operación dos veces.
- **Reconstrucción**: un marcador de un píxel recupera la región conexa completa
  bajo la máscara, y un marcador fuera de la máscara es rechazado.
- **Convergencia vs. parcialidad**: `vpx_skeletonize` reduce un bloque 3×3 a un
  píxel; `vpx_thin` se verifica con desigualdades (`0 < n <= n_original`),
  no con igualdad.
- **Los mensajes de error**, con `pytest.raises(ValueError, match="...")` sobre
  el texto literal. Esto convierte los mensajes en contrato público.

## Convenciones que la suite impone

- Las `vpx_*` viven en **0/255**, nunca 0/1: el helper `_to_uint8` de
  `test_morphology.py` multiplica por 255 antes de cada llamada.
- Las `gray_*` operan sobre los valores nativos (10, 20, …, 90) y no normalizan.
- Toda entrada inválida produce `ValueError`, nunca `TypeError` ni `assert`.
- `vispyx.morphology` debe seguir funcionando como import path: los tests del
  núcleo importan desde la fachada, no desde `morphology_binary`.
- `vispyx.__version__ == "0.2.0"` está clavado en un test: **subir la versión
  rompe la suite si no se actualiza también ahí**.

## Huecos de cobertura

Reales, verificados leyendo la suite completa.

### Módulos sin ningún test

- **`segmentation.py`**: `segment_otsu` solo aparece nombrada dentro del set
  `expected_symbols`. Nunca se invoca. Ni umbral, ni binarizado, ni dtype.
- **`utils.py`**: `read_grayscale` y `show_image`, igual — mencionadas, nunca
  ejecutadas. En particular no se prueba que `read_grayscale` devuelva `None`
  ante un archivo inexistente, que es su comportamiento más peligroso.

### `cli.py`: 3 de 17 métodos

Sin cubrir: `run_clahe`, `run_otsu`, los cinco `run_vpx_*` básicos y
`_run_grayscale_method` (que sirve a los siete `gray_*`). Tampoco se prueba
`main()`: ni el parseo de `argparse`, ni una sola flag, ni el guardado con
`cv2.imwrite`, ni la creación de directorios, ni el mensaje
`"Imagen procesada. No se guardó."`, ni `parser.error("--mask es obligatorio
para vpx_reconstruct")`, ni `FileNotFoundError` cuando `cv2.imread` devuelve
`None`.

### `preprocessing.py`: test puramente estructural

Solo se verifica shape y que el retorno sea `np.ndarray`. No se prueba el efecto
de `clip_limit`, ni el dtype, ni el alias histórico `title_grid_size`. Además el
test genera ruido aleatorio **sin semilla fija**: es no determinista, aunque hoy
solo mire la forma.

### Ramas de validación nunca ejercidas

`validate_kernel` tiene cuatro caminos de error y los tests solo pisan uno
(dimensión par). Quedan sin probar:

- `kernel must be a 2D array`
- `kernel must not be empty`
- `kernel must contain at least one active element`
- `marker and mask must have the same shape`
- `kernel_hit and kernel_miss must have the same shape`
- `size must be a positive integer` (la rama no-par de `_validate_size`)

### Casos límite sin probar

Imágenes vacías o de un píxel; kernels más grandes que la imagen; kernels no
cuadrados (`3×5`); `max_iterations` explícito en `vpx_reconstruct` y
`vpx_skeletonize` (solo se prueba el camino hasta convergencia);
`iterations > 2`.

### Propiedades matemáticas ausentes

Nada verifica las garantías clásicas de morfología como **propiedad general**,
solo como casos puntuales:

- idempotencia: `open(open(x)) == open(x)`, `close(close(x)) == close(x)`
- dualidad por complemento: `erode(x, k) == ¬dilate(¬x, k)`
- extensividad: `open(x) ⊆ x ⊆ close(x)` para cualquier `x`

Son exactamente el tipo de invariante que se verifica bien con imágenes
aleatorias y semilla fija, y hoy no hay ninguno.

### El oráculo que no se usa

`morph_scipy.py` implementa `erode`, `dilate`, `open`, `close` y `gradient`
binarios sobre `scipy.ndimage`, con la misma semántica de kernel e iteraciones
que `vispyx`. **Ningún test lo importa.** Es la validación cruzada obvia para un
paquete cuyo argumento de venta es "implementado desde cero": comparar la
implementación propia contra una de referencia establecida. Hoy esa comparación
no existe en ninguna forma — tampoco contra `cv2.morphologyEx` ni contra
`skimage.morphology`.

Con una diferencia a tener en cuenta si se hace: el `gradient` de
`morph_scipy.py` re-binariza los resultados de `dilate`/`erode` antes de restar,
mientras que `vpx_gradient` resta en `int16` sobre 0/255. Para entradas binarias
el resultado coincide, pero los pasos intermedios no son idénticos.

## Prioridad sugerida

Si hay que elegir dónde poner el siguiente test, en este orden:

1. `segment_otsu` y `read_grayscale` — cero cobertura en funciones que están en
   la primera línea de cualquier pipeline
2. `main()` del CLI con `monkeypatch` sobre `sys.argv` — 222 líneas casi sin
   tocar, y es la superficie que usa la gente
3. Diferencial contra `morph_scipy.py` — el oráculo ya está escrito
4. Invariantes con imágenes aleatorias y semilla fija — idempotencia y dualidad
5. Las ramas de `validate_kernel` que faltan — baratas, una línea cada una

## Ver también

- [architecture.md](./architecture.md)
- [../CONTRIBUTING.md](../CONTRIBUTING.md)
