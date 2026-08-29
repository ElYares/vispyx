# API Reference

Referencia de la API pública de `vispyx` (`0.4.0`). Todo lo listado aquí se
importa directamente desde el paquete raíz:

```python
from vispyx import vpx_open, gray_close, kernel_disk, segment_otsu
```

## Superficie pública

`vispyx.__all__` expone **28 símbolos**: `__version__`, 4 generadores de
kernels, 7 operaciones grayscale, 12 operaciones binarias, y 4 utilidades
(`apply_clahe`, `segment_otsu`, `read_grayscale`, `show_image`).

`vispyx.morphology` es una **fachada de compatibilidad**: preserva el import
histórico `from vispyx.morphology import ...` y reexporta las mismas funciones,
sin envolverlas ni alterar su comportamiento. Su `__all__` tiene 20 nombres e
incluye uno que **no** llega al paquete raíz: `vpx_pad_image` (alias de
`vispyx.morphology_common.pad_image`). Es accesible como
`vispyx.morphology.vpx_pad_image`, nunca como `vispyx.vpx_pad_image`.

## Convenciones que atraviesan toda la API

| Aspecto | Binario `vpx_*` | Grayscale `gray_*` |
|---|---|---|
| Entrada | cualquier array 2D numérico; se binariza con `> 0` | array 2D numérico, **sin binarizar** |
| Salida | `uint8` con valores exactamente `{0, 255}` | mismo dtype que la entrada |
| Kernel por defecto | `np.ones((3, 3))` (vecindad de 8 + centro) | igual |
| Padding | reflejo (`np.pad(mode="reflect")`) | igual |
| Errores | siempre `ValueError`, con mensajes estables | igual |

Dos excepciones al padding y a la forma: `vpx_skeletonize`/`vpx_thin` usan
padding **de ceros** (`mode="constant"`), porque Zhang-Suen asume fondo fuera de
la imagen. Es una inconsistencia deliberada respecto al resto del módulo.

El `mode="reflect"` de NumPy refleja **sin duplicar el borde**: equivale a
`cv2.BORDER_REFLECT_101`, no a `cv2.BORDER_REFLECT`.

## Generadores de kernels

Todos devuelven `uint8` con valores `{0, 1}` (elementos estructurantes planos,
no ponderados).

### `kernel_square(size)`

Cuadrado lleno de unos, `(size, size)`.

### `kernel_cross(size)`

Fila y columna centrales activas.

```text
kernel_cross(3)     kernel_cross(5)
[[0 1 0]            [[0 0 1 0 0]
 [1 1 1]             [0 0 1 0 0]
 [0 1 0]]            [1 1 1 1 1]
                     [0 0 1 0 0]
                     [0 0 1 0 0]]
```

### `kernel_diamond(size)`

Activo donde la distancia **Manhattan** al centro es `<= size // 2`.

```text
kernel_diamond(5)
[[0 0 1 0 0]
 [0 1 1 1 0]
 [1 1 1 1 1]
 [0 1 1 1 0]
 [0 0 1 0 0]]
```

### `kernel_disk(radius)`

Lado `2 * radius + 1`. Activo donde la distancia **euclídea al cuadrado** al
centro es `<= radius ** 2` (comparación sin `sqrt`).

`kernel_disk(0)` es `[[1]]`. Para radios pequeños coincide con otras formas:
`kernel_disk(1) == kernel_cross(3) == kernel_diamond(3)` y
`kernel_disk(2) == kernel_diamond(5)`. Es coincidencia geométrica de la malla
discreta, no una propiedad general — a radios mayores divergen.

### Validación

| Condición | Error |
|---|---|
| `size` no `int` o `<= 0` | `ValueError("size must be a positive integer")` |
| `size` par | `ValueError("size must be odd")` |
| `radius` no `int` o `< 0` | `ValueError("radius must be a non-negative integer")` |

`size=2.0` falla: `float` no es `int`. `radius=0` sí es válido.

## Morfología binaria

Ver [binary_morphology_usage.md](./binary_morphology_usage.md) para el detalle
algorítmico. Firmas:

```python
vpx_erode(image, kernel=None, iterations=1)
vpx_dilate(image, kernel=None, iterations=1)
vpx_open(image, kernel=None, iterations=1)
vpx_close(image, kernel=None, iterations=1)
vpx_gradient(image, kernel=None, iterations=1)
vpx_tophat(image, kernel=None, iterations=1)
vpx_blackhat(image, kernel=None, iterations=1)
vpx_boundary(image, kernel=None, iterations=1)
vpx_hitmiss(image, kernel_hit, kernel_miss)                    # sin iterations
vpx_reconstruct(marker, mask, kernel=None, max_iterations=None)
vpx_skeletonize(image, max_iterations=None)
vpx_thin(image, iterations=1)
```

Todas devuelven `uint8` en `{0, 255}`.

## Morfología grayscale

Ver [grayscale_morphology_usage.md](./grayscale_morphology_usage.md). Firmas:

```python
gray_erode(image, kernel=None, iterations=1)
gray_dilate(image, kernel=None, iterations=1)
gray_open(image, kernel=None, iterations=1)
gray_close(image, kernel=None, iterations=1)
gray_gradient(image, kernel=None, iterations=1)
gray_tophat(image, kernel=None, iterations=1)
gray_blackhat(image, kernel=None, iterations=1)
```

Todas conservan el dtype de la entrada. **No existen** contrapartes grayscale de
`boundary`, `hitmiss`, `reconstruct`, `skeletonize` ni `thin`: la API grayscale
es deliberadamente más chica.

## Preprocesamiento

### `apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8), title_grid_size=None)`

Valida y delega en OpenCV:

```python
img = validate_grayscale_image(image)
if img.dtype not in (np.uint8, np.uint16):
    raise ValueError("image must be uint8 or uint16")
clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
return clahe.apply(img)
```

`title_grid_size` (con el typo "title") es un shim de compatibilidad: si se pasa
distinto de `None`, **sobrescribe** a `tile_grid_size`. No aparece en el
docstring y no debería usarse en código nuevo.

Reusa `validate_grayscale_image`, así que comparte los dos mensajes de error
del resto del paquete, y agrega uno propio: OpenCV solo implementa CLAHE para
`CV_8UC1` y `CV_16UC1`. Verificado contra la versión instalada — `int16`,
`int32`, `float32` y `float64` fallaban dentro de `clahe.cpp`.

**Cambio de contrato**: antes esos casos salían como `cv2.error` sin traducir.
Ahora salen como `ValueError`, igual que toda validación del paquete. Devuelve
un array con el mismo shape y dtype de la entrada.

## Segmentación

### `segment_otsu(image)`

```python
img = validate_grayscale_image(image)
thresh = threshold_otsu(img)        # skimage.filters
binary = img > thresh               # umbralización en NumPy puro
return binary.astype(np.uint8) * 255
```

Devuelve `uint8` en `{0, 255}`. El umbral viene de scikit-image; la
umbralización no usa `cv2.threshold`.

Reusa `validate_grayscale_image`, así que comparte los dos mensajes de error del
resto del paquete. **Cambio de contrato**: antes una imagen de tres canales
**no fallaba** — skimage emitía un `UserWarning` que nadie ve y devolvía un
resultado sin sentido; una lista anidada salía como `AttributeError` y un dtype
no numérico como `UFuncTypeError`. Ahora los tres son `ValueError`, y las listas
anidadas **funcionan**, igual que en las `gray_*` desde `0.2.1`.

**Una imagen constante no falla**: el umbral de Otsu es esa misma constante, así
que `image > thresh` no deja nada y sale una máscara toda negra, sin warning ni
error. Está fijado por un test.

Es la puerta de entrada natural al bloque `vpx_*`, porque produce exactamente la
convención 0/255 que ese bloque espera.

## Utilidades

### `read_grayscale(path)`

Lee una imagen y la devuelve en escala de grises: `np.ndarray` 2D `uint8`.

```python
img = read_grayscale(path)     # y ya: si vuelve, es una imagen
```

`cv2.imread` no lanza cuando falla — devuelve `None` tanto si el archivo no
existe como si existe y no se puede decodificar. `read_grayscale` traduce esos
dos casos a excepciones distintas:

| Situación | Error |
|---|---|
| no hay nada en `path` | `FileNotFoundError: No se encontró la imagen en {path}` |
| el archivo existe pero no es una imagen legible | `ValueError: No se pudo decodificar la imagen en {path}` |

Distinguirlos importa: un `FileNotFoundError` sobre un archivo que sí está te
manda a revisar una ruta que estaba bien.

**El CLI usa esta misma función**, así que el error es idéntico se entre por
Python o por la línea de comandos.

### `show_image(image, title="Imagen", cmap="gray", figsize=None)`

`plt.imshow` + `title` + `axis("off")` + `plt.show()`. Devuelve `None`, es puro
efecto secundario. Requiere un backend de matplotlib con display.

`figsize` crea una figura propia de ese tamaño y ajusta los márgenes con
`tight_layout`. Omitirlo dibuja sobre la figura activa, que es el comportamiento
histórico y el que espera un notebook. **Es la única implementación del
paquete**: `cli.py` tenía una copia con el tamaño fijo en `(8, 6)`, y ahora pasa
ese valor por el parámetro.

## Catálogo de errores

Todos son `ValueError` — nunca `TypeError` ni `AssertionError` — y sus mensajes
son estables (los tests dependen de ellos literalmente).

| Mensaje | Origen |
|---|---|
| `image must be a 2D array` | `validate_binary_image`, `validate_grayscale_image` |
| `image must not be empty` | `validate_binary_image`, `validate_grayscale_image` |
| `image must contain numeric values` | `validate_grayscale_image` (dtype no numérico) |
| `image must be uint8 or uint16` | `apply_clahe` (OpenCV solo implementa esos dos) |
| `iterations must be a positive integer` | `validate_iterations` |
| `kernel must be a 2D array` | `validate_kernel` |
| `kernel must not be empty` | `validate_kernel` |
| `kernel dimensions must be odd` | `validate_kernel` |
| `kernel must contain at least one active element` | `validate_kernel` |
| `kernel_hit and kernel_miss must have the same shape` | `vpx_hitmiss` |
| `kernel_hit and kernel_miss must not overlap` | `vpx_hitmiss` |
| `marker and mask must have the same shape` | `vpx_reconstruct` |
| `marker must be a subset of mask` | `vpx_reconstruct` |
| `size must be a positive integer` / `size must be odd` | `kernels._validate_size` |
| `radius must be a non-negative integer` | `kernels._validate_radius` |

## Rarezas del contrato (verificadas por ejecución)

Cosas que no se deducen leyendo las firmas:

1. **`iterations=True` es rechazado** con
   `ValueError("iterations must be a positive integer")`, aunque `bool` sea
   subclase de `int`. Es deliberado: `iterations=True` es un error que conviene
   reportar, no una petición de una iteración.
2. **Cualquier tipo entero sirve para `iterations`**, NumPy incluido:
   `np.int64(2)`, `np.int32(2)` y `np.uint8(2)` son equivalentes a `2`. La
   validación acepta `numbers.Integral`. Un `float` como `2.0` sigue siendo un
   error.
3. **Los kernels no necesitan ser cuadrados**, solo tener ambas dimensiones
   impares: `(3, 5)` y `(1, 7)` son válidos.
4. **Un kernel más grande que la imagen no falla.** `np.pad(mode="reflect")`
   repite el patrón reflejado cuantas veces haga falta; el resultado deja de ser
   un reflejo con sentido físico, pero no hay excepción.
5. **`vpx_hitmiss` es la única binaria sin `iterations`**: siempre una pasada.
6. **`vpx_thin(image)` con el default hace una sola pasada** de Zhang-Suen, no
   el esqueleto completo. Para el esqueleto usa `vpx_skeletonize`.

## Ejemplo completo

```python
import numpy as np
from vispyx import (
    apply_clahe, gray_close, kernel_disk, read_grayscale,
    segment_otsu, vpx_open, vpx_skeletonize,
)

img = read_grayscale("archive/all-mias/mdb001.pgm")

realzada = apply_clahe(img, clip_limit=3.0, tile_grid_size=(8, 8))
suave    = gray_close(realzada, kernel=kernel_disk(1))
mascara  = segment_otsu(suave)                       # uint8 {0, 255}
limpia   = vpx_open(mascara, kernel=kernel_disk(1), iterations=1)
esqueleto = vpx_skeletonize(limpia)

assert limpia.dtype == np.uint8
assert set(np.unique(limpia)) <= {0, 255}
```

## Ver también

- [system_usage.md](./system_usage.md)
- [cli_reference.md](./cli_reference.md)
- [architecture.md](./architecture.md)
