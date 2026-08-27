# API Reference

Referencia de la API pública de `vispyx` (`0.2.1`). Todo lo listado aquí se
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

Delega íntegramente en OpenCV:

```python
clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
return clahe.apply(image)
```

`title_grid_size` (con el typo "title") es un shim de compatibilidad: si se pasa
distinto de `None`, **sobrescribe** a `tile_grid_size`. No aparece en el
docstring y no debería usarse en código nuevo.

Sin validación propia: la entrada debe ser de un canal, 8 o 16 bits. Cualquier
otra cosa produce `cv2.error` sin traducir. Devuelve un array con el mismo shape
y dtype de la entrada.

## Segmentación

### `segment_otsu(image)`

```python
thresh = threshold_otsu(image)      # skimage.filters
binary = image > thresh             # umbralización en NumPy puro
return binary.astype(np.uint8) * 255
```

Devuelve `uint8` en `{0, 255}`. El umbral viene de scikit-image; la
umbralización no usa `cv2.threshold`. Sin validación propia: una imagen
constante o vacía hace fallar a skimage directamente.

Es la puerta de entrada natural al bloque `vpx_*`, porque produce exactamente la
convención 0/255 que ese bloque espera.

## Utilidades

### `read_grayscale(path)`

`cv2.imread(path, cv2.IMREAD_GRAYSCALE)` sin más. **Si el archivo no existe o no
es decodificable devuelve `None` en silencio**, no lanza excepción. Hay que
comprobarlo en el llamador:

```python
img = read_grayscale(path)
if img is None:
    raise FileNotFoundError(path)
```

El CLI no usa esta función: tiene su propio `_read_grayscale` que sí valida.

### `show_image(image, title="Imagen", cmap="gray")`

`plt.imshow` + `title` + `axis("off")` + `plt.show()`. Devuelve `None`, es puro
efecto secundario. Requiere un backend de matplotlib con display.

## Catálogo de errores

Todos son `ValueError` — nunca `TypeError` ni `AssertionError` — y sus mensajes
son estables (los tests dependen de ellos literalmente).

| Mensaje | Origen |
|---|---|
| `image must be a 2D array` | `validate_binary_image`, `validate_grayscale_image` |
| `image must contain numeric values` | `validate_grayscale_image` (dtype no numérico) |
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
if img is None:
    raise FileNotFoundError("no se pudo leer la imagen")

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
