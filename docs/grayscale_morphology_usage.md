# Morfología grayscale (`gray_*`)

Siete operaciones sobre imágenes de intensidad, sin binarizar en ningún punto.
Igual que el bloque binario, están implementadas desde cero: `numpy` para los
reducers y bucles Python para la ventana deslizante. No hay `cv2.morphologyEx`
ni `scipy.ndimage` detrás.

## Contrato común

- **Entrada**: array 2D con dtype **numérico** (`np.issubdtype(dtype,
  np.number)`). No se binariza, no se normaliza.
- **Salida**: mismo shape y **mismo dtype que la entrada** — `uint8` entra,
  `uint8` sale; `float64` entra, `float64` sale.
- **Kernel**: `None` es `np.ones((3, 3))`. Elemento estructurante **plano**: el
  kernel dice qué vecinos entran al mínimo/máximo, no los pondera.
- **Padding**: reflejo (`mode="reflect"`, equivalente a
  `cv2.BORDER_REFLECT_101`).

## El motor

```python
for _ in range(iterations):
    padded = pad_image(img, kernel)
    output = np.zeros_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            region = padded[i:i+kh, j:j+kw]
            output[i, j] = reducer(region[active_mask])
    img = output
return img.astype(image.dtype, copy=False)
```

Mismo esquema que el binario, con dos diferencias: el `reducer` recibe solo la
región (sin `active_count`), y el resultado no se multiplica por 255.

## Primitivas

### `gray_erode(image, kernel=None, iterations=1)`

`reducer = np.min`. Cada píxel toma el **mínimo** de su vecindad activa. Oscurece
y expande las zonas oscuras; borra picos claros pequeños.

### `gray_dilate(image, kernel=None, iterations=1)`

`reducer = np.max`. Cada píxel toma el **máximo**. Aclara y expande las zonas
claras; borra valles oscuros pequeños.

```python
import numpy as np
from vispyx import gray_dilate, gray_erode

img = np.full((5, 5), 50, np.uint8)
img[2, 2] = 200

gray_erode(img)[2, 2]    # 50: el pico desaparece
gray_dilate(img)[1, 1]   # 200: el pico se propaga a la vecindad
```

## Compuestas

### `gray_open(image, kernel=None, iterations=1)`

`gray_dilate(gray_erode(image, k, n), k, n)`. Elimina **picos claros** más
chicos que el kernel, dejando el resto de la intensidad casi intacta.

### `gray_close(image, kernel=None, iterations=1)`

`gray_erode(gray_dilate(image, k, n), k, n)`. Rellena **valles oscuros**
pequeños. Es el suavizador estándar antes de segmentar.

Como en el bloque binario, `iterations=n` en las compuestas significa *n
erosiones y luego n dilataciones*, no *n aperturas*.

### `gray_gradient(image, kernel=None, iterations=1)`

```python
gradient = (dilated.astype(np.int32) - eroded.astype(np.int32)).clip(min=0)
return gradient.astype(img.dtype, copy=False)
```

Realce de bordes: mide el rango local de intensidad. El cálculo intermedio va en
`int32` para no desbordar `uint8`.

### `gray_tophat(image, kernel=None, iterations=1)`

`image - gray_open(image)`. Extrae detalles **claros** sobre fondo oscuro y, de
paso, corrige iluminación de fondo desigual: si el fondo varía suavemente, la
apertura lo captura y la resta lo elimina.

### `gray_blackhat(image, kernel=None, iterations=1)`

`gray_close(image) - image`. Simétrico: detalles **oscuros** sobre fondo claro.

## Diferencias frente al bloque binario

| | `vpx_*` | `gray_*` |
|---|---|---|
| Binariza la entrada | sí (`> 0`) | no |
| dtype de salida | siempre `uint8` `{0,255}` | el de la entrada |
| Operaciones disponibles | 12 | 7 |
| `boundary`, `hitmiss`, `reconstruct`, `skeletonize`, `thin` | sí | **no existen** |
| Cast intermedio en las restas | `int16` | `int32` |

## Sobre el dtype de salida

El resultado conserva el dtype de la entrada, y la entrada puede ser cualquier
cosa que `np.asarray` acepte:

```python
gray_erode([[1, 2, 3], [4, 5, 6], [7, 8, 9]])   # funciona, dtype inferido
gray_erode(np.ones((3, 3), np.float32)).dtype   # float32
```

Hasta `0.2.0` las cuatro operaciones primitivas lanzaban `AttributeError` con
listas anidadas, porque el cast final iba contra el argumento crudo. Desde
`0.2.1` va contra el array ya validado, como siempre hicieron `gray_gradient`,
`gray_tophat` y `gray_blackhat`.

## Receta: preparar una imagen antes de segmentar

```python
from vispyx import apply_clahe, gray_close, gray_tophat, kernel_disk, segment_otsu

realzada  = apply_clahe(img, clip_limit=3.0, tile_grid_size=(8, 8))
sin_fondo = gray_tophat(realzada, kernel=kernel_disk(5))   # aplana iluminación
suave     = gray_close(sin_fondo, kernel=kernel_disk(1))   # cierra micro-valles
mascara   = segment_otsu(suave)                            # ahora sí, binarizar
```

La idea: todo el trabajo sobre intensidad ocurre en `gray_*`, y `segment_otsu`
es la frontera a partir de la cual manda el bloque `vpx_*`.

## Rendimiento

Los bucles son Python puro. Una imagen de 1024×1024 con kernel 5×5 hace
26 millones de evaluaciones de `np.min` sobre arrays chiquitos, y eso son
minutos, no segundos. Para trabajo interactivo, recorta la región de interés o
submuestrea antes. La lentitud es el precio explícito de tener el algoritmo
legible; ver [architecture.md](./architecture.md).

## Ver también

- [binary_morphology_usage.md](./binary_morphology_usage.md)
- [api_reference.md](./api_reference.md)
