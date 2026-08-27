# Morfología binaria (`vpx_*`)

Doce operaciones binarias implementadas desde cero, sin OpenCV ni SciPy en el
camino algorítmico. Este documento describe **lo que el código hace**, no la
teoría general.

## Contrato común

- **Entrada**: cualquier array 2D numérico. Se binariza internamente con
  `(image > 0)`: 0/255, 0/1, `bool`, floats y negativos son todos aceptables.
- **Salida**: siempre `uint8` con valores exactamente `{0, 255}`.
- **Kernel**: `None` significa `np.ones((3, 3))`. Se normaliza con `> 0`, así
  que solo importa qué celdas son activas, no sus valores.
- **Padding**: reflejo (`mode="reflect"`), grosor `kh // 2` × `kw // 2`.
- **Iteraciones**: se aplican por composición literal — la salida de una pasada
  es la entrada de la siguiente.

## El motor

Todas las operaciones de vecindad pasan por `apply_binary_operation`, en
`morphology_common.py`:

```python
for _ in range(iterations):
    padded = pad_image(img, kernel)
    output = np.zeros_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            region = padded[i:i+kh, j:j+kw]
            output[i, j] = reducer(region[active_mask], active_count)
    img = output
return img * 255
```

Es un doble bucle Python **sin vectorizar**: `region[active_mask]` entrega solo
los valores bajo las celdas activas del kernel. Lo que distingue una operación
de otra es únicamente el `reducer`. Esto es deliberado — el objetivo del paquete
es que el algoritmo sea legible — pero significa que el costo es
`O(iterations · H · W · |kernel|)` en Python puro. Para imágenes grandes se
nota.

## Operaciones primitivas

### `vpx_erode(image, kernel=None, iterations=1)`

`reducer = lambda region, active_count: int(np.sum(region) == active_count)`

Un píxel sobrevive solo si **todas** las celdas activas del kernel caen sobre
primer plano. Adelgaza objetos y borra ruido menor que el kernel.

### `vpx_dilate(image, kernel=None, iterations=1)`

`reducer = lambda region, _: int(np.any(region))`

Un píxel se enciende si **alguna** celda activa cae sobre primer plano. Engorda
objetos y cierra grietas.

```python
import numpy as np
from vispyx import kernel_cross, vpx_dilate, vpx_erode

img = np.zeros((7, 7), np.uint8)
img[3, 3] = 255

vpx_dilate(img, kernel=kernel_cross(3))   # rombo de 5 píxeles
vpx_erode(img)                            # todo a 0: un píxel no sobrevive a 3x3
```

## Compuestas

### `vpx_open(image, kernel=None, iterations=1)`

`vpx_dilate(vpx_erode(image, k, n), k, n)`

Elimina objetos y salientes más chicos que el kernel, preservando el tamaño de
lo que sobrevive. Es la operación de limpieza de máscaras por defecto.

### `vpx_close(image, kernel=None, iterations=1)`

`vpx_erode(vpx_dilate(image, k, n), k, n)`

Rellena huecos y uniones finas más chicos que el kernel.

**Ojo con `iterations` en las compuestas**: `vpx_open(img, k, 3)` hace **tres
erosiones seguidas y después tres dilataciones**, no tres aperturas encadenadas.
No es lo mismo, y el segundo caso hay que escribirlo a mano si es lo que se
quiere.

### `vpx_gradient(image, kernel=None, iterations=1)`

```python
dilated = vpx_dilate(image, kernel, iterations)
eroded  = vpx_erode(image, kernel, iterations)   # sobre la ORIGINAL, no sobre dilated
gradient = (dilated.astype(np.int16) - eroded.astype(np.int16)).clip(min=0)
```

Contorno de grosor proporcional al kernel. El cast a `int16` evita el wraparound
de `uint8` en la resta; el `clip(min=0)` es defensivo (dilatación ⊇ erosión
siempre se cumple aquí).

### `vpx_tophat` / `vpx_blackhat`

- `tophat = image - vpx_open(image)`: lo que la apertura se llevó, es decir los
  detalles claros pequeños.
- `blackhat = vpx_close(image) - image`: lo que el cierre rellenó, los huecos
  oscuros pequeños.

Ambas normalizan primero la entrada a 0/255 y hacen la resta en `int16`.

### `vpx_boundary(image, kernel=None, iterations=1)`

`image - vpx_erode(image)`. Frontera **interna**: los píxeles de primer plano
que la erosión se come. A diferencia de `vpx_gradient`, no engorda hacia afuera.

## Especializadas

### `vpx_hitmiss(image, kernel_hit, kernel_miss)`

Única función binaria **sin `iterations`**: siempre una pasada.

```python
hit  = vpx_erode(img,       kernel=kernel_hit,  iterations=1)
miss = vpx_erode(255 - img, kernel=kernel_miss, iterations=1)
return np.logical_and(hit > 0, miss > 0).astype(np.uint8) * 255
```

Detecta un patrón exacto: `kernel_hit` marca dónde debe haber primer plano,
`kernel_miss` dónde debe haber fondo. Los dos kernels deben tener la **misma
forma** y **no solaparse**:

```python
import numpy as np
from vispyx import vpx_hitmiss

hit  = np.array([[0,1,0],[1,1,1],[0,1,0]], np.uint8)   # cruz
miss = np.array([[1,0,1],[0,0,0],[1,0,1]], np.uint8)   # esquinas
esquinas = vpx_hitmiss(mascara, hit, miss)
```

Errores propios: `kernel_hit and kernel_miss must have the same shape`,
`kernel_hit and kernel_miss must not overlap`.

### `vpx_reconstruct(marker, mask, kernel=None, max_iterations=None)`

Reconstrucción geodésica: dilata el marcador repetidamente, recortando cada vez
contra la máscara, hasta que deja de cambiar.

```python
current = marker * 255
while True:
    dilated = vpx_dilate(current, kernel=kernel, iterations=1)
    updated = np.minimum(dilated, mask_uint8)
    if np.array_equal(updated, current):
        return updated                    # convergió
    current = updated
    steps += 1
    if max_iterations is not None and steps >= max_iterations:
        return current                    # reconstrucción PARCIAL
```

Uso típico: recuperar los componentes conexos completos que tocan una semilla,
descartando el resto.

Precondiciones estrictas: `marker.shape == mask.shape` y `marker ⊆ mask` punto a
punto. Si el marcador se sale de la máscara —
`ValueError("marker must be a subset of mask")`. Con marcador todo en ceros
converge en la primera vuelta y devuelve ceros.

`max_iterations=None` significa "hasta converger". Un valor finito puede
devolver una reconstrucción incompleta, y no hay forma de saber desde el
resultado si convergió o si se truncó.

### `vpx_skeletonize(image, max_iterations=None)`

Adelgazamiento de **Zhang-Suen** escrito a mano, no delegado en skimage. Cada
iteración tiene dos subpasadas; en cada una se marcan los píxeles a borrar y se
borran **todos juntos al final** de la subpasada (eliminación diferida, como
manda el algoritmo).

Un píxel de primer plano se elimina si cumple las cuatro condiciones:

1. `2 <= B(P1) <= 6` — entre 2 y 6 vecinos activos
2. `A(P1) == 1` — exactamente una transición 0→1 recorriendo los 8 vecinos en
   círculo (`_count_transitions`)
3. subpasada 0: `p2·p4·p6 == 0` **y** `p4·p6·p8 == 0`
4. subpasada 1: `p2·p4·p8 == 0` **y** `p2·p6·p8 == 0`

con `p2..p9` = N, NE, E, SE, S, SW, W, NW.

**Esta función usa padding de ceros**, no de reflejo: el exterior de la imagen
se trata como fondo. Es lo estándar para Zhang-Suen, pero es inconsistente con
el resto del módulo — conviene tenerlo presente si comparas resultados en el
borde.

`max_iterations=None` corre hasta punto fijo. Un bloque sólido 3×3 se reduce a
un único píxel central.

### `vpx_thin(image, iterations=1)`

Literalmente `vpx_skeletonize(image, max_iterations=iterations)`.

Con el default `iterations=1` hace **una sola iteración** (dos subpasadas) de
Zhang-Suen: adelgazamiento parcial. Si lo que quieres es el esqueleto,
`vpx_skeletonize` es la función; `vpx_thin` sirve para ver el proceso paso a
paso o para adelgazar sin llegar al hueso.

## Receta: limpiar una máscara de segmentación

```python
from vispyx import kernel_disk, segment_otsu, vpx_close, vpx_open, vpx_boundary

mascara = segment_otsu(imagen_gris)                       # uint8 {0,255}
sin_ruido = vpx_open(mascara, kernel=kernel_disk(1))      # borra motas
sin_huecos = vpx_close(sin_ruido, kernel=kernel_disk(2))  # rellena poros
contorno = vpx_boundary(sin_huecos)                       # frontera interna
```

El orden importa: `open` antes que `close` borra ruido antes de rellenar; al
revés, el cierre consolida el ruido y la apertura ya no lo distingue del objeto.

## Ver también

- [grayscale_morphology_usage.md](./grayscale_morphology_usage.md)
- [api_reference.md](./api_reference.md)
