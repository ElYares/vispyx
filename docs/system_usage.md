# Guía integral de uso

Cómo instalar `vispyx`, cómo pensar un pipeline con él y qué trampas evitar.
Para el detalle función por función, ver [api_reference.md](./api_reference.md).

## Qué es y qué no es

`vispyx` es un paquete de procesamiento de imágenes con un **núcleo de
morfología matemática implementado desde cero**. Erosión, dilatación,
reconstrucción geodésica y Zhang-Suen están escritos con bucles de NumPy, no
delegados en OpenCV ni en SciPy. OpenCV y scikit-image se usan solo en los
bordes: leer archivos, CLAHE y el umbral de Otsu.

Eso define para qué sirve. Es un paquete para **entender y controlar** lo que
hace cada operación, y para trabajar sobre regiones de interés de tamaño
moderado. No es un paquete para procesar lotes grandes: los bucles son Python
puro y se nota (ver [architecture.md](./architecture.md)).

## Instalación

```bash
pip install -e .          # paquete + dependencias
pip install -e .[dev]     # además pytest
```

Requiere Python `>= 3.7`. Dependencias declaradas: `opencv-python`, `numpy`,
`scikit-image`, `matplotlib`.

Verificar que quedó bien:

```bash
python -c "import vispyx; print(vispyx.__version__)"   # 0.2.0
vispyx --help
pytest -q                                              # 49 passed
```

Combinación verificada: Python 3.14.5, numpy 2.5.2, OpenCV 5.0,
scikit-image 0.26, matplotlib 3.11. El `requires-python = ">=3.7"` del
`pyproject.toml` es optimista — no hay CI que pruebe versiones viejas.

`scipy` **no** es dependencia declarada, aunque `morph_scipy.py` (script suelto
en la raíz, fuera del paquete) lo importe. Llega de refilón como dependencia de
scikit-image.

## El modelo mental

Todo pipeline de `vispyx` cruza una frontera:

```text
  imagen de grises                        máscara binaria
  ────────────────                        ───────────────
  apply_clahe          ──segment_otsu──►  vpx_erode / vpx_dilate
  gray_erode/dilate                       vpx_open / vpx_close
  gray_open/close                         vpx_gradient / vpx_boundary
  gray_gradient                           vpx_tophat / vpx_blackhat
  gray_tophat/blackhat                    vpx_hitmiss / vpx_reconstruct
                                          vpx_skeletonize / vpx_thin
```

A la izquierda se trabaja con **intensidad** y el dtype se conserva. A la derecha
se trabaja con **forma** y todo es `uint8` en `{0, 255}`. `segment_otsu` es el
puente, y produce exactamente la convención que el lado derecho espera.

**La trampa número uno**: pasar una imagen de grises directamente a una `vpx_*`.
No falla — se binariza con `> 0`, así que todo píxel que no sea negro puro se
vuelve 255, y sale una máscara casi sólida. Silencio total, resultado inútil.
Segmenta primero.

## Pipeline completo en Python

```python
import numpy as np
from vispyx import (
    apply_clahe, gray_close, kernel_disk, read_grayscale,
    segment_otsu, vpx_boundary, vpx_open, vpx_skeletonize,
)

# 1. leer  ─ read_grayscale devuelve None si falla, hay que comprobarlo
img = read_grayscale("archive/all-mias/mdb001.pgm")
if img is None:
    raise FileNotFoundError("no se pudo leer la imagen")

# 2. realzar contraste local
realzada = apply_clahe(img, clip_limit=3.0, tile_grid_size=(8, 8))

# 3. suavizar en grises: cierra micro-valles sin mover los bordes grandes
suave = gray_close(realzada, kernel=kernel_disk(1))

# 4. cruzar la frontera
mascara = segment_otsu(suave)                # uint8 {0, 255}

# 5. limpiar la forma
limpia = vpx_open(mascara, kernel=kernel_disk(1), iterations=1)

# 6. extraer lo que interesa
contorno  = vpx_boundary(limpia)
esqueleto = vpx_skeletonize(limpia)

assert limpia.dtype == np.uint8 and set(np.unique(limpia)) <= {0, 255}
```

El mismo pipeline por CLI está en
[cli_reference.md](./cli_reference.md#pipeline-típico-en-una-línea-de-comandos).

## Elegir el kernel

El kernel decide **qué se considera vecino**, y con eso, qué tan agresiva es cada
operación. Cuatro generadores, todos con valores `{0, 1}`:

| Generador | Forma | Cuándo |
|---|---|---|
| `kernel_square(n)` | cuadrado lleno | agresivo, isotrópico en 8 direcciones; el default implícito |
| `kernel_cross(n)` | cruz | conservador, favorece estructuras horizontales y verticales |
| `kernel_diamond(n)` | rombo (Manhattan) | intermedio |
| `kernel_disk(r)` | disco (euclídeo) | el más "natural" para formas orgánicas |

`size` debe ser **impar** y `radius` no negativo. Para radios chicos varias
formas coinciden: `kernel_disk(1) == kernel_cross(3) == kernel_diamond(3)`, y
`kernel_disk(2) == kernel_diamond(5)`. A radios mayores divergen.

Regla práctica: el kernel debe ser **más chico que lo que quieres conservar y
más grande que lo que quieres borrar**. Si una apertura se come el objeto, el
kernel es muy grande.

## Elegir `iterations`

`iterations=n` aplica la operación n veces por composición. En las compuestas
esto significa algo específico: `vpx_open(img, k, 3)` hace **tres erosiones
seguidas y luego tres dilataciones**, no tres aperturas encadenadas. Si lo que
quieres es lo segundo, escríbelo:

```python
resultado = mascara
for _ in range(3):
    resultado = vpx_open(resultado, kernel=k)
```

Casi siempre es mejor un kernel más grande con `iterations=1` que un kernel
chico repetido: es más barato y más predecible.

## Cosas que sorprenden

Recogidas del código, todas verificadas:

- **`vpx_thin(img)` no da el esqueleto.** El default `iterations=1` hace una sola
  pasada de Zhang-Suen. El esqueleto es `vpx_skeletonize(img)`.
- **`iterations=np.int64(2)` falla** con `ValueError`. La validación exige `int`
  nativo. Usa `int(n)` si el valor viene de NumPy.
- **`gray_erode([[1,2],[3,4]])` lanza `AttributeError`**, no `ValueError`. Pasa
  siempre `np.ndarray` a las `gray_*`.
- **`read_grayscale` devuelve `None` sin avisar** si el archivo no existe o no se
  puede decodificar.
- **`apply_clahe` acepta `title_grid_size`** (con typo) por compatibilidad, y si
  se pasa, sobrescribe a `tile_grid_size`. No lo uses en código nuevo.
- **`vpx_reconstruct` exige `marker ⊆ mask`** punto a punto, o lanza
  `ValueError`. Y con `max_iterations` finito puede devolver una reconstrucción
  parcial sin ninguna señal de que se truncó.
- **`vpx_skeletonize` usa padding de ceros** mientras el resto usa reflejo. Los
  resultados en el borde no son comparables.

## Rendimiento

El costo es `O(iterations · H · W · |kernel activo|)` en bucles Python. Una
imagen de 1024×1024 con kernel 5×5 son 26 millones de reducciones: minutos, no
segundos.

Estrategias, en orden de efectividad:

1. **Recorta la región de interés** antes de operar
2. **Submuestrea** para explorar parámetros, y corre una sola vez a resolución
   completa
3. **Kernel más chico**: el costo es lineal en el número de celdas activas
4. **`kernel_cross` en vez de `kernel_square`**: 5 celdas activas contra 9

## Manejo de errores

Todas las validaciones lanzan `ValueError` con mensajes estables. Los tests
casan contra el texto literal, así que son contrato público — la tabla completa
está en [api_reference.md](./api_reference.md#catálogo-de-errores).

Las excepciones a esa regla, y por lo tanto lo que hay que envolver a mano:

- `read_grayscale` → `None` silencioso
- `apply_clahe` y `segment_otsu` → `cv2.error` / excepciones de skimage sin
  traducir
- las `gray_*` con listas → `AttributeError`

## Ver también

- [api_reference.md](./api_reference.md) — firmas, contratos y errores
- [cli_reference.md](./cli_reference.md) — el comando `vispyx`
- [binary_morphology_usage.md](./binary_morphology_usage.md)
- [grayscale_morphology_usage.md](./grayscale_morphology_usage.md)
- [architecture.md](./architecture.md) — cómo está armado por dentro
- [testing.md](./testing.md) — estado real de la suite
