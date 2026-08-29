# CLI Reference

Referencia completa del comando `vispyx`, generado por el entry point
`vispyx = "vispyx.cli:main"` declarado en `pyproject.toml`.

## Instalación del comando

El comando solo existe tras instalar el paquete:

```bash
pip install -e .
vispyx --help
```

`vispyx/cli.py` **no tiene** bloque `if __name__ == "__main__":`, por lo que
`python vispyx/cli.py ...` y `python -m vispyx.cli ...` no ejecutan nada: solo
importan el módulo. La única forma soportada de invocar el CLI es el comando
`vispyx` instalado (o llamar `vispyx.cli.main()` desde Python).

## Forma del comando

```text
vispyx <method> <image_path> [opciones]
```

El CLI **no usa subparsers**. Es un único `ArgumentParser` plano donde el
"subcomando" es el primer argumento posicional `method`, restringido por
`choices`. Consecuencias prácticas:

- no existe `vispyx clahe --help` con ayuda propia: todos los métodos comparten
  el mismo bloque de flags
- no hay validación cruzada flag/método, salvo el caso especial de `--mask`
  para `vpx_reconstruct`
- las flags que un método no usa se ignoran en silencio (por ejemplo
  `--clip` en `vpx_erode`)

## Argumentos posicionales

| Posición | Nombre | Tipo | Descripción |
|---|---|---|---|
| 1 | `method` | `str` (restringido) | Método de procesamiento |
| 2 | `image_path` | `str` | Imagen de entrada. En `vpx_reconstruct` es el **marker**, no la máscara |

## Métodos disponibles (17)

```text
clahe   otsu
vpx_erode   vpx_dilate   vpx_open   vpx_close   vpx_gradient
vpx_tophat   vpx_blackhat   vpx_boundary   vpx_hitmiss
vpx_reconstruct   vpx_skeletonize   vpx_thin
gray_erode   gray_dilate   gray_open   gray_close
gray_gradient   gray_tophat   gray_blackhat
```

Un valor fuera de la lista produce error de `argparse` y salida `2`:

```text
vispyx: error: argument method: invalid choice: 'foo' (choose from 'clahe', ...)
```

Nota: `vpx_tophat`, `vpx_blackhat`, `vpx_boundary` y `vpx_hitmiss` **existen en
la API de Python pero no están expuestos en el CLI**.

## Flags

| Flag | `dest` | Tipo | Default | Métodos que la usan |
|---|---|---|---|---|
| `--mask`, `--mask-path` | `mask_path` | `str` | `None` | `vpx_reconstruct` (obligatoria ahí) |
| `--output`, `-o` | `output` | `str` | `None` | todos |
| `--show` | `show` | flag | `False` | todos |
| `--clip` | `clip` | `float` | `2.0` | `clahe` |
| `--grid` | `grid` | `int` | `8` | `clahe` (se usa como `(grid, grid)`) |
| `--kernel-size` | `kernel_size` | `int` | `3` | todos los `vpx_*`/`gray_*` con kernel |
| `--kernel` | `kernel_size` | `int` | alias | igual que `--kernel-size` |
| `--kernel-shape` | `kernel_shape` | `str` | `square` | todos los `vpx_*`/`gray_*` con kernel |
| `--iterations` | `iterations` | `int` | `1` | `vpx_erode/dilate/open/close/gradient`, `vpx_thin`, todos los `gray_*` |
| `--max-iterations` | `max_iterations` | `int` | `None` | `vpx_reconstruct`, `vpx_skeletonize` |

`--kernel` y `--kernel-size` comparten `dest`. El default efectivo es `3`
porque `--kernel-size` se registra primero; pasar `--kernel N` sobrescribe
correctamente el valor.

### `--kernel-shape`

Los valores válidos son `square`, `cross`, `diamond` y `disk`; cualquier otro lo
rechaza `argparse` con salida `2`. Cada uno delega en el generador homónimo de
`vispyx.kernels`:

| Valor | Kernel construido |
|---|---|
| `square` (default) | `kernel_square(--kernel-size)` |
| `cross` | `kernel_cross(--kernel-size)` |
| `diamond` | `kernel_diamond(--kernel-size)` |
| `disk` | `kernel_disk(--kernel-size // 2)` |

**El disco es el caso raro**: `kernel_disk` toma un **radio**, no un lado, así
que el CLI deriva `radio = --kernel-size // 2`. `--kernel-shape disk
--kernel-size 5` es `kernel_disk(2)`, que mide 5×5. El lado pedido y el lado
obtenido coinciden porque el tamaño se exige impar — ver abajo.

**Las cuatro formas exigen `--kernel-size` impar**, el disco incluido. Sin esa
validación un `4` daría radio `2` y **el mismo disco que un `5`**, en silencio,
mientras las otras tres formas lo rechazan.

**Formas distintas no siempre dan kernels distintos.** Para radios chicos
coinciden entre sí:

| Tamaño | Qué coincide |
|---|---|
| `3` | `cross` == `diamond` == `disk` |
| `5` | `diamond` == `disk` |
| `7` | las cuatro difieren |

Si estás comparando formas y no ves diferencia, sube el tamaño antes de suponer
que la flag no funciona.

## Qué hace cada método

Todas las rutas leen la imagen con `vispyx.utils.read_grayscale`, la misma
función que usa la API pública (siempre escala de grises). Los métodos `vpx_*` además **rebinarizan** la entrada con
`(img > 0) * 255`: cualquier píxel que no sea negro puro pasa a 255. Ojo: esa
binarización **no usa Otsu**; si vienes de una imagen de grises, pasa primero
por `vispyx otsu`.

| Método | Llamada interna | Flags que consume |
|---|---|---|
| `clahe` | `apply_clahe(img, clip_limit=--clip, tile_grid_size=(--grid, --grid))` | `--clip`, `--grid` |
| `otsu` | `segment_otsu(img)` | ninguna |
| `vpx_erode` | `vpx_erode(binary, kernel, iterations)` | `--kernel-size`, `--kernel-shape`, `--iterations` |
| `vpx_dilate` | `vpx_dilate(binary, kernel, iterations)` | `--kernel-size`, `--kernel-shape`, `--iterations` |
| `vpx_open` | `vpx_open(binary, kernel, iterations)` | `--kernel-size`, `--kernel-shape`, `--iterations` |
| `vpx_close` | `vpx_close(binary, kernel, iterations)` | `--kernel-size`, `--kernel-shape`, `--iterations` |
| `vpx_gradient` | `vpx_gradient(binary, kernel, iterations)` | `--kernel-size`, `--kernel-shape`, `--iterations` |
| `vpx_tophat` | `vpx_tophat(binary, kernel, iterations)` | `--kernel-size`, `--kernel-shape`, `--iterations` |
| `vpx_blackhat` | `vpx_blackhat(binary, kernel, iterations)` | `--kernel-size`, `--kernel-shape`, `--iterations` |
| `vpx_boundary` | `vpx_boundary(binary, kernel, iterations)` | `--kernel-size`, `--kernel-shape`, `--iterations` |
| `vpx_hitmiss` | `vpx_hitmiss(binary, hit, miss)` del patrón elegido | `--pattern` (**obligatoria**) |
| `vpx_reconstruct` | `vpx_reconstruct(marker, mask, kernel, max_iterations)` | `--mask`, `--kernel-size`, `--kernel-shape`, `--max-iterations` |
| `vpx_skeletonize` | `vpx_skeletonize(binary, max_iterations)` | `--max-iterations` |
| `vpx_thin` | `vpx_thin(binary, iterations)` | `--iterations` |
| `gray_*` | la función `gray_*` correspondiente, **sin binarizar** | `--kernel-size`, `--kernel-shape`, `--iterations` |

Detalles que sorprenden si no se leen:

- `vpx_reconstruct` ignora `--iterations`; su límite es `--max-iterations`
- `vpx_skeletonize` ignora `--kernel-size`, `--kernel-shape` y `--iterations`
- `vpx_thin` ignora `--kernel-size`, `--kernel-shape` y `--max-iterations`, y con el default
  `--iterations 1` ejecuta **una sola pasada** de Zhang-Suen (adelgazamiento
  parcial, no esqueleto completo)
- el CLI construye un cuadrado de unos **solo si se omite `--kernel-shape`**.
  Los cuatro generadores de `vispyx.kernels` son alcanzables desde la línea de
  comandos; lo único que sigue fuera de alcance son los kernels no cuadrados
  (`3×5`), que la API acepta y la flag no puede expresar

## `--pattern`: hit-or-miss sin escribir kernels

`vpx_hitmiss` toma **dos** estructurantes y no toma `iterations`, así que no
entra en el molde de `--kernel-shape`, que expresa una sola forma. El CLI lo
expone por patrones con nombre, cada uno con su par `hit`/`miss` ya escrito.

| `--pattern` | Detecta |
|---|---|
| `corner` | las cuatro esquinas: la unión de las cuatro orientaciones |
| `corner-nw`, `corner-ne`, `corner-se`, `corner-sw` | una esquina cada uno |
| `isolated` | píxeles sin ningún vecino, **ni siquiera en diagonal** |

```bash
vispyx vpx_hitmiss mascara.pgm --pattern corner -o esquinas.pgm
```

- **`--pattern` es obligatoria** para `vpx_hitmiss`: omitirla termina con código
  `2` y el mensaje `--pattern es obligatorio para vpx_hitmiss`
- `--kernel-size`, `--kernel-shape` e `--iterations` se **ignoran**, igual que
  `vpx_skeletonize` ya ignora `--kernel-size`
- los pares del catálogo viven en `PATTERNS`, en `cli.py`. Cumplen por
  construcción las tres reglas de `validate_hitmiss_kernels` — misma forma, sin
  solapamiento, y con al menos un elemento activo cada uno — y hay un test que
  recorre el catálogo entero para que un par inválido falle al agregarlo, no al
  usarlo
- **para un par propio hay que usar la API de Python.** El CLI no expresa pares
  arbitrarios a propósito: derivar el `miss` del complemento del `hit` falla con
  `--kernel-shape square`, que es el default, porque el complemento de un
  cuadrado de unos es todo ceros

## Salida

```python
if args.output:
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    try:
        guardada = cv2.imwrite(args.output, result)
    except cv2.error:
        parser.error(f"No se pudo guardar la imagen en {args.output}: "
                     "OpenCV no reconoce la extension")
    if not guardada:
        parser.error(f"No se pudo guardar la imagen en {args.output}: "
                     "no se pudo abrir el archivo para escritura")
    print(f"Imagen guardada en: {args.output}")
```

- crea el directorio de salida si `--output` incluye una ruta con carpetas. Si
  no puede crearlo — el padre no lo permite, o el padre no es un directorio —
  falla con código **2** antes de procesar nada más
- el formato lo decide la extensión del archivo (`cv2.imwrite`)
- **un guardado que falla nunca se anuncia como exitoso**. `cv2.imwrite` falla
  de dos maneras distintas y ninguna se ignora: devuelve `False` cuando no logra
  abrir el archivo (permisos, la ruta es un directorio) y lanza `cv2.error`
  cuando no tiene codec para la extensión. Las dos salen por `parser.error`, con
  código **2** y mensaje en `stderr`, y `--show` ya no se ejecuta
- sin `--output` imprime `Imagen procesada. No se guardó.`
- `--output` y `--show` son combinables: primero guarda, después muestra

`--show` usa matplotlib con backend `TkAgg`, forzado a nivel de módulo
(`matplotlib.use("TkAgg")` al importar `cli.py`). En un entorno sin display ni
Tkinter, `--show` falla.

## Errores y códigos de salida

`main()` no tiene `try/except`. Hay dos comportamientos distintos:

**Salida 2, mensaje limpio (errores de `argparse`)**

- método inválido, falta de argumentos, tipo incorrecto (`--clip abc`)
- `vpx_reconstruct` sin `--mask`:
  `vispyx: error: --mask es obligatorio para vpx_reconstruct`

**Salida 1, traceback crudo (excepciones no capturadas)**

| Error | Origen |
|---|---|
| `FileNotFoundError: No se encontró la imagen en <ruta>` | `read_grayscale`, cuando no hay ningún archivo en la ruta |
| `ValueError: No se pudo decodificar la imagen en <ruta>` | `read_grayscale`, cuando el archivo existe pero no es una imagen legible |
| `ValueError: --kernel-size debe ser un entero positivo` | `_build_kernel`, con `kernel_size <= 0` |
| `ValueError: size must be odd` | `_validate_size` en `kernels.py`, con `--kernel-size` **par**, en cualquiera de las cuatro formas. Falla al construir el kernel, **antes** de leer la imagen |
| `ValueError: Forma de kernel no reconocida: <valor>` | `_build_kernel` llamado desde Python con una forma que `argparse` no filtró. Por el CLI es inalcanzable: ahí lo ataja `choices` con salida `2` |
| `ValueError: iterations must be a positive integer` | `validate_iterations`, con `--iterations 0` o `--max-iterations 0` |
| `ValueError: marker must be a subset of mask` | `vpx_reconstruct`, marcador fuera de la máscara |
| `ValueError: marker and mask must have the same shape` | `vpx_reconstruct`, formas distintas |
| `cv2.error` | OpenCV, p. ej. `--grid 0` en CLAHE |

Resumen: solo los errores de parseo salen amigables. Todo error de dominio sale
como traceback.

## Ejemplos

```bash
# contraste
vispyx clahe imagen.pgm --clip 3.0 --grid 8 -o outputs/clahe.pgm

# segmentación (ver el resultado sin guardarlo)
vispyx otsu imagen.pgm --show

# erosión binaria, kernel 5x5, dos iteraciones
vispyx vpx_erode mascara.pgm --kernel 5 --iterations 2 -o outputs/erode.pgm

# apertura y cierre
vispyx vpx_open mascara.pgm --kernel-size 3 -o outputs/open.pgm
vispyx vpx_close mascara.pgm --kernel-size 3 -o outputs/close.pgm

# gradiente morfológico
vispyx vpx_gradient mascara.pgm --kernel-size 3 -o outputs/gradient.pgm

# reconstrucción: el posicional es el MARKER, la máscara va en --mask
vispyx vpx_reconstruct marker.pgm --mask mask.pgm --kernel-size 3 -o outputs/rec.pgm

# esqueleto completo, y esqueleto acotado
vispyx vpx_skeletonize mascara.pgm -o outputs/skeleton.pgm
vispyx vpx_skeletonize mascara.pgm --max-iterations 5 -o outputs/skeleton_parcial.pgm

# adelgazamiento parcial: una pasada
vispyx vpx_thin mascara.pgm --iterations 1 -o outputs/thin.pgm

# morfología en grises (sin binarizar)
vispyx gray_close imagen.pgm --kernel-size 3 -o outputs/gray_close.pgm
vispyx gray_tophat imagen.pgm --kernel-size 5 -o outputs/tophat.pgm

# forma del elemento estructurante: la cruz preserva las esquinas
vispyx vpx_erode mascara.pgm --kernel-size 5 --kernel-shape cross -o outputs/cross.pgm

# disco de radio 2 (5x5): el radio sale de --kernel-size // 2
vispyx vpx_open mascara.pgm --kernel-size 5 --kernel-shape disk -o outputs/disk.pgm

# para separar diamante de disco hace falta 7: en 5 son el mismo kernel
vispyx gray_tophat imagen.pgm --kernel-size 7 --kernel-shape diamond -o outputs/diamond.pgm
```

## Pipeline típico en una línea de comandos

```bash
vispyx clahe mdb001.pgm --clip 3.0 --grid 8 -o out/1_clahe.pgm
vispyx gray_close out/1_clahe.pgm --kernel-size 3 -o out/2_smooth.pgm
vispyx otsu out/2_smooth.pgm -o out/3_mask.pgm
vispyx vpx_open out/3_mask.pgm --kernel 3 -o out/4_clean.pgm
vispyx vpx_skeletonize out/4_clean.pgm -o out/5_skeleton.pgm
```

## Ver también

- [api_reference.md](./api_reference.md)
- [system_usage.md](./system_usage.md)
