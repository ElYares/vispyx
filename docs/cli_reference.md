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
| `--iterations` | `iterations` | `int` | `1` | `vpx_erode/dilate/open/close/gradient`, `vpx_thin`, todos los `gray_*` |
| `--max-iterations` | `max_iterations` | `int` | `None` | `vpx_reconstruct`, `vpx_skeletonize` |

`--kernel` y `--kernel-size` comparten `dest`. El default efectivo es `3`
porque `--kernel-size` se registra primero; pasar `--kernel N` sobrescribe
correctamente el valor.

## Qué hace cada método

Todas las rutas leen la imagen con `cv2.imread(path, 0)` (siempre escala de
grises). Los métodos `vpx_*` además **rebinarizan** la entrada con
`(img > 0) * 255`: cualquier píxel que no sea negro puro pasa a 255. Ojo: esa
binarización **no usa Otsu**; si vienes de una imagen de grises, pasa primero
por `vispyx otsu`.

| Método | Llamada interna | Flags que consume |
|---|---|---|
| `clahe` | `apply_clahe(img, clip_limit=--clip, tile_grid_size=(--grid, --grid))` | `--clip`, `--grid` |
| `otsu` | `segment_otsu(img)` | ninguna |
| `vpx_erode` | `vpx_erode(binary, kernel, iterations)` | `--kernel-size`, `--iterations` |
| `vpx_dilate` | `vpx_dilate(binary, kernel, iterations)` | `--kernel-size`, `--iterations` |
| `vpx_open` | `vpx_open(binary, kernel, iterations)` | `--kernel-size`, `--iterations` |
| `vpx_close` | `vpx_close(binary, kernel, iterations)` | `--kernel-size`, `--iterations` |
| `vpx_gradient` | `vpx_gradient(binary, kernel, iterations)` | `--kernel-size`, `--iterations` |
| `vpx_reconstruct` | `vpx_reconstruct(marker, mask, kernel, max_iterations)` | `--mask`, `--kernel-size`, `--max-iterations` |
| `vpx_skeletonize` | `vpx_skeletonize(binary, max_iterations)` | `--max-iterations` |
| `vpx_thin` | `vpx_thin(binary, iterations)` | `--iterations` |
| `gray_*` | la función `gray_*` correspondiente, **sin binarizar** | `--kernel-size`, `--iterations` |

Detalles que sorprenden si no se leen:

- `vpx_reconstruct` ignora `--iterations`; su límite es `--max-iterations`
- `vpx_skeletonize` ignora `--kernel-size` y `--iterations`
- `vpx_thin` ignora `--kernel-size` y `--max-iterations`, y con el default
  `--iterations 1` ejecuta **una sola pasada** de Zhang-Suen (adelgazamiento
  parcial, no esqueleto completo)
- el CLI construye siempre un kernel cuadrado de unos (`np.ones((n, n))`). Los
  generadores `kernel_cross`, `kernel_diamond` y `kernel_disk` **no están
  expuestos**; para usarlos hay que ir por la API de Python

## Salida

```python
if args.output:
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(args.output, result)
    print(f"Imagen guardada en: {args.output}")
```

- crea el directorio de salida si `--output` incluye una ruta con carpetas
- el formato lo decide la extensión del archivo (`cv2.imwrite`)
- **no se verifica el valor de retorno de `cv2.imwrite`**: si la escritura
  falla devolviendo `False`, el CLI igual imprime "Imagen guardada en: ..."
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
| `FileNotFoundError: No se encontró la imagen en <ruta>` | `_read_grayscale`, cuando `cv2.imread` devuelve `None` (archivo ausente **o** ilegible/corrupto: ambos casos dan el mismo mensaje) |
| `ValueError: --kernel-size debe ser un entero positivo` | `_build_kernel`, con `kernel_size <= 0` |
| `ValueError: kernel dimensions must be odd` | `validate_kernel`, con `--kernel-size` **par**. El CLI no lo valida antes, el error aparece ya dentro de la operación morfológica |
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
