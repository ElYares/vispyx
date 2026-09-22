# vispyx

Paquete Python de procesamiento de imágenes con un núcleo de morfología
matemática **implementado desde cero**. Versión `0.5.0`, estado alpha. Trae un
backend opcional en Rust (`native/`, distribución aparte `vispyx-native`) que
acelera las 19 operaciones con resultados idénticos bit a bit.

## Regla que manda sobre todo

De `CONTRIBUTING.md`: *"do not introduce external packages to perform the
morphological operations themselves"*. OpenCV, scikit-image y matplotlib se usan
para leer archivos, CLAHE, el umbral de Otsu y mostrar imágenes — nunca para
erosionar, dilatar ni esqueletizar. Si una tarea parece pedir
`cv2.morphologyEx` o `scipy.ndimage`, la respuesta es escribir el algoritmo.

## Estructura

```text
vispyx/
├── _backend.py              elige el motor: VISPYX_BACKEND=auto|python|rust
├── morphology_common.py     validaciones + los dos motores de ventana deslizante
├── morphology_binary.py     12 operaciones vpx_* (0/255)
├── morphology_grayscale.py  7 operaciones gray_* (dtype nativo)
├── morphology.py            fachada de compatibilidad, solo imports
├── kernels.py               kernel_square/cross/diamond/disk
├── preprocessing.py         apply_clahe (OpenCV)
├── segmentation.py          segment_otsu (skimage)
├── utils.py                 read_grayscale, show_image
├── cli.py                   comando `vispyx`, 21 métodos
├── __main__.py              `python -m vispyx`
└── __init__.py              superficie pública, 28 símbolos
native/src/lib.rs            el backend en Rust: 3 funciones, sin validación
.github/workflows/ci.yml     suite en los dos motores, wheels, publicación
```

`morphology_common.py` es el corazón: tocarlo cambia las 19 operaciones a la vez.
Los bucles de Python son la **implementación de referencia**: el Rust tiene que
coincidir con ellos bit a bit, y `test/test_backend_parity.py` lo exige.

## Comandos

```bash
VIRTUAL_ENV=.venv uv pip install -e '.[dev]'    # el .venv es de uv, no trae pip
cd native && maturin develop --release && cd ..  # el nativo, opcional
pytest -q            # 1079 con el nativo; sin él, 436 pasan y 6 se saltan
VISPYX_BACKEND=python pytest -q                  # la referencia, con el nativo puesto
vispyx --help
python native/bench.py                           # Python contra Rust
```

Después de tocar `native/src/lib.rs` hay que recompilar: los tests corren contra
el `.so` instalado, no contra el fuente.

## Convenciones no negociables

- **Dos dominios de valores.** `vpx_*` binariza la entrada con `> 0` y devuelve
  siempre `uint8` en `{0, 255}`. `gray_*` no binariza y conserva el dtype de
  entrada. `segment_otsu` es el puente entre ambos.
- **Toda validación lanza `ValueError`**, nunca `TypeError` ni `assert`.
- **Los mensajes de error son contrato público.** Los tests casan contra el
  texto literal con `match=`. Cambiar una palabra rompe la suite.
- **Kernels con ambas dimensiones impares.** No hace falta que sean cuadrados.
- **Padding por reflejo** en todo, salvo Zhang-Suen (`vpx_skeletonize`/
  `vpx_thin`), que usa ceros a propósito.
- **Los bucles Python no se vectorizan.** La lentitud es el precio explícito de
  que el algoritmo sea legible. La velocidad viene del backend en Rust, que no
  valida ni lanza mensajes: todo error sale de Python, antes de llegar al nativo.
- **Todo error de entrada del CLI sale con código 2** y una línea en stderr.
  `ValueError`/`FileNotFoundError` se convierten con `parser.error`; cualquier
  otra excepción es un bug y sale con traceback, a propósito.

## Agregar una operación

Toca cinco lugares, en este orden: la implementación en `morphology_binary.py` o
`morphology_grayscale.py` → `__all__` de `morphology.py` → `__all__` de
`__init__.py` → `expected_symbols` en `test/test_public_api.py` → test de valor
exacto en `test/test_morphology.py`. Si además va al CLI: lista `methods`,
despacho con `_run_binary_method` o `_run_grayscale_method`, y
`docs/cli_reference.md`. Cierra con `CHANGELOG.md`. Solo necesita su propia
`run_*` si no toma `(image, kernel, iterations)`.

## Trampas conocidas

- `vpx_thin(img)` con el default hace **una pasada**, no el esqueleto completo
- **Las cuatro formas de kernel coinciden entre sí para radios chicos**: en `3`
  cruz, diamante y disco son la misma matriz; en `5` lo son diamante y disco.
  Un test de forma escrito con `5` deja pasar que esas dos ramas estén
  cambiadas. Para separar las cuatro hace falta `7`
- `--kernel-shape disk` deriva el radio como `size // 2`, porque `kernel_disk`
  toma radio y no lado. La paridad se valida aparte en `cli.py`: sin eso, un `4`
  daría el mismo disco que un `5` en silencio
- **Una imagen vacía se rechaza con `image must not be empty`**, igual que un
  kernel vacío. Antes cada función fallaba distinto y `apply_clahe` **se
  colgaba**. Si se quita esa validación, los tests de imagen vacía no fallan:
  cuelgan
- `iterations=True` es rechazado a propósito, aunque `bool` sea subclase de
  `int`. Los enteros de NumPy sí se aceptan
- `read_grayscale` lanza: `FileNotFoundError` si no hay archivo, `ValueError` si
  lo hay pero no se puede decodificar. El CLI usa esa misma función
- `iterations=n` en `open`/`close` significa *n erosiones y luego n
  dilataciones*, no *n aperturas*
- `vispyx.__version__` está clavado en un test: subir la versión sin actualizar
  `test_public_api.py` rompe la suite. La versión aparece en ~12 archivos:
  `grep -rn "0\.5\.0"` antes de subirla
- **Antes de creerle a un test de paridad, mutar el Rust** y ver que muera. Un
  kernel sólido no distingue reflejo de repetición de borde: los tests de borde
  necesitan kernels con hueco. Y como las dos rutas dan lo mismo, un test de
  paridad no sabe qué ruta corrió (por ejemplo, van Herk o `sweep`)
- Con `VISPYX_BACKEND=rust`, `test_backend_parity.py` **falla** si el nativo no
  está instalado, en vez de saltarse. Es lo que usa el CI
- `cli.py` cambia matplotlib a TkAgg **solo con `--show`**. Forzarlo al importar
  rompía el comando entero en una máquina sin display
- `morph_scipy.py` (raíz) está fuera del paquete instalable, pero
  `test/test_reference_scipy.py` lo usa como oráculo de referencia. El
  `conftest.py` de la raíz existe solo para que ese import funcione

## Documentación

`docs/` tiene la documentación completa. Entrada: `docs/README.md`. Para tocar el
código, leer antes `docs/architecture.md` y `docs/testing.md`.
