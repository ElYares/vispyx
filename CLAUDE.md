# vispyx

Paquete Python de procesamiento de imágenes con un núcleo de morfología
matemática **implementado desde cero**. Versión `0.2.1`, estado alpha.

## Regla que manda sobre todo

De `CONTRIBUTING.md`: *"do not introduce external packages to perform the
morphological operations themselves"*. OpenCV, scikit-image y matplotlib se usan
para leer archivos, CLAHE, el umbral de Otsu y mostrar imágenes — nunca para
erosionar, dilatar ni esqueletizar. Si una tarea parece pedir
`cv2.morphologyEx` o `scipy.ndimage`, la respuesta es escribir el algoritmo.

## Estructura

```text
vispyx/
├── morphology_common.py     validaciones + los dos motores de ventana deslizante
├── morphology_binary.py     12 operaciones vpx_* (0/255)
├── morphology_grayscale.py  7 operaciones gray_* (dtype nativo)
├── morphology.py            fachada de compatibilidad, solo imports
├── kernels.py               kernel_square/cross/diamond/disk
├── preprocessing.py         apply_clahe (OpenCV)
├── segmentation.py          segment_otsu (skimage)
├── utils.py                 read_grayscale, show_image
├── cli.py                   comando `vispyx`, 17 métodos
└── __init__.py              superficie pública, 28 símbolos
```

`morphology_common.py` es el corazón: tocarlo cambia las 19 operaciones a la vez.

## Comandos

```bash
pip install -e .[dev]
pytest -q            # 84 tests, ~1.6 s
vispyx --help
```

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
  que el algoritmo sea legible.

## Agregar una operación

Toca cinco lugares, en este orden: la implementación en `morphology_binary.py` o
`morphology_grayscale.py` → `__all__` de `morphology.py` → `__all__` de
`__init__.py` → `expected_symbols` en `test/test_public_api.py` → test de valor
exacto en `test/test_morphology.py`. Si además va al CLI: lista `methods`,
función `run_*`, y `docs/cli_reference.md`. Cierra con `CHANGELOG.md`.

## Trampas conocidas

- `vpx_thin(img)` con el default hace **una pasada**, no el esqueleto completo
- `iterations=True` es rechazado a propósito, aunque `bool` sea subclase de
  `int`. Los enteros de NumPy sí se aceptan
- `read_grayscale` devuelve `None` en silencio si falla la lectura; el CLI usa
  su propio `_read_grayscale`, que sí lanza
- `iterations=n` en `open`/`close` significa *n erosiones y luego n
  dilataciones*, no *n aperturas*
- `vispyx.__version__` está clavado en un test: subir la versión sin actualizar
  `test_public_api.py` rompe la suite
- `morph_scipy.py` (raíz) está fuera del paquete instalable, pero
  `test/test_reference_scipy.py` lo usa como oráculo de referencia. El
  `conftest.py` de la raíz existe solo para que ese import funcione

## Documentación

`docs/` tiene la documentación completa. Entrada: `docs/README.md`. Para tocar el
código, leer antes `docs/architecture.md` y `docs/testing.md`.
