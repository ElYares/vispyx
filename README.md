<p align="center">
  <img src="assets/vispyx.png" alt="vispyx cover" width="420">
</p>

<h1 align="center">vispyx</h1>

<p align="center">
  Technical image processing with a morphology core implemented from scratch.
</p>

`vispyx` es un paquete de procesamiento de imágenes en Python, orientado a flujos de trabajo técnicos (médico, industrial y científico), con énfasis en **morfología implementada desde cero**.

## Características

- Preprocesamiento de contraste con `CLAHE`
- Segmentación binaria con umbral de `Otsu`
- Operaciones morfológicas propias (`vpx_*`):
  - `vpx_erode`
  - `vpx_dilate`
  - `vpx_open`
  - `vpx_close`
  - `vpx_gradient`
- CLI para ejecutar pipelines sin escribir código

## Estado del Proyecto

Versión actual: `0.1.0`  
Enfoque actual: estabilización de core de procesamiento y calidad de interfaz/uso.

## Requisitos

- Python `>= 3.7`
- Dependencias principales:
  - `opencv-python`
  - `numpy`
  - `scikit-image`
  - `matplotlib`

## Instalación

```bash
pip install -e .
```

## Uso por CLI

Sintaxis general:

```bash
vispyx <method> <image_path> [flags]
```

Métodos disponibles:

- `clahe`
- `otsu`
- `vpx_erode`
- `vpx_dilate`
- `vpx_open`
- `vpx_close`
- `vpx_gradient`

Ejemplos:

```bash
# CLAHE
vispyx clahe archive/all-mias/mdb001.pgm --clip 3.0 --grid 8 --output outputs/mdb001_clahe.pgm

# Otsu
vispyx otsu archive/all-mias/mdb001.pgm --output outputs/mdb001_otsu.pgm

# Morfología desde cero (alias --kernel)
vispyx vpx_erode archive/all-mias/mdb001.pgm --kernel 5 --iterations 2 --output outputs/mdb001_erode.pgm
```

## Uso por API (Python)

```python
from vispyx.preprocessing import apply_clahe
from vispyx.segmentation import segment_otsu
from vispyx.morphology import vpx_erode
from vispyx.utils import read_grayscale
import numpy as np

img = read_grayscale("archive/all-mias/mdb001.pgm")
clahe = apply_clahe(img, clip_limit=3.0, tile_grid_size=(8, 8))
binary = segment_otsu(clahe)
kernel = np.ones((5, 5), dtype=np.uint8)
eroded = vpx_erode(binary, kernel=kernel, iterations=1)
```

## Estructura

```text
vispyx/
├── vispyx/
│   ├── cli.py
│   ├── preprocessing.py
│   ├── segmentation.py
│   ├── morphology.py
│   └── utils.py
├── test/
├── examples/
└── docs/
```

## Testing

```bash
pytest -q
```

## Licencia

MIT License. Ver [LICENSE](./LICENSE).
