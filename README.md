<p align="center">
  <img src="assets/vispyx.png" alt="vispyx cover" width="420">
</p>

<h1 align="center">vispyx</h1>

<p align="center">
  Technical image processing with a morphology core implemented from scratch.
</p>

`vispyx` es un paquete Python de procesamiento de imágenes orientado a flujos técnicos y científicos, con un núcleo de **morfología matemática implementada desde cero**.

## Overview

`vispyx` actualmente cubre:

- preprocesamiento con `CLAHE`
- segmentación binaria con `Otsu`
- morfología binaria `vpx_*`
- morfología grayscale `gray_*`
- kernels estructurantes reutilizables
- CLI para experimentación y pipelines rápidos

## Estado del Proyecto

Versión actual: `0.2.0`  
Estado: `alpha`  
Enfoque actual: consolidación del core morfológico, API pública y empaquetado.

## Requisitos

- Python `>= 3.7`

## Instalación

```bash
pip install -e .
```

`vispyx` usa empaquetado moderno con `pyproject.toml`.

Para instalar dependencias de desarrollo:

```bash
pip install -e .[dev]
```

## Quick Start

### API

```python
from vispyx import (
    apply_clahe,
    gray_close,
    kernel_disk,
    read_grayscale,
    segment_otsu,
    vpx_open,
)

img = read_grayscale("archive/all-mias/mdb001.pgm")
clahe = apply_clahe(img, clip_limit=3.0, tile_grid_size=(8, 8))
smoothed = gray_close(clahe, kernel=kernel_disk(1))
binary = segment_otsu(smoothed)
clean_mask = vpx_open(binary, kernel=kernel_disk(1), iterations=1)
```

### CLI

```bash
# CLAHE
vispyx clahe imagen.pgm --clip 3.0 --grid 8 --output outputs/clahe.pgm

# Segmentación Otsu
vispyx otsu imagen.pgm --output outputs/otsu.pgm

# Morfología binaria
vispyx vpx_open mascara.pgm --kernel 3 --output outputs/open.pgm

# Skeletonization
vispyx vpx_skeletonize mascara.pgm --output outputs/skeleton.pgm
```

## Documentation

- Guía integral: [docs/system_usage.md](./docs/system_usage.md)
- Referencia de API: [docs/api_reference.md](./docs/api_reference.md)
- Referencia de CLI: [docs/cli_reference.md](./docs/cli_reference.md)
- Morfología binaria: [docs/binary_morphology_usage.md](./docs/binary_morphology_usage.md)
- Morfología grayscale: [docs/grayscale_morphology_usage.md](./docs/grayscale_morphology_usage.md)
- Índice completo: [docs/README.md](./docs/README.md)

## Package Layout

```text
vispyx/
├── vispyx/
│   ├── __init__.py
│   ├── cli.py
│   ├── kernels.py
│   ├── morphology_common.py
│   ├── morphology_binary.py
│   ├── morphology_grayscale.py
│   ├── preprocessing.py
│   ├── segmentation.py
│   ├── morphology.py
│   └── utils.py
├── test/
├── examples/
└── docs/
```

## Development

```bash
pytest -q
```

Historial de cambios:

- [CHANGELOG.md](./CHANGELOG.md)

## Licencia

MIT License. Ver [LICENSE](./LICENSE).
