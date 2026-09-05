# vispyx-native

Backend opcional en Rust para el motor de morfología binaria de
[`vispyx`](https://github.com/ElYares/vispyx).

No es un reemplazo del núcleo. `vispyx` sigue siendo un paquete puro de Python
y sus bucles legibles siguen siendo la implementación de referencia; este crate
recorre exactamente el mismo algoritmo en Rust y produce resultados **idénticos
bit a bit**. Si no está instalado, `vispyx` funciona igual, solo más lento.

## Alcance actual (spike)

| Operación | Estado |
|---|---|
| `vpx_erode`, `vpx_dilate` | nativo |
| `vpx_open`, `vpx_close`, `vpx_gradient`, `vpx_tophat`, `vpx_blackhat`, `vpx_boundary`, `vpx_hitmiss`, `vpx_reconstruct` | nativo por composición: se apoyan en las dos anteriores |
| `gray_*` | Python |
| `vpx_skeletonize`, `vpx_thin` | Python |

## Instalación

Todavía no está publicado en PyPI, así que se instala desde este directorio:

```bash
pip install maturin
cd native && maturin develop --release
```

Requiere toolchain de Rust (`rustc`, `cargo`). Ese es justamente el motivo de
que viva en una distribución aparte: `pip install vispyx` nunca necesita un
compilador.

## Selección de backend

`vispyx` elige solo, pero `VISPYX_BACKEND` manda:

```bash
VISPYX_BACKEND=auto   pytest -q   # nativo si está, Python si no (default)
VISPYX_BACKEND=python pytest -q   # fuerza los bucles de Python
VISPYX_BACKEND=rust   pytest -q   # falla al importar si el nativo no está
```

## Qué garantiza la paridad

`test/test_backend_parity.py` corre las mismas entradas por los dos backends y
exige igualdad exacta, incluyendo los casos que el resto de la suite evita: las
que **tocan el borde**, donde el padding por reflejo es el único responsable, y
kernels más grandes que la propia imagen, donde el reflejo se pliega varias
veces.

## Lo que el crate no hace

No valida. `vispyx.morphology_common` valida y normaliza antes de llamar, y
todos los mensajes de error siguen viniendo de Python, donde son contrato
público que los tests casan literalmente.
