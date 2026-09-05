# Backend nativo en Rust

Estado: **spike**. Cubre `vpx_erode` y `vpx_dilate`; el resto del paquete no
cambió. Distribución aparte, opcional, en `native/`.

## Qué problema resuelve

Los bucles de Python no se vectorizan a propósito: la legibilidad del algoritmo
vale más que la velocidad, y esa decisión no se toca. Pero el costo crece con el
área y se vuelve prohibitivo antes de lo que uno quisiera.

```text
caso                    tamaño     python       rust   speedup
--------------------------------------------------------------
vpx_erode  3x3  x1     64x64      0.0185s    0.0003s       61x
vpx_erode  3x3  x1    128x128     0.0723s    0.0002s      301x
vpx_erode  3x3  x1    256x256     0.2936s    0.0008s      376x
vpx_erode  5x5  x1    256x256     0.2943s    0.0007s      409x
vpx_open   3x3  x2    256x256     1.1710s    0.0024s      478x
```

Reproducible con `python native/bench.py`.

Un `vpx_open` con dos iteraciones sobre 512×512 pasa de unos 5 segundos a unos
10 milisegundos. Ese es el punto: no es una optimización marginal, es la
diferencia entre poder usar el paquete sobre una imagen real y no poder.

## Qué **no** es

No es un reemplazo del núcleo. Los bucles de Python siguen ahí, sin tocar, y
siguen siendo la implementación de referencia. Rust recorre el mismo algoritmo y
produce el mismo resultado bit a bit; si el nativo no está instalado, todo
funciona igual, solo más lento.

Tampoco contradice la regla de `CONTRIBUTING.md` — *"do not introduce external
packages to perform the morphological operations themselves"*. El algoritmo está
escrito desde cero, igual que antes. Lo único que cambió es el lenguaje en el
que corre la ruta rápida.

## Cómo se instala

```bash
pip install maturin
cd native && maturin develop --release
```

Requiere `cargo` y `rustc`. Vive en una distribución aparte (`vispyx-native`)
por exactamente ese motivo: **`pip install vispyx` nunca debe necesitar un
compilador**. El extra declarado en `pyproject.toml` es `vispyx[fast]`, pendiente
de publicación en PyPI.

## Cómo se elige

`vispyx` detecta el nativo al importar. `VISPYX_BACKEND` manda sobre eso:

| Valor | Comportamiento |
|---|---|
| `auto` (default) | nativo si está instalado, Python si no |
| `python` | fuerza los bucles de Python |
| `rust` | exige el nativo; `ImportError` al importar `vispyx` si falta |

```bash
VISPYX_BACKEND=python pytest -q
VISPYX_BACKEND=rust   pytest -q
```

Un valor desconocido lanza `ValueError`, igual que cualquier otra validación del
paquete. La resolución es perezosa a propósito: si fuera al importar, un valor
mal escrito rompería `import vispyx`, mucho antes y mucho más lejos del lugar
donde el usuario puede hacer algo al respecto.

### Desde el CLI

`vispyx --version` es lo único que responde "¿qué motor tengo?" sin correr nada:

```bash
$ vispyx --version
vispyx 0.4.0 (backend: rust, vispyx-native 0.1.0)
```

`--backend {auto,python,rust}` elige el motor para esa invocación y tiene
prioridad sobre la variable de entorno. `--time` reporta cuánto tardó y con qué
motor. `--compare` corre los dos, mide y verifica que coincidan:

```bash
$ vispyx vpx_open mask.png --kernel-size 7 --kernel-shape disk --compare -o out.png
backend         tiempo    relativo
python        11.9268s        1.0x
rust           0.0491s      242.7x
resultados identicos: si
```

Vale más que cronometrar dos invocaciones desde la shell: el arranque del
intérprete —cerca de un segundo entre numpy, OpenCV, scikit-image y
matplotlib— queda fuera de la medición. En imágenes chicas ese segundo tapa por
completo la diferencia entre los motores.

Detalle de las tres: `--backend rust` sin el paquete instalado **falla** en vez
de caer a Python en silencio. Pedir un motor y recibir otro invalida cualquier
medición, que es justo para lo que existen estas flags.

Los detalles están en [cli_reference.md](./cli_reference.md).

## Qué cubre hoy

| Operación | Motor |
|---|---|
| `vpx_erode`, `vpx_dilate` | Rust |
| `vpx_open`, `vpx_close`, `vpx_gradient`, `vpx_tophat`, `vpx_blackhat`, `vpx_boundary`, `vpx_hitmiss`, `vpx_reconstruct` | Rust por composición: se apoyan en las dos anteriores |
| las 7 `gray_*` | Python |
| `vpx_skeletonize`, `vpx_thin` | Python |

Las compuestas no necesitaron una sola línea: ya estaban escritas como
composición explícita de erosión y dilatación, y heredaron la aceleración
completa. Es el dividendo de que `morphology_binary.py` no repitiera el motor.

## Dónde se engancha

Un solo lugar, `apply_binary_operation` en `morphology_common.py`:

```python
if native_op is not None:
    backend = _backend.native()
    if backend is not None:
        return backend.binary_op(img, kernel, int(iterations), native_op) * 255
```

Va **después** de las tres validaciones y antes del bucle. Eso fija la frontera:

- el nativo recibe la imagen ya binarizada a `{0, 1}` y el kernel ya normalizado;
- el nativo **no valida y no lanza `ValueError`**. Todos los mensajes de error
  siguen viniendo de Python, donde son contrato público que los tests casan
  literalmente;
- el `* 255` se hace en Python, así que la convención de dominio de valores vive
  en un solo lado.

`vpx_erode` y `vpx_dilate` pasan `native_op="erode"` / `"dilate"`. Una operación
sin `native_op` cae al bucle de Python sin ninguna rama extra.

## Los tres detalles que rompen un port así

1. **Padding por reflejo.** `np.pad(mode="reflect")` espeja *sin repetir el
   borde*: `[1, 2, 3]` con pad 1 es `[2, 1, 2, 3, 2]`. No es `edge` ni
   `symmetric`. En Rust es `index.rem_euclid(2 * (n - 1))` seguido de un pliegue,
   y el `rem_euclid` primero importa: con un kernel 7×7 sobre un eje de 2
   píxeles, el reflejo se pliega varias veces.
2. **El padding se recalcula en cada iteración**, dentro del bucle, igual que en
   Python. Colapsar `iterations` en una ventana más grande da otro resultado en
   los bordes.
3. **Ejes de longitud 1.** `n == 1` no tiene período; se devuelve el índice 0.

## Qué garantiza la paridad

`test/test_backend_parity.py` — 231 tests que corren la misma entrada por los
dos backends y exigen igualdad exacta de valores y de dtype. Más 13 en
`test_cli_main.py` para las tres flags nuevas, incluida la rama de divergencia,
que se alcanza reemplazando el despacho.

Es el complemento de `test_reference_scipy.py`, no una copia: aquel rodea cada
imagen con un marco de fondo para que la operación nunca alcance el borde,
porque scipy trata el exterior distinto. Este hace lo contrario a propósito.
El borde es donde el port se rompe, así que hay foreground pegado al margen, un
píxel solo en la esquina, kernels más grandes que la imagen, ejes de longitud 1,
imágenes saturadas y vacías, y una vista no contigua.

Se salta solo (`importorskip`) si el nativo no está instalado, e incluye un test
que verifica que el backend bajo prueba **es** el nativo: una suite verde que
nunca ejercitó Rust no prueba nada.

### Verificado por mutación, no por optimismo

La primera versión de este archivo se creía más fuerte de lo que era. Al
reemplazar a propósito el reflejo por `clamp` (repetición de borde) en el Rust y
recompilar, **ninguno de los 438 tests preexistentes falló**, y de los 131 de
paridad cayeron solo 15 — todos del mismo kernel.

El motivo es geométrico y vale la pena recordarlo: **un kernel sólido no
distingue el reflejo de la repetición de borde**. En la columna 0, el reflejo
muestrea `{img[1], img[0], img[1]}` y la repetición `{img[0], img[0], img[1]}`;
como conjunto son idénticos, y `min`/`max` no ven diferencia. Cruz, diamante y
disco tampoco sirven: son simétricos y contienen el centro.

Lo único que discrimina es un soporte que **excluya el centro** y sea
asimétrico. Por eso la lista de kernels incluye `[[1,0,1]]`, `[[1,0,0]]`,
`[[1],[0],[0]]` y un 3×3 con solo la esquina noroeste activa. Con esos cuatro
agregados, la misma mutación cae en 69 tests en vez de 15.

Es la misma trampa que documenta `CLAUDE.md` para los kernels de radio chico,
en otra forma: un test que parece cubrir el borde puede no estar mirándolo.

La suite completa corre limpia con los dos: 556 tests en ambos modos.

## Siguientes pasos

En orden de rendimiento por esfuerzo:

1. **`vpx_skeletonize`** (Zhang-Suen). Es el peor caso absoluto —dos pasadas
   completas por iteración hasta converger, 0.43 s en 128×128— y el único que no
   se beneficia en nada de este spike.
2. **Los dos motores `gray_*`.** `min`/`max` sobre el soporte activo. Requiere
   decidir el manejo de dtypes: lo razonable es `u8` nativo y caer a Python para
   los exóticos, en vez de hacer genérico el crate desde el día uno.
3. **`vpx_reconstruct` nativo.** Ya se acelera por composición, pero el bucle
   geodésico sigue en Python y paga un cruce de frontera por iteración.
4. **Liberar el GIL** en `binary_op`, para que el nativo no bloquee otros hilos
   durante la pasada.
5. **CI con wheels.** El repo no tiene `.github/` todavía. Antes de publicar
   `vispyx-native` hace falta una matriz de `maturin-action`, y correr la suite
   con `VISPYX_BACKEND` en `python` y `rust`.
