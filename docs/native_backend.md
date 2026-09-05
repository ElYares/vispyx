# Backend nativo en Rust

Estado: **spike**. Cubre los dos motores de ventana deslizante, binario y
grayscale. Distribución aparte, opcional, en `native/`.

## Qué problema resuelve

Los bucles de Python no se vectorizan a propósito: la legibilidad del algoritmo
vale más que la velocidad, y esa decisión no se toca. Pero el costo crece con el
área y se vuelve prohibitivo antes de lo que uno quisiera.

```text
caso                    tamaño     python       rust   speedup
--------------------------------------------------------------
vpx_erode   3x3  x1   256x256     0.3821s    0.0008s      469x
vpx_open    3x3  x2   256x256     1.2005s    0.0023s      516x
gray_erode  3x3  x1   256x256     0.2008s    0.0008s      242x
gray_erode  7x7  x1   256x256     0.2025s    0.0051s       40x
gray_erode 15x15 x1   256x256     0.2266s    0.0295s        8x
gray_open   3x3  x1   256x256     0.4465s    0.0017s      261x
```

Reproducible con `python native/bench.py`.

### El speedup cae con el tamaño del kernel, y no es un defecto

En Python el costo es **por píxel**: lo que domina es el overhead de numpy por
ventana —`region[active_mask]`, `np.min`—, no las celdas. Medido, un
`gray_erode` sobre 128×128 tarda lo mismo con kernel 3×3 (0.0564 s) que con
15×15 (0.0549 s), aunque el segundo tenga 25 veces más celdas.

En Rust se invierte: el costo pasa a ser proporcional a las celdas activas. De
ahí que el mismo port dé 242x en 3×3 y 8x en 15×15. Ocho veces sigue siendo una
mejora, pero conviene no vender el número del 3×3 como si fuera general.

Es también la puerta al siguiente paso real para kernels grandes: una
descomposición tipo van Herk / Gil-Werman haría el costo independiente del
tamaño del kernel.

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
| `gray_erode`, `gray_dilate` | Rust, dtypes enteros |
| `vpx_open`, `vpx_close`, `vpx_gradient`, `vpx_tophat`, `vpx_blackhat`, `vpx_boundary`, `vpx_hitmiss`, `vpx_reconstruct` | Rust por composición |
| `gray_open`, `gray_close`, `gray_gradient`, `gray_tophat`, `gray_blackhat` | Rust por composición |
| `gray_*` con dtype flotante | Python, a propósito |
| `vpx_skeletonize`, `vpx_thin` | Python |

**17 de 19 operaciones**, y solo cuatro funciones nativas. Las trece compuestas
no necesitaron una sola línea: ya estaban escritas como composición explícita de
erosión y dilatación, y heredaron la aceleración completa. Es el dividendo de
que ni `morphology_binary.py` ni `morphology_grayscale.py` repitieran el motor.

### Por qué los flotantes se quedan en Python

`Ord` en Rust es un orden total, y los flotantes no lo tienen. Reproducir bit a
bit la propagación de `NaN` de `np.min` no vale el riesgo en un spike, así que
el despacho mira `dtype.kind` y manda `float32`/`float64` al bucle de Python sin
avisar: mismo resultado, solo más lento. Los ocho dtypes enteros —`uint8`,
`int8`, `uint16`, `int16`, `uint32`, `int32`, `uint64`, `int64`— sí van al
nativo. `int64` importa más de lo que parece: es el dtype por defecto de
`np.array([[1, 2]])` en Linux.

## Dónde se engancha

Dos lugares simétricos en `morphology_common.py`, `apply_binary_operation` y
`apply_grayscale_operation`. El binario:

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

El grayscale es igual, más el filtro de dtype:

```python
if native_op is not None and img.dtype.kind in _NATIVE_GRAYSCALE_KINDS:
    backend = _backend.native()
    if backend is not None:
        resultado = backend.grayscale_op(img, kernel, int(iterations), native_op)
        return resultado.astype(source.dtype, copy=False)
```

Las cuatro operaciones elementales pasan `native_op="erode"` / `"dilate"`. Una
operación sin `native_op` cae al bucle de Python sin ninguna rama extra.

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

`test/test_backend_parity.py` — 463 tests que corren la misma entrada por los
dos backends y exigen igualdad exacta de valores y de dtype. Más 13 en
`test_cli_main.py` para las tres flags nuevas, incluida la rama de divergencia,
que se alcanza reemplazando el despacho.

El bloque grayscale agrega lo suyo: los ocho dtypes enteros con verificación de
que el dtype se conserva, y dos tests sobre el camino que **no** se toma —
que un flotante nunca llegue al nativo (se comprueba rompiéndolo: si lo tocara,
explotaría) y que `NaN` siga propagándose por el bucle de Python.

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

El motor grayscale se escribió ya sabiendo esto, y aun así dos de sus tests
sobrevivieron a la misma mutación aplicada solo a `sweep_gray`: uno usaba
`kernel_diamond(7)` suelto y el otro `kernel_cross(7)`, ambos simétricos. Se
parametrizaron sobre la lista completa, y la mutación pasó de 58 a 75 fallos.
La receta para repetirlo está en la sección siguiente.

La suite completa corre limpia con los dos: 556 tests en ambos modos.

### Cómo repetir la prueba de mutación

```bash
cp native/src/lib.rs /tmp/lib.rs.bak
# editar reflect(): reemplazar el cuerpo por `index.clamp(0, len - 1) as usize`
cd native && maturin develop --release && cd ..
pytest -q          # ~69 fallos, todos en test_backend_parity.py
cp /tmp/lib.rs.bak native/src/lib.rs
cd native && maturin develop --release && cd ..
```

Otras que vale la pena: sacar el `.rem_euclid` (rompe kernels más grandes que la
imagen), quitar el `if len == 1` (rompe ejes de longitud 1), invertir
`Op::Erode`/`Op::Dilate`, arrancar el acumulador de `sweep_gray` desde el centro
en vez del primer offset activo (221 fallos), o mover el padding fuera del bucle
de iteraciones.

Si una mutación **no** hace fallar nada, eso no significa que el Rust esté bien:
significa que ese comportamiento no está cubierto. Vale más una mutación que
sobrevive que diez que mueren.

### Simular la ausencia del nativo

```bash
mkdir -p /tmp/sin-nativo
printf 'raise ModuleNotFoundError("No module named %s", name="vispyx_native")\n' \
  "'vispyx_native'" > /tmp/sin-nativo/vispyx_native.py
PYTHONPATH=/tmp/sin-nativo pytest -q     # 433 pasan, 6 se saltan
```

Tiene que ser `ModuleNotFoundError` y no un `ImportError` genérico: desde pytest
8.2, `importorskip` solo trata el primero como dependencia ausente y deja
propagar el segundo, porque un `ImportError` desde el cuerpo de un módulo indica
un problema real. Con el genérico, la suite falla en la colección y parece un
bug del paquete.

## Siguientes pasos

En orden de rendimiento por esfuerzo:

1. **`vpx_skeletonize`** (Zhang-Suen). El peor caso absoluto y lo único pesado
   que queda: no escala con los píxeles sino con píxeles × iteraciones, y las
   iteraciones crecen con el grosor de los objetos. Medido: 0.63 s en 128×128,
   5.10 s en 256×256, **43.37 s en 512×512**. Sobre 1024×1024 son minutos.
   `vpx_thin` hereda el problema.
2. **CI con wheels.** El repo no tiene `.github/` todavía. Sin eso
   `vispyx-native` no se puede publicar y `pip install vispyx[fast]` sigue sin
   funcionar: el extra está declarado pero apunta a un paquete que no existe en
   PyPI. Hace falta una matriz de `maturin-action` y correr la suite con
   `VISPYX_BACKEND` en `python` y `rust`.
3. **Liberar el GIL** en `binary_op` y `grayscale_op`, para que el nativo no
   bloquee otros hilos durante la pasada.
4. **van Herk / Gil-Werman** para kernels grandes, donde el speedup actual baja
   a 8x. Haría el costo independiente del tamaño del kernel.
5. **Flotantes en el motor grayscale**, si aparece la necesidad. Requiere
   decidir y fijar por test la semántica de `NaN`.

`vpx_reconstruct` **ya no está en la lista**: medido, el bucle geodésico está
dominado por la dilatación, que es nativa. Con el backend puesto tarda 0.0766 s
sobre 256×256 contra 28.16 s en Python puro, un 368x que salió gratis. Portar el
bucle daría casi nada.
