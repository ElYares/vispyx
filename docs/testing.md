# Testing

Estado real de la suite, qué contrato fija y dónde están los huecos.

## Correr los tests

```bash
pip install -e .[dev]
pytest -q
```

Estado verificado: **356 tests, 356 pasan, ~4.0 s** (Python 3.13.13, numpy
2.5.2, OpenCV 5.0, scikit-image 0.26, matplotlib 3.11, pytest 9.1,
scipy 1.18).

Hay un `conftest.py` vacío en la raíz: existe solo para poner el directorio del
repo en `sys.path`, y así permitir que `test_reference_scipy.py` importe
`morph_scipy`, que vive fuera del paquete instalable.

Si `pytest` falla con `ModuleNotFoundError`, casi siempre es el entorno y no el
código: `test_cli.py` y `test_preprocessing.py` importan `cv2` a nivel de
módulo, y `vispyx/__init__.py` arrastra `skimage` vía `segmentation.py`. Sin
esas dependencias **ningún** archivo de test llega siquiera a colectarse.

## Reparto

| Archivo | Tests | Qué fija |
|---|---|---|
| `test_invariants.py` | 193 | las leyes de la morfología como propiedad general |
| `test_cli_main.py` | 54 | el parser: flags, guardado y códigos de salida |
| `test_morphology.py` | 47 | el núcleo algorítmico, binario y grayscale |
| `test_reference_scipy.py` | 29 | diferencial contra la implementación de referencia en SciPy |
| `test_utils.py` | 9 | el lector de imágenes y `show_image` |
| `test_public_api.py` | 8 | cableado de la superficie pública y versión |
| `test_kernels.py` | 6 | forma exacta de los cuatro generadores |
| `test_cli.py` | 3 | las tres `run_*` que no encajan en el molde, con I/O real |
| `test_preprocessing.py` | 7 | qué hace `apply_clahe` y que sus parámetros lleguen |

## Qué se verifica de verdad

La estrategia principal es **valor exacto sobre matrices chicas hechas a mano**,
comparadas con `assert_array_equal` contra un resultado escrito a mano. A eso se
suman dos capas que no dependen de casos elegidos: comparación contra una
implementación de referencia, desde `test_reference_scipy.py`, y property-based
testing sobre imágenes aleatorias con semilla fija, desde `test_invariants.py`.

El grueso del conteo vive ahí: 193 de los 330 tests son invariantes, porque cada
propiedad se multiplica por cuatro formas de kernel y cuatro semillas. Son
también el grueso del tiempo — la suite pasó de ~1.4 s a ~2.8 s — y el precio se
paga en imágenes de 20×20, porque los bucles no están vectorizados y el costo
crece con el área: un `vpx_open` con kernel 7×7 tarda 4 ms en 20×20 y 37 ms en
64×64.

Lo que sí queda amarrado:

- **Composición operacional**: que `open = erode → dilate`, `close = dilate →
  erode`, `gradient = dilate − erode`, `tophat = image − opening`,
  `blackhat = closing − image`, `boundary = image − erosion` producen los
  números correctos en casos concretos, en ambos dominios.
- **Efecto del kernel**: dilatar con cruz da un rombo y con cuadrado da un
  cuadrado; confirma que `active_mask` restringe la vecindad de verdad.
- **Iteraciones**: `iterations=2` equivale a aplicar la operación dos veces.
- **Reconstrucción**: un marcador de un píxel recupera la región conexa completa
  bajo la máscara, y un marcador fuera de la máscara es rechazado.
- **Convergencia vs. parcialidad**: `vpx_skeletonize` reduce un bloque 3×3 a un
  píxel; `vpx_thin` se verifica con desigualdades (`0 < n <= n_original`),
  no con igualdad.
- **Los mensajes de error**, con `pytest.raises(ValueError, match="...")` sobre
  el texto literal. Esto convierte los mensajes en contrato público.
- **Los tipos que `iterations` acepta y rechaza**: enteros de NumPy sí,
  `bool` no, `float` y `str` no.

## Convenciones que la suite impone

- Las `vpx_*` viven en **0/255**, nunca 0/1: el helper `_to_uint8` de
  `test_morphology.py` multiplica por 255 antes de cada llamada.
- Las `gray_*` operan sobre los valores nativos (10, 20, …, 90) y no normalizan.
- Toda entrada inválida produce `ValueError`, nunca `TypeError` ni `assert`.
- `vispyx.morphology` debe seguir funcionando como import path: los tests del
  núcleo importan desde la fachada, no desde `morphology_binary`.
- `vispyx.__version__ == "0.3.0"` está clavado en un test: **subir la versión
  rompe la suite si no se actualiza también ahí**.

## Lo que cubre `test_cli_main.py`

`main()` son 222 líneas de parseo y despacho, y hasta la `0.3.0` no tenían un
solo test. Ahora sí: los **20 métodos** corren de punta a punta escribiendo a
disco, más las flags (`--output`, `--kernel`/`--kernel-size`, `--kernel-shape`,
`--iterations`, `--clip`/`--grid`), la creación de directorios, el mensaje
`"Imagen procesada. No se guardó."`, los códigos de salida `2` de `argparse`, y
que los errores de lectura sean los mismos que ve quien usa la API.

**Por qué la parametrización sobre los 20 métodos vale más de lo que parece**:
si alguien agrega un método a la lista `methods` y olvida su rama de despacho,
ese test falla. Es la razón por la que la rama
`else: raise ValueError("Método no reconocido")` **se conserva** pese a ser
inalcanzable desde `argparse`: no es código muerto, es la red para ese error de
programación, y la parametrización la ejercita.

Fuera de alcance a propósito: `--show`, que fuerza el backend `TkAgg` y necesita
display.

**El tamaño del kernel decide qué puede probar un test de forma.** Las cuatro
formas coinciden entre sí para radios chicos: en `3` la cruz, el diamante y el
disco son la misma matriz; en `5` lo son el diamante y el disco; recién en `7`
las cuatro difieren. Un test de `--kernel-shape` escrito con `3` no distingue
nada, y uno escrito con `5` deja pasar que las ramas `diamond` y `disk` estén
cambiadas. Verificado mutando: con la suite entera en `5`, esa permutación pasa
sin una sola falla.

## Huecos de cobertura

Reales, verificados leyendo la suite completa.

### Módulos sin ningún test

- **`segmentation.py`**: `segment_otsu` solo aparece nombrada dentro del set
  `expected_symbols`. Nunca se invoca. Ni umbral, ni binarizado, ni dtype.
`utils.py` dejó de estar en esta lista: `test_utils.py` cubre las dos funciones,
incluida la distinción entre archivo ausente e ilegible, y que el CLI use ese
mismo lector.

### `preprocessing.py`

Los dos tests originales miraban shape y tipo sobre ruido **sin semilla fija**.
Está medido lo poco que eso amarraba: con ellos, `apply_clahe` devolviendo la
entrada **sin tocar** pasa, e ignorar `clip_limit` o `tile_grid_size` también.

Ahora se afirma lo que CLAHE existe para hacer — **expandir el contraste
local** — sobre una imagen construida, no sorteada: una mitad con rango completo
y otra casi plana. La mitad plana pasa de un rango de 15 niveles a más de 150.
Sobre un gradiente uniforme el efecto sería chico y el test no distinguiría
gran cosa, que es por qué la imagen tiene dos mitades.

También quedan fijados los dos parámetros y el alias histórico
`title_grid_size` — un typo que `preprocessing.py:14` conserva por
compatibilidad y que nada amarraba: borrarlo no rompía ningún test.

**Lo que sigue sin cubrirse:** `apply_clahe` no valida su entrada. Un array
`float64`, uno de tres canales o una lista anidada salen como `cv2.error` de
`clahe.cpp`, no como el `ValueError` que el resto del paquete garantiza. Es un
hueco de contrato, no de tests: arreglarlo cambia comportamiento.

### Ramas de validación

Todas las de `validate_kernel` están ejercidas, incluidas las dos que no lanzan:
el default `kernel=None` (un 3×3 de unos) y la normalización con `> 0`. También
`marker and mask must have the same shape`, `kernel_hit and kernel_miss must
have the same shape` y la rama no-par de `_validate_size`
(`size must be a positive integer`).

**Dos de esos tests necesitan más de una afirmación para morder**, y la razón es
la misma en los dos casos: la mutación que deberían atrapar hace degenerar el
caso en otro que sigue coincidiendo.

- el default se compara contra el 3×3 explícito **y** contra el 5×5. Solo la
  primera mitad dejaría pasar un default más grande
- la normalización compara la cruz pesada contra la cruz limpia **y** contra el
  cuadrado. Bajo `>= 0` la cruz pesada se vuelve un cuadrado, y las dos cruces
  degeneran en lo mismo: la primera afirmación seguiría pasando

La imagen de ese segundo test tiene una esquina mordida a propósito. **Sobre un
bloque sólido, cruz y cuadrado erosionan idéntico** — la misma trampa que obliga
a usar kernels de 7 en `test_invariants.py`.

### Casos límite sin probar

Imágenes vacías o de un píxel; kernels más grandes que la imagen; kernels no
cuadrados (`3×5`); `max_iterations` explícito en `vpx_reconstruct` y
`vpx_skeletonize` (solo se prueba el camino hasta convergencia);
`iterations > 2`.

### Propiedades matemáticas

`test_invariants.py` verifica las garantías clásicas como **propiedad general**
sobre imágenes aleatorias con semilla fija, en los dos dominios de valores:

- idempotencia: `open(open(x)) == open(x)`, `close(close(x)) == close(x)`
- dualidad por complemento: `erode(x, k) == ¬dilate(¬x, k)`, y lo mismo un nivel
  más arriba con `open` contra `close`
- orden: `erode(x) ⊆ open(x) ⊆ x ⊆ close(x) ⊆ dilate(x)`
- monotonía: `x ⊆ y` implica `erode(x) ⊆ erode(y)`, e igual para las otras tres

Cada uno corre una vez por forma de kernel. Tres decisiones de diseño valen la
pena:

- **El tamaño de kernel es 7, nunca 3 ni 5.** Las cuatro formas se solapan en
  radios chicos: en 3, cruz, diamante y disco son la misma matriz, y en 5 lo
  siguen siendo diamante y disco. Recién en 7 las cuatro difieren (pesan 49, 13,
  25 y 29 píxeles). `test_the_four_kernel_shapes_are_distinct_at_this_size`
  guarda esa premisa, para que bajar la constante falle en vez de degradar en
  silencio cuatro ramas a dos.
- **Las imágenes no van enmarcadas en fondo**, a diferencia del diferencial.
  Estas propiedades se comparan contra `vispyx` mismo, y el reflejo las preserva
  también en el borde. Afirmar sobre el array completo es más simple y más
  estricto.
- **Alcance medido frente al padding.** Cambiar el reflejo por ceros rompe 96 de
  estos tests: el relleno con ceros no es una extensión de la imagen y las leyes
  dejan de valer en la orilla. Cambiarlo por `edge` no rompe **ninguno**: la
  replicación preserva todas estas propiedades igual que el reflejo. Es decir,
  restringen la *clase* de padding, no la elección dentro de ella. Esa elección
  es la de la Decisión 001 y la sigue cubriendo `test_reference_scipy`.

**Lo que no pueden atrapar, por construcción**: las propiedades valen para
*cualquier* elemento estructurante, así que permutar las ramas `diamond` y
`disk` las pasa enteras. La corrección de cada forma es trabajo de
`test_kernels.py`.

### Lo que el diferencial contra SciPy ya cubre

`test_reference_scipy.py` compara `vpx_erode`, `vpx_dilate`, `vpx_open`,
`vpx_close` y `vpx_gradient` contra `morph_scipy.MorphologicalProcessor`, que
implementa las mismas operaciones sobre `scipy.ndimage` con idéntica semántica
de kernel e iteraciones. Cubre kernels 3×3 y 5×5, de una a tres iteraciones, y
densidades de 0.2 a 0.8.

**Por qué vale**: una mutación que debilita la erosión — cambiar
`np.sum(region) == active_count` por `>= active_count - 1` — deja pasar los 49
tests originales **sin una sola falla** y rompe 11 de los nuevos. Era exactamente
el hueco: los tests de valor exacto verifican casos elegidos a mano, y una
implementación sutilmente incorrecta puede acertar en todos ellos.

Dos cosas que el diferencial no puede hacer directamente, y cómo están
resueltas:

- **El borde diverge a propósito.** `vispyx` rellena por reflejo y
  `scipy.ndimage` trata el exterior como fondo. Las comparaciones corren sobre
  imágenes rodeadas de un marco de fondo de ancho
  `(kernel_size // 2) * iterations * 2`, suficiente para que la operación nunca
  alcance el borde. La divergencia en sí está cubierta aparte, en
  `test_border_handling_differs_on_purpose`.
- **El `gradient` de `morph_scipy.py` re-binariza** los resultados de
  `dilate`/`erode` antes de restar, mientras que `vpx_gradient` resta en `int16`
  sobre 0/255. La comparación se hace sobre máscaras normalizadas a 0/1.

`scipy` está declarado en el extra `dev`, y el archivo empieza con
`pytest.importorskip("scipy")`: sin scipy, esos 29 tests se saltan en vez de
romper la suite.

Sigue sin haber comparación contra `cv2.morphologyEx` ni contra
`skimage.morphology`. Las operaciones sin oráculo son las que `morph_scipy.py`
no implementa: `tophat`, `blackhat`, `boundary`, `hitmiss`, `reconstruct`,
`skeletonize`, `thin` y todo el bloque `gray_*`.

## Prioridad sugerida

Si hay que elegir dónde poner el siguiente test, en este orden:

1. La validación de entrada de `apply_clahe`: hoy filtra `cv2.error` en vez de
   `ValueError`, contra la convención del resto del paquete
2. `vpx_hitmiss` en el CLI — la única de las cuatro binarias que quedó fuera, y
   necesita decidir cómo se expresan dos estructurantes en una línea de comandos

Hecho: el diferencial contra `morph_scipy.py`, la cobertura de `utils.py`, la de
`main()`, los invariantes de `test_invariants.py`, el guardado fallido del CLI,
las ramas de validación y `apply_clahe`.

## Ver también

- [architecture.md](./architecture.md)
- [../CONTRIBUTING.md](../CONTRIBUTING.md)
