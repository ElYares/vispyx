# Changelog

## No publicado

### Motor grayscale

**El motor grayscale tambien corre en Rust: 17 de 19 operaciones aceleradas.**

`gray_erode` y `gray_dilate` se suman al backend nativo, y las cinco compuestas
--`gray_open`, `gray_close`, `gray_gradient`, `gray_tophat`, `gray_blackhat`--
heredan la aceleracion sin una linea nueva, igual que paso con las binarias.
Con cuatro funciones nativas en total quedan cubiertas 17 de las 19 operaciones
del paquete.

Medido: **242x en gray_erode 3x3 sobre 256x256**, 261x en gray_open, 272x en
uint16. Resultados identicos bit a bit, y el dtype de entrada se conserva.

- ocho dtypes enteros van al nativo: uint8, int8, uint16, int16, uint32, int32,
  uint64, int64. `int64` no es exotico: es el dtype por defecto de
  `np.array([[1, 2]])` en Linux
- **los flotantes se quedan en Python a proposito.** `Ord` en Rust es un orden
  total y los flotantes no lo tienen; reproducir bit a bit la propagacion de
  `NaN` de `np.min` no vale el riesgo en un spike. El despacho mira
  `dtype.kind` y manda float32/float64 al bucle de siempre, sin avisar: mismo
  resultado, solo mas lento. Dos tests cubren ese camino, incluido uno que
  rompe el nativo para probar que un float nunca lo toca
- **el speedup cae con el tamano del kernel, y no es un defecto del port.** En
  Python el costo es por pixel --domina el overhead de numpy por ventana-- y en
  Rust es por celda activa. Medido: un `gray_erode` sobre 128x128 tarda lo mismo
  con kernel 3x3 (0.0564 s) que con 15x15 (0.0549 s), pese a tener 25 veces mas
  celdas. Por eso el mismo port da 242x en 3x3 y 8x en 15x15
- `active_offsets` y `parse_op` salen a funciones compartidas: los dos motores
  del crate usan la misma geometria y el mismo despacho de operacion
- la suite pasa de 669 a 901 tests con el nativo instalado; sin el, 433 pasan y
  6 se saltan
- **dos tests nuevos sobrevivieron a la mutacion y se reforzaron.** Aplicando
  clamp en vez de reflejo solo en `sweep_gray`, `kernel_diamond(7)` y
  `kernel_cross(7)` no lo detectaban: son simetricos y contienen el centro, la
  misma trampa que ya habia aparecido en el bloque binario. Parametrizados sobre
  la lista completa de kernels, la mutacion pasa de 58 a 75 fallos
- `docs/native_backend.md` documenta la receta de mutacion y como simular la
  ausencia del nativo. Ese segundo detalle tiene su propia trampa: el modulo
  falso tiene que lanzar `ModuleNotFoundError` y no `ImportError`, porque desde
  pytest 8.2 `importorskip` solo trata el primero como dependencia ausente
- `vpx_reconstruct` sale de la lista de pendientes. Medido, su bucle geodesico
  ya esta dominado por la dilatacion nativa: 0.0766 s contra 28.16 s en Python
  puro sobre 256x256, un 368x que salio gratis por composicion
- **fuera de alcance**: `vpx_skeletonize` y `vpx_thin` siguen en Python, y son
  ahora lo unico pesado que queda. No escalan con los pixeles sino con pixeles
  por iteraciones: 0.63 s en 128x128, 5.10 s en 256x256, 43.37 s en 512x512

### Motor binario

**Spike: backend opcional en Rust para erosion y dilatacion binaria.**

`vpx_erode` y `vpx_dilate` pueden ejecutarse en Rust, con resultados identicos
bit a bit a los del bucle de Python. Las ocho operaciones compuestas
(`open`, `close`, `gradient`, `tophat`, `blackhat`, `boundary`, `hitmiss`,
`reconstruct`) heredan la aceleracion sin cambios, porque ya estaban escritas
como composicion explicita de esas dos. Medido: **376x en `vpx_erode` 3x3 sobre
256x256**, 478x en `vpx_open` con dos iteraciones.

Ningun simbolo publico nuevo, ninguna firma cambiada, ningun mensaje de error
distinto. Sin el nativo instalado, el paquete se comporta exactamente igual que
antes.

- nuevo `vispyx/_backend.py`: modulo hoja, sin dependencias del paquete, que
  resuelve el motor una sola vez al importar. `VISPYX_BACKEND` acepta `auto`
  (default), `python` y `rust`; cualquier otro valor lanza `ValueError`
- `apply_binary_operation` toma un parametro nuevo `native_op`. Cuando el
  backend nativo esta activo y la operacion lo declara, delega el recorrido.
  La rama va **despues** de las tres validaciones: el nativo nunca ve una
  entrada sin normalizar y nunca lanza los mensajes de error, que siguen siendo
  contrato publico de Python
- nueva distribucion `vispyx-native` en `native/` (PyO3 + maturin), separada a
  proposito: `pip install vispyx` no debe necesitar un compilador. Declarada
  como extra `vispyx[fast]`, pendiente de publicar en PyPI
- nuevo `test/test_backend_parity.py`, 231 tests, mas 13 en `test_cli_main.py`
  para las flags nuevas. Corre la misma entrada por los
  dos backends y exige igualdad exacta de valores y dtype. A diferencia de
  `test_reference_scipy.py`, que rodea cada imagen con un marco de fondo para
  evitar el borde, este lo toca a proposito: foreground pegado al margen, un
  pixel en la esquina, kernels mas grandes que la imagen, ejes de longitud 1 y
  vistas no contiguas. Se salta solo si el nativo no esta instalado, e incluye
  un test que verifica que el backend bajo prueba **es** el nativo
- la suite completa pasa en los dos modos: 669 tests con nativo, 438 sin el
- **la paridad esta verificada por mutacion**. Cambiando a proposito el reflejo
  por repeticion de borde en el Rust, ninguno de los 438 tests preexistentes
  falla: un kernel solido no distingue las dos cosas, porque en la columna 0 el
  reflejo muestrea `{img[1], img[0], img[1]}` y la repeticion
  `{img[0], img[0], img[1]}`, que como conjunto son iguales. Solo discrimina un
  soporte que excluya el centro y sea asimetrico, y por eso la lista de kernels
  incluye `[[1,0,1]]`, `[[1,0,0]]`, `[[1],[0],[0]]` y un 3x3 con solo la esquina
  activa. Con esos, la misma mutacion cae en 69 tests en vez de 15
- **el CLI aprende a hablar del backend**, que antes no tenia forma de saberse
  desde la linea de comandos:
  - `vispyx --version` imprime `vispyx 0.4.0 (backend: rust, vispyx-native
    0.1.0)`. Es lo unico que responde "que motor tengo" sin correr una operacion
  - `--backend {auto,python,rust}` elige el motor para esa invocacion, con
    prioridad sobre `VISPYX_BACKEND`. Pedir `rust` sin el paquete instalado
    **falla con salida 2** en vez de caer a Python en silencio: recibir un motor
    distinto del pedido invalida cualquier medicion
  - `--time` reporta cuanto tardo la operacion y con que motor
  - `--compare` corre los dos, mide, y verifica que el resultado coincida.
    Sirve mas que cronometrar dos invocaciones desde la shell porque deja el
    arranque del interprete fuera de la medicion: es cerca de un segundo, y en
    imagenes chicas tapa por completo la diferencia. Si los dos motores
    divergen sale con codigo 1, porque eso es un bug del paquete y no un error
    de uso
- la cadena de despacho de `main()` sale a `_dispatch()`. Era la unica forma de
  cronometrarla y de correrla dos veces sobre backends distintos
- la resolucion del backend pasa a ser perezosa. Hacerla al importar convertia
  un `VISPYX_BACKEND` mal escrito en un fallo de `import vispyx`
- nuevo `docs/native_backend.md`; `native/bench.py` reproduce las mediciones
- **fuera de alcance por ahora**: las 7 `gray_*`, `vpx_skeletonize` y `vpx_thin`
  siguen 100% en Python

## 0.4.0

**El CLI queda completo y el paquete deja de filtrar excepciones ajenas.**

Las cuatro operaciones binarias que faltaban ya se alcanzan desde la linea de
comandos, y ninguna funcion deja escapar un error de OpenCV, scikit-image o
numpy: toda entrada invalida sale como `ValueError` con un mensaje del paquete.

Tres cambios de contrato, todos detallados abajo: `apply_clahe` y `segment_otsu`
pasan a validar su entrada, y `--kernel-size` par falla con otro mensaje. Ningun
camino feliz cambia.

Cobertura de 137 a 425 tests, y de lineas al **100%**.

- `cli.py` deja de tener su propia copia de `show_image`. Era identica salvo por
  `figsize=(8, 6)` y `tight_layout()`, y **la cobertura la mostraba con cero
  ejecuciones** mientras la de `utils.py` si estaba cubierta. Ahora hay una
  sola, y `figsize` es un parametro opcional: omitirlo conserva el
  comportamiento historico de dibujar sobre la figura activa
- de paso sale `import matplotlib.pyplot as plt` de `cli.py`, que solo usaba esa
  copia. `matplotlib.use("TkAgg")` se queda: es lo que `--show` necesita
- **cobertura de lineas al 100%**. `--show` sigue sin poder ejercitarse (necesita
  display), pero si se fija que despache a la unica implementacion, y que
  `cli.show_image is utils.show_image` para que la copia no vuelva
- cobertura de 422 a 425 tests
- **fix**: una imagen vacia se comportaba de cuatro maneras distintas. Las 17
  `vpx_*`/`gray_*` con padding por reflejo lanzaban `ValueError` pero con el
  mensaje interno de numpy (`can't extend empty axis 0...`); `vpx_skeletonize` y
  `vpx_thin` **no fallaban** y devolvian `(0, 0)`, porque Zhang-Suen usa padding
  de ceros; `segment_otsu` lanzaba `IndexError` desde skimage; y **`apply_clahe`
  devolvia `None` con `(0, 0)` y se colgaba indefinidamente con `(0, 5)` o
  `(5, 0)`**. Ahora las cuatro lanzan `ValueError: image must not be empty`,
  desde `validate_binary_image` y `validate_grayscale_image`. `validate_kernel`
  ya rechazaba kernels vacios; las imagenes no tenian el chequeo equivalente
- `test/test_edge_cases.py`: 35 tests de entradas degeneradas y caminos que
  nadie recorria — imagenes de un pixel y de una sola fila, kernels mas grandes
  que la imagen, kernels no cuadrados, `max_iterations` explicito e
  `iterations > 2`. Cobertura de 387 a 422 tests
- **cambio de contrato**: `segment_otsu` valida su entrada. Antes una imagen de
  tres canales **no fallaba** — skimage emitia un `UserWarning` que nadie ve y
  devolvia un resultado sin sentido; una lista anidada salia como
  `AttributeError` y un dtype no numerico como `UFuncTypeError`. Ahora los tres
  son `ValueError`, reusando `validate_grayscale_image`, y las **listas
  anidadas funcionan**, igual que en las `gray_*` desde `0.2.1`
- `test/test_segmentation.py`: 8 tests. `segmentation.py` era el ultimo modulo
  del paquete sin un solo test funcional, y marcaba **100% de cobertura de
  lineas** porque el test de punta a punta del CLI corre `otsu` sin afirmar nada
  sobre el resultado
- dos tests para las ramas defensivas que `argparse` hace inalcanzables:
  `Patron no reconocido` en `run_vpx_hitmiss` y `Método no reconocido` en
  `main()`. Con eso, las unicas lineas sin cubrir del paquete son las de
  `--show`, excluido a proposito porque necesita display
- cobertura de 377 a 387 tests
- `vpx_hitmiss` en el CLI por `--pattern`, con patrones de nombre en vez de dos
  kernels sueltos: `corner` (las cuatro esquinas), `corner-nw`, `corner-ne`,
  `corner-se`, `corner-sw` e `isolated`. El CLI pasa de 20 a 21 metodos y
  **cierra la ultima deuda accionable de `docs/architecture.md`**
- `--pattern` es obligatoria para `vpx_hitmiss`: omitirla sale con codigo 2,
  igual que `--mask` para `vpx_reconstruct`
- `cli.py` expone `METHODS`, `PATTERNS` y `PATTERN_NAMES` como constantes de
  modulo. `METHODS` estaba dentro de `main()` y no se podia comprobar desde los
  tests: la lista `METODOS` de `test_cli_main.py` es a mano, y agregar un metodo
  al CLI lo dejaba **sin cobertura en silencio**. Ahora un test compara las dos
- cobertura de 364 a 377 tests
- **cambio de contrato**: `apply_clahe` valida su entrada. Antes una imagen
  `float64`, una de tres canales o una lista anidada salian como `cv2.error`
  desde `clahe.cpp`; ahora salen como `ValueError`, igual que toda validacion
  del paquete. Reusa `validate_grayscale_image`, asi que comparte los mensajes
  `image must be a 2D array` e `image must contain numeric values`, y agrega
  `image must be uint8 or uint16` — OpenCV solo implementa CLAHE para `CV_8UC1`
  y `CV_16UC1`. El camino feliz no cambia
- `preprocessing.py` pasa a importar de `morphology_common`. Es la unica arista
  fuera de la morfologia y es deliberada: los mensajes de error son contrato
  publico y no deben duplicarse. `common` sigue sin importar a nadie
- cobertura de 356 a 364 tests
- `test/test_preprocessing.py` reescrito: 7 tests en vez de 2. Los viejos
  miraban shape y tipo sobre ruido **sin semilla fija**, y esta medido lo poco
  que amarraban — con ellos, `apply_clahe` devolviendo la entrada sin tocar
  pasa, e ignorar `clip_limit` o `tile_grid_size` tambien. Ahora se afirma la
  expansion del contraste local, el dtype, el determinismo, que los dos
  parametros lleguen a OpenCV, y el alias historico `title_grid_size`, que nada
  fijaba. Cobertura de 351 a 356 tests
- refactor interno de `cli.py`: las ocho `run_vpx_*` que eran la misma funcion
  de cuatro lineas con otro nombre se colapsan en `_run_binary_method`, el
  gemelo de `_run_grayscale_method` que ya existia. 64 lineas menos, sin cambio
  de comportamiento. Sobreviven `run_vpx_reconstruct`, `run_vpx_skeletonize` y
  `run_vpx_thin`, que no toman `(image, kernel, iterations)`. Ninguna de las
  ocho estaba en `__init__.py` ni se usaba fuera de `cli.py`
- `vpx_tophat`, `vpx_blackhat` y `vpx_boundary` en el CLI. Existian en la API
  desde `0.2.0` y solo se alcanzaban desde Python, aunque sus primos grises
  (`gray_tophat`, `gray_blackhat`) si estaban. El CLI pasa de 17 a 20 metodos y
  las tres aceptan `--kernel-size`, `--kernel-shape` e `--iterations`
- **`vpx_hitmiss` queda afuera a proposito**: toma dos estructurantes y no toma
  `iterations`, asi que no entra en el molde de `--kernel-shape`. Necesita una
  decision de diseno propia, no mas plomeria
- cobertura de 345 a 351 tests
- `test/test_morphology.py`: 11 tests sobre las ramas de validacion que no
  ejercia nadie. Las seis de `validate_kernel` — incluidas las dos que no
  lanzan, el default `kernel=None` y la normalizacion con `> 0` — mas
  `marker and mask must have the same shape`,
  `kernel_hit and kernel_miss must have the same shape` y la rama no-par de
  `_validate_size`. Cobertura de 334 a 345 tests
- **fix**: `cli.py` anunciaba "Imagen guardada en: ..." y salia con codigo 0
  aunque no hubiera guardado nada. `cv2.imwrite` falla de dos maneras y ninguna
  se comprobaba: devuelve `False` cuando no logra abrir el archivo (permisos,
  la ruta es un directorio) y lanza `cv2.error` cuando no tiene codec para la
  extension — esto ultimo llegaba al usuario como traceback de OpenCV. Las dos
  salen ahora por `parser.error`, con codigo **2** y mensaje en `stderr`, y
  `--show` deja de ejecutarse sobre un guardado fallido
- **fix**: `--output` bajo un directorio que no se puede crear salia como
  traceback de `os` en vez de como error del CLI. `os.makedirs` corre antes de
  escribir y falla con `PermissionError` si el padre no deja crear, o
  `NotADirectoryError` si el padre no es un directorio. Ahora sale por
  `parser.error` con codigo **2** y el `strerror` real del sistema
- `test/test_cli_main.py`: 4 tests del guardado fallido, uno por cada camino.
  Cobertura de 330 a 334 tests
- `--kernel-shape` en el CLI: `square` (default), `cross`, `diamond` y `disk`.
  Los cuatro generadores de `vispyx.kernels` dejan de ser alcanzables solo
  desde Python. Omitir la flag construye el mismo cuadrado de unos de siempre
- para `disk`, el radio se deriva como `--kernel-size // 2`, porque
  `kernel_disk` toma radio y no lado. `--kernel-size 5` da un disco de 5x5
- **cambio de mensaje de error**: `--kernel-size` par ahora falla con
  `ValueError: size must be odd`, lanzado al construir el kernel y antes de
  leer la imagen. Antes salia `ValueError: kernel dimensions must be odd`,
  desde `validate_kernel` y ya dentro de la operacion. Vale para las cuatro
  formas, el disco incluido: sin esa validacion un `4` daria radio `2` y el
  mismo disco que un `5`, en silencio
- `test/test_cli_main.py`: 44 tests sobre `main()`, que no tenia ninguno. Cubre
  los 17 metodos de punta a punta, las flags, la creacion de directorios, los
  codigos de salida y los errores de lectura
- `test/test_invariants.py`: 193 tests que verifican las leyes clasicas de la
  morfologia como **propiedad general** sobre imagenes aleatorias con semilla
  fija, no como casos elegidos a mano. Idempotencia, dualidad por complemento,
  orden (`erode ⊆ open ⊆ x ⊆ close ⊆ dilate`) y monotonia, en los dos dominios
  de valores y una vez por cada forma de kernel
- esos tests corren con kernel de **7**, no de 3 ni de 5: recien en 7 las cuatro
  formas difieren entre si. Un guard falla si se baja la constante, para que la
  degradacion de cuatro ramas a dos no pase en silencio
- afirman sobre la imagen completa, sin marco de fondo. El reflejo preserva las
  cuatro familias tambien en el borde; queda medido que pasar el padding a ceros
  rompe 96 de estos tests y que pasarlo a `edge` no rompe ninguno
- cobertura de 93 a 330 tests. La suite pasa de ~1.4 s a ~2.8 s

## 0.3.0

Un solo contrato para leer imagenes. **Cambio incompatible.**

- `read_grayscale` deja de devolver `None` en silencio cuando falla la lectura.
  Ahora lanza `FileNotFoundError` si no hay ningun archivo en la ruta, y
  `ValueError` si el archivo existe pero no se puede decodificar. Antes los dos
  casos eran indistinguibles
- el CLI deja de tener su propio `_read_grayscale` y usa la funcion publica: el
  error es identico se entre por Python o por linea de comandos
- `examples/demo.ipynb` deja de estar vacio: notebook ejecutable con el pipeline
  completo paso a paso, la comparacion medida de que preprocesamiento resuelve
  el problema, las cuatro formas de kernel y la trampa de pasar grises a una
  `vpx_*`
- cobertura de 84 a 93 tests

Si tu codigo comprobaba `if img is None` despues de `read_grayscale`, esa rama
ya no se alcanza: la excepcion salta antes. Se puede borrar.

## 0.2.1

Pulido de contrato, sin comportamiento nuevo.

- `iterations` y `max_iterations` aceptan cualquier tipo entero, incluidos los
  de NumPy: `np.int64(2)` era rechazado con `ValueError` aunque fuera un conteo
  valido, y castigaba a quien derivaba el valor de un array
- `iterations=True` pasa a ser rechazado. `bool` es subclase de `int`, asi que
  antes se colaba silenciosamente como una iteracion
- las operaciones `gray_*` aceptan listas anidadas: antes lanzaban
  `AttributeError` en vez de funcionar, porque el cast final iba contra el
  argumento crudo y no contra el array validado
- se eliminan imports muertos: `numpy` en `preprocessing.py` y `utils.py`,
  `cv2` en `segmentation.py`
- documentacion interna completa en `docs/`, mas `CLAUDE.md`
- oraculo diferencial contra `scipy.ndimage` en `test/test_reference_scipy.py`
- cobertura de 49 a 84 tests

## 0.2.0

- refactor de morfologia hacia modulos binario, grayscale y helpers compartidos
- API publica consolidada desde `vispyx`
- generadores formales de kernels
- bloque binario ampliado con:
  - `vpx_tophat`
  - `vpx_blackhat`
  - `vpx_boundary`
  - `vpx_hitmiss`
  - `vpx_reconstruct`
  - `vpx_skeletonize`
  - `vpx_thin`
- bloque grayscale ampliado con:
  - `gray_erode`
  - `gray_dilate`
  - `gray_open`
  - `gray_close`
  - `gray_gradient`
  - `gray_tophat`
  - `gray_blackhat`
- CLI ampliada para morfologia binaria, grayscale y reconstruccion con `--mask`
- migracion de `setup.py` a `pyproject.toml`
- documentacion de uso binario y grayscale
- cobertura de tests ampliada
