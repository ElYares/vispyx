# Changelog

## No publicado

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
