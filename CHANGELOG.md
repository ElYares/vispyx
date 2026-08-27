# Changelog

## No publicado

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
- `test/test_cli_main.py`: 30 tests sobre `main()`, que no tenia ninguno. Cubre
  los 17 metodos de punta a punta, las flags, la creacion de directorios, los
  codigos de salida y los errores de lectura
- cobertura de 93 a 137 tests

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
