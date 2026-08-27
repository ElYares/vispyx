# Changelog

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
