# Changelog

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
