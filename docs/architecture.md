# Arquitectura interna

Cómo está armado `vispyx` por dentro y por qué. Documento para quien va a
modificar el paquete, no para quien lo usa.

## Capas

```text
                    vispyx/__init__.py
                    superficie pública (28 símbolos)
                              |
        +---------------------+---------------------+
        |                     |                     |
   vispyx/cli.py      vispyx/morphology.py    preprocessing.py
   (17 métodos)       fachada de compat.      segmentation.py
        |                     |                utils.py
        |          +----------+----------+          |
        |          |                     |          |
        |   morphology_binary.py   morphology_grayscale.py
        |          |                     |
        |          +----------+----------+
        |                     |
        +----------> morphology_common.py
                validaciones + motor de ventana
```

Regla de dependencias: `morphology_common` no importa a nadie del paquete. Los
dos módulos de morfología importan solo de `common`. `morphology.py` no importa
lógica, solo reagrupa. El CLI importa de la fachada.

## Los tres archivos que importan

### `morphology_common.py` — el corazón

Contiene todo lo que se repite, y por eso concentra el riesgo:

| Función | Rol |
|---|---|
| `validate_binary_image` | 2D → `uint8` 0/1 (umbral `> 0`) |
| `validate_grayscale_image` | 2D + dtype numérico, **sin transformar** |
| `validate_iterations` | `int` nativo y `> 0` |
| `validate_kernel` | 2D, no vacío, dims impares, ≥1 celda activa; normaliza a 0/1 |
| `validate_hitmiss_kernels` | dos kernels válidos, misma forma, sin solape |
| `pad_image` | `np.pad(mode="reflect")` con grosor `kh//2 × kw//2` |
| `apply_binary_operation` | motor de ventana deslizante binario |
| `apply_grayscale_operation` | motor de ventana deslizante grayscale |

Los dos motores son idénticos salvo en tres puntos: el binario binariza la
entrada, pasa `active_count` al reducer y multiplica el resultado por 255; el
grayscale no hace nada de eso y recasta al dtype original.

**Cambiar cualquier cosa aquí cambia las 19 operaciones a la vez.** Es el lugar
donde una prueba de más vale por diez.

### `morphology_binary.py` — 12 operaciones

Erosión y dilatación son dos `reducer` de una línea sobre el motor común. Las
compuestas (`open`, `close`, `gradient`, `tophat`, `blackhat`, `boundary`) son
composición explícita de esas dos, más una resta en `int16` con `clip(min=0)`.

Tres funciones se salen del motor y hacen su propio recorrido:

- `vpx_reconstruct`: bucle de dilatación geodésica con `np.minimum` contra la
  máscara
- `vpx_skeletonize`: Zhang-Suen completo, con su propio padding **de ceros** y
  eliminación diferida por subpasada
- `vpx_hitmiss`: dos erosiones (imagen y complemento) y un `logical_and`

### `morphology.py` — fachada

Trece líneas de `import`. Existe para que `from vispyx.morphology import
vpx_erode` siga funcionando después del split de `0.2.0`. No envuelve, no
adapta, no valida. Si se agrega una operación nueva hay que tocarla, y también
`__init__.py`.

## Decisiones de diseño y sus consecuencias

### Bucles Python en vez de vectorización

Los motores recorren píxel por píxel con dos `for` anidados. Vectorizar con
`stride_tricks` o `scipy.ndimage` sería órdenes de magnitud más rápido — y es
exactamente lo que `CONTRIBUTING.md` prohíbe: *"do not introduce external
packages to perform the morphological operations themselves"*.

El costo es real: `O(iterations · H · W · |kernel activo|)` en Python puro. Una
imagen de 1024×1024 con kernel 5×5 son 26 millones de reducciones. Es un
paquete para entender morfología y para trabajar sobre regiones de interés, no
para procesar lotes.

Si alguna vez se optimiza, el lugar es `apply_binary_operation` /
`apply_grayscale_operation` **sin cambiar su firma**: los reducers y las 19
funciones públicas seguirían intactos.

### Dos convenciones de valores, a propósito

`vpx_*` trabaja en 0/255 y `gray_*` en el rango nativo. Internamente el binario
usa 0/1 y multiplica por 255 solo al salir. La frontera entre ambos mundos es
`segment_otsu`, que produce justo la convención que el bloque binario espera.

El riesgo: una imagen de grises pasada a `vpx_*` no falla — se binariza con
`> 0`, así que **cualquier píxel que no sea negro puro se vuelve 255**. Silencio
absoluto y resultado inútil. Es la trampa más fácil de pisar del paquete.

### Padding inconsistente

Todo usa reflejo salvo Zhang-Suen, que usa ceros. Es correcto en ambos casos
(Zhang-Suen asume fondo fuera de la imagen), pero significa que los resultados
en el borde no son comparables entre `vpx_skeletonize` y el resto.

### Validaciones que devuelven `ValueError` con mensajes fijos

Los tests casan contra el texto literal con `pytest.raises(..., match=...)`.
**Los mensajes son parte del contrato público**: cambiar una palabra rompe la
suite. La tabla completa está en
[api_reference.md](./api_reference.md#catálogo-de-errores).

## Deuda técnica identificada

Cosas reales del código, no hipótesis:

1. **`read_grayscale` devuelve `None` en silencio** cuando falla la lectura,
   mientras que el CLI tiene su propio `_read_grayscale` que sí lanza
   `FileNotFoundError`. Dos lectores con contratos distintos para lo mismo.
2. **`cli.py` no comprueba el retorno de `cv2.imwrite`**: puede imprimir
   "Imagen guardada" sin haber guardado nada.
3. **`morph_scipy.py` sigue fuera del paquete instalable**, pero ya no es código
   huérfano: `test/test_reference_scipy.py` lo usa como oráculo de referencia
   para validar las operaciones binarias. `scipy` está declarado en el extra
   `dev`. Cubre erode, dilate, open, close y gradient; el resto de las
   operaciones sigue sin oráculo — ver [testing.md](./testing.md).
4. **`examples/demo.ipynb` está vacío** (0 bytes) desde el commit que lo
   introdujo. El README no lo menciona, pero el directorio `examples/` sugiere
   contenido que no existe.
5. **El CLI no expone los generadores de kernels.** Construye siempre
   `np.ones((n, n))`; `kernel_cross`, `kernel_diamond` y `kernel_disk` solo son
   alcanzables desde Python.
6. **Cuatro operaciones binarias no están en el CLI**: `vpx_tophat`,
   `vpx_blackhat`, `vpx_boundary` y `vpx_hitmiss`.

## Cómo agregar una operación

1. Escribe el `reducer` y la función pública en `morphology_binary.py` o
   `morphology_grayscale.py`, apoyándote en el motor de `morphology_common.py`.
2. Agrégala al `import` y al `__all__` de `morphology.py`.
3. Agrégala al `import` y al `__all__` de `__init__.py`.
4. Súmala al set `expected_symbols` de `test/test_public_api.py`.
5. Escribe el test de valor exacto en `test/test_morphology.py` sobre una matriz
   chica hecha a mano, y el test de la validación que rechaza.
6. Si la expones en el CLI: agrégala a la lista `methods`, escribe su `run_*`, y
   documéntala en [cli_reference.md](./cli_reference.md).
7. Actualiza `CHANGELOG.md`.

## Ver también

- [api_reference.md](./api_reference.md)
- [testing.md](./testing.md)
