# Documentación de vispyx

Índice de la documentación interna. `vispyx` `0.2.0`.

## Para usar el paquete

| Documento | Qué contiene |
|---|---|
| [system_usage.md](./system_usage.md) | Guía integral: instalación, modelo mental, pipeline completo, rendimiento y trampas |
| [api_reference.md](./api_reference.md) | Los 28 símbolos públicos: firmas, contratos, dtypes y catálogo de errores |
| [cli_reference.md](./cli_reference.md) | El comando `vispyx`: 17 métodos, todas las flags, códigos de salida |

## Para entender la morfología

| Documento | Qué contiene |
|---|---|
| [binary_morphology_usage.md](./binary_morphology_usage.md) | Las 12 operaciones `vpx_*`, con el algoritmo real de cada una |
| [grayscale_morphology_usage.md](./grayscale_morphology_usage.md) | Las 7 operaciones `gray_*` y sus diferencias con el bloque binario |

## Para modificar el paquete

| Documento | Qué contiene |
|---|---|
| [architecture.md](./architecture.md) | Capas, decisiones de diseño, deuda técnica, cómo agregar una operación |
| [testing.md](./testing.md) | Qué fija la suite, qué no cubre, dónde poner el siguiente test |

## Por dónde empezar

- **Nunca usaste el paquete** → [system_usage.md](./system_usage.md)
- **Buscas una firma concreta** → [api_reference.md](./api_reference.md)
- **Quieres procesar imágenes sin escribir Python** → [cli_reference.md](./cli_reference.md)
- **Vas a tocar el código** → [architecture.md](./architecture.md), después [testing.md](./testing.md)

## Fuera de `docs/`

- [../README.md](../README.md) — presentación del proyecto
- [../CONTRIBUTING.md](../CONTRIBUTING.md) — alcance y checklist de PR
- [../CHANGELOG.md](../CHANGELOG.md) — historial de versiones

## Regla de mantenimiento

Estos documentos describen **lo que el código hace hoy**, incluidas sus rarezas.
Cuando una rareza se arregle, hay que borrarla de aquí: una trampa documentada
que ya no existe confunde más que la ausencia de documentación.
