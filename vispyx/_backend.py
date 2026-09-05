"""Selección del motor que ejecuta las operaciones morfológicas.

`vispyx` implementa la morfología desde cero en Python y esos bucles siguen
siendo la implementación de referencia: legible antes que rápida, y el oráculo
contra el que se valida todo lo demás. Este módulo permite que, cuando el
paquete opcional ``vispyx-native`` está instalado, el mismo algoritmo corra en
Rust con resultados idénticos bit a bit.

El módulo es una hoja: no importa nada de ``vispyx``, así que
``morphology_common`` puede depender de él sin abrir un ciclo.

La variable de entorno ``VISPYX_BACKEND`` manda sobre la detección:

``auto``
    nativo si está instalado, Python si no. Es el valor por defecto.
``python``
    fuerza los bucles de Python aunque el nativo esté disponible.
``rust``
    exige el nativo y falla al importar si no está.
"""

import os
from contextlib import contextmanager

VALID_MODES = ("auto", "python", "rust")

_ENV_VARIABLE = "VISPYX_BACKEND"


def _load_native():
    """Import the optional native extension, or None when it is absent."""
    try:
        import vispyx_native
    except ImportError:
        return None
    return vispyx_native


def _resolve(mode):
    """Return the backend module for ``mode``, or None to stay on Python."""
    if mode not in VALID_MODES:
        raise ValueError(
            "VISPYX_BACKEND must be one of: " + ", ".join(VALID_MODES)
        )
    if mode == "python":
        return None

    native = _load_native()
    if native is None and mode == "rust":
        raise ImportError(
            "VISPYX_BACKEND=rust requires the optional vispyx-native package; "
            "install it with `cd native && maturin develop --release`"
        )
    return native


# La resolución es perezosa a propósito. Hacerla al importar convertiría un
# `VISPYX_BACKEND` mal escrito en un fallo de `import vispyx`, mucho antes y
# mucho más lejos del lugar donde el usuario puede hacer algo al respecto.
_UNRESOLVED = object()

_MODE = os.environ.get(_ENV_VARIABLE, "auto").strip().lower() or "auto"
_ACTIVE = _UNRESOLVED


def available():
    """Whether the optional native extension can be imported at all.

    Independent of which backend is active: the extension can be installed and
    deliberately switched off with ``VISPYX_BACKEND=python``.
    """
    return _load_native() is not None


def native():
    """Return the active native backend, or None when running on Python."""
    global _ACTIVE

    if _ACTIVE is _UNRESOLVED:
        _ACTIVE = _resolve(_MODE)
    return _ACTIVE


def name():
    """Return the name of the active backend: ``"rust"`` or ``"python"``."""
    return "python" if native() is None else "rust"


def describe():
    """Human-readable backend line for ``vispyx --version``.

    Never raises: a broken ``VISPYX_BACKEND`` has to be reportable, and
    ``--version`` is exactly where someone looks to find out why.
    """
    try:
        backend = native()
    except (ValueError, ImportError) as error:
        return "no disponible: {}".format(error)
    if backend is None:
        return "python"
    return "rust, vispyx-native {}".format(backend.__version__)


@contextmanager
def override(mode):
    """Temporarily force a backend.

    Lo usan los tests de paridad y el CLI, para ``--backend`` y ``--compare``.
    No es parte de la API pública: una librería elige el motor una sola vez, por
    entorno, y no lo cambia a mitad de camino. Una herramienta de línea de
    comandos sí necesita hacerlo, y por eso esto existe.
    """
    global _ACTIVE

    previous = _ACTIVE
    _ACTIVE = _resolve(mode)
    try:
        yield
    finally:
        _ACTIVE = previous
