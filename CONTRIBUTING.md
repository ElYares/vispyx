# Contributing

## Scope

`vispyx` is centered on image morphology implemented from scratch. Contributions should reinforce that direction instead of adding dependencies that replace the core algorithms.

## Principles

- keep the public API stable
- prefer clear, testable implementations over clever shortcuts
- do not introduce external packages to perform the morphological operations themselves
- the optional Rust backend in `native/` is not an exception to that rule: it is the
  same algorithm written from scratch in another language, it must match the Python
  engine bit for bit, and the package has to keep working without it
- update tests and docs together with code

## Development Setup

```bash
pip install -e .[dev]
pytest -q
```

## Areas of the codebase

- `vispyx/morphology_binary.py`: binary morphology
- `vispyx/morphology_grayscale.py`: grayscale morphology
- `vispyx/morphology_common.py`: shared validation and helpers
- `vispyx/_backend.py`: engine selection, Python loops or the optional Rust backend
- `vispyx/cli.py`: command-line interface
- `native/`: optional Rust backend, published separately as `vispyx-native`
- `test/`: unit and integration tests
- `docs/`: user and maintainer documentation

## Pull Request Checklist

- code follows the existing package structure
- new public behavior is documented
- tests cover the new behavior
- CLI changes are reflected in `docs/cli_reference.md`
- API changes are reflected in `docs/api_reference.md`

## Release Notes

For public-facing changes, update:

- `CHANGELOG.md`
- `README.md` if user-facing behavior changed
