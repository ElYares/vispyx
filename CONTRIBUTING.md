# Contributing

## Scope

`vispyx` is centered on image morphology implemented from scratch. Contributions should reinforce that direction instead of adding dependencies that replace the core algorithms.

## Principles

- keep the public API stable
- prefer clear, testable implementations over clever shortcuts
- do not introduce external packages to perform the morphological operations themselves
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
- `vispyx/cli.py`: command-line interface
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
