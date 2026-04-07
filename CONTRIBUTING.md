# Contributing

This repository currently serves as a research-code release accompanying a manuscript. External contributions are welcome only after coordination with the authors.

## Local setup

```bash
pip install -e .[dev,experiments]
```

## Before opening a pull request

1. Keep the experimental design unchanged unless the authors explicitly approve a methodological update.
2. Preserve the documented numerical setup.
3. Run:
   ```bash
   pytest -q
   python -m build
   ```
4. Update documentation if code behavior or interfaces change.

## Coding style

- Use English comments and docstrings.
- Prefer small, reviewable commits.
- Avoid changing notebook outputs unless necessary.
