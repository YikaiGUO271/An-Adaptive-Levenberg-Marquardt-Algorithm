# Reproducibility Notes

This file is intended to support manuscript submission, archival release, and experiment reproduction.

## 1. Software environment

Please fill in the exact versions used for the final experiments.

- **Operating system:** TO BE FILLED
- **Python version:** TO BE FILLED
- **JAX version:** TO BE FILLED
- **jaxlib version:** TO BE FILLED
- **NumPy version:** TO BE FILLED
- **SciPy version:** TO BE FILLED
- **Matplotlib version:** TO BE FILLED
- **tqdm version:** TO BE FILLED
- **Hardware:** TO BE FILLED
- **CPU/GPU details:** TO BE FILLED

## 2. Installation

```bash
pip install -e .[dev,experiments]
```

## 3. Validation

```bash
pytest -q
python -m build
```

## 4. Experiment entry point

The original experiment notebook is:

```text
Experiments.ipynb
```

If the notebook depends on a specific execution order, please document that order here.

## 5. Randomness and seeds

List the exact seeds used in the final manuscript experiments here.

- **Global NumPy seed:** 42 in package code where applicable
- **Additional experiment seeds:** TO BE FILLED

## 6. Expected outputs

Describe which notebook sections, figures, or tables correspond to the final manuscript.

- Figure/Table mapping: TO BE FILLED
- Runtime expectations: TO BE FILLED
- Any machine-dependent behavior: TO BE FILLED

## 7. Archival checklist

Before release, verify that:
- metadata files are completed,
- build artifacts can be produced,
- the notebook still executes with the documented environment,
- GitHub release assets match the tagged version.
