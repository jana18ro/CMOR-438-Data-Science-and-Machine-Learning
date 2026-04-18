# FILE: 2026_Data_Science_and_Machine_Learning\.github\workflows\README.md

```markdown
# .github/workflows/

GitHub Actions workflow definitions for automated CI.

## `test.yml`

Runs the full `pytest` unit test suite on every push to `main` and on every
incoming pull request.

### Trigger Events

```yaml
on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]
```

### Steps

1. **Checkout** the repository (`actions/checkout@v3`)
2. **Set up Python 3.11** (`actions/setup-python@v4`)
3. **Install dependencies** — `pip install -r requirements.txt`
4. **Install the package** in editable mode — `pip install -e .`
5. **Run tests** — `pytest tests/ -v`

### Adding More Workflows

To add linting (e.g., `flake8`) or coverage reporting (e.g., `codecov`),
create additional `.yml` files in this directory following the same trigger pattern.
A coverage workflow example:

```yaml
- name: Run tests with coverage
  run: pytest tests/ --cov=rice_ml --cov-report=xml

- name: Upload coverage to Codecov
  uses: codecov/codecov-action@v3
```
```

---