# FILE: 2026_Data_Science_and_Machine_Learning\.github\README.md

```markdown
# .github/

GitHub configuration for this repository — CI/CD workflows, issue templates,
and pull request guidelines.

## Contents

| Path | Purpose |
|------|---------|
| [`workflows/test.yml`](workflows/test.yml) | GitHub Actions CI pipeline: runs the full `pytest` suite on every push and pull request |
| [`ISSUE_TEMPLATE/bug_report.md`](ISSUE_TEMPLATE/bug_report.md) | Structured template for reporting bugs in `rice_ml` or the example notebooks |
| [`pull_request_template.md`](pull_request_template.md) | Checklist shown when opening a pull request |

## CI/CD

Every push to `main` and every pull request triggers the `test.yml` workflow,
which installs the package in editable mode and runs:

```bash
pytest tests/ -v
```

The workflow status badge can be added to the root README once the repository
is hosted on GitHub:

```markdown
![Tests](https://github.com/<your-username>/2026_Data_Science_and_Machine_Learning/actions/workflows/test.yml/badge.svg)
```
```

---
