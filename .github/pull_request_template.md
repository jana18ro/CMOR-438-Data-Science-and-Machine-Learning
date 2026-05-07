## Summary

Briefly explain what this pull request changes.

## Type of Change

Check all that apply:

- [ ] New algorithm or model
- [ ] Bug fix
- [ ] Refactor / code cleanup
- [ ] Tests added or updated
- [ ] Documentation or README update
- [ ] Example notebook added or updated
- [ ] Packaging / project configuration
- [ ] Other:

## Package Area Affected

Check all that apply:

- [ ] `src/rice_ml/supervised_learning`
- [ ] `src/rice_ml/unsupervised_learning`
- [ ] `src/rice_ml/processing`
- [ ] `src/rice_ml/measures`
- [ ] `examples/`
- [ ] `tests/`
- [ ] `.github/`
- [ ] Project metadata files

## What Was Added or Changed?

Describe the main changes in plain language.

If this PR adds an algorithm, include:
- what the algorithm does
- the expected estimator interface
- any key parameters
- any important limitations

## Testing

Describe how you tested the changes.

- [ ] I ran `pytest`
- [ ] I ran a specific test file:
- [ ] I tested with a notebook or small example
- [ ] I checked edge cases or invalid inputs
- [ ] Not applicable

Add details here:

```bash
pytest
```

## Documentation

- [ ] I added or updated docstrings
- [ ] I added or updated README content
- [ ] I added or updated an example notebook
- [ ] No documentation changes needed

## Project Requirements Check

- [ ] Core algorithm code is written from scratch
- [ ] NumPy is used appropriately for numerical work
- [ ] scikit-learn is not used inside the algorithm implementation itself
- [ ] The public API follows the project style, such as `fit`, `predict`, `fit_predict`, `transform`, or `score`
- [ ] Names, imports, and file paths match the package structure
- [ ] The change should not break editable install with `pip install -e .`

## Risk / Impact

Are there any possible breaking changes?

- [ ] No breaking changes expected
- [ ] Possible breaking change; explanation below

Explain any risks, limitations, or compatibility concerns:

## Related Issue

Closes #

## Reviewer Notes

Add anything specific reviewers should pay attention to.
