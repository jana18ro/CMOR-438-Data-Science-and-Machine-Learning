---
name: Feature request
about: Suggest a new algorithm, utility, notebook, test, or documentation improvement
title: "[Feature]: "
labels: enhancement
assignees: ""
---

## Feature Summary

Describe the feature you would like added to `rice_ml`.

## Where Would This Fit?

Check or describe the area of the package this request affects:

- [ ] Supervised learning
- [ ] Unsupervised learning
- [ ] Processing / preprocessing
- [ ] Metrics / validation
- [ ] Example notebooks
- [ ] Tests
- [ ] Documentation
- [ ] Packaging / CI
- [ ] Other:

## Motivation

Why would this feature be useful for the project?

Explain the learning goal, algorithmic purpose, or practical problem it solves.

## Proposed Behavior

Describe what the feature should do.

If this is an algorithm or estimator, include the expected interface if possible:

```python
model = ExampleModel(...)
model.fit(X, y)
predictions = model.predict(X)
score = model.score(X, y)
```

## Implementation Notes

Share any ideas about how this could be implemented.

Consider:
- important formulas or algorithm steps
- expected inputs and outputs
- edge cases to handle
- whether it should avoid scikit-learn inside core algorithm code
- whether it needs a matching notebook or test file

## Testing Ideas

How should this feature be tested?

Examples:
- simple toy dataset
- shape checks
- known numerical output
- error handling
- comparison against a trusted reference for sanity checking

## Additional Context

Add screenshots, links, examples, or notes that may help explain the request.
