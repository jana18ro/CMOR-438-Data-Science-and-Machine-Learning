# FILE: 2026_Data_Science_and_Machine_Learning\tests\__init__.py

"""
tests
=====
Unit test suite for the rice_ml package.

This package contains pytest-based unit tests for every module in rice_ml.
Tests are organized by algorithm/module and cover:

  - Correct mathematical outputs (e.g., regression coefficients, cluster labels)
  - Consistent fit/predict interface behaviour
  - Edge cases: empty arrays, single-sample inputs, invalid parameters
  - Preprocessing correctness: scaling, encoding, splitting with no leakage
  - Metric computation against known ground-truth values

Running the Tests
-----------------
From the repository root:

    pytest                          # run all tests
    pytest -v                       # verbose output
    pytest tests/test_perceptron.py # run a single test file
    pytest --cov=rice_ml            # with coverage report

Tests also run automatically on every push and pull request via
GitHub Actions (.github/workflows/test.yml).
"""
