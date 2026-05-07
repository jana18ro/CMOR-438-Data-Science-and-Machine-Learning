"""
test_pca.py

Comprehensive tests for the pca class.

Covers:
- Constructor & n_components variants (int, float, None)
- fit() attribute shapes and values
- Orthogonality of components
- transform() projection shapes
- fit_transform() consistency
- inverse_transform() roundtrip
- reconstruction_error() and score()
- get_covariance() shape
- Error handling (unfitted model, bad input)
- Integration on Iris dataset
"""

import numpy as np
import pytest
import sys, os
sys.path.insert(0, r"/Jana CMOR/2026_Data_Science_and_Machine_Learning/src/rice_ml/unsupervised_learning")
from rice_ml.unsupervised_learning.pca import pca


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def iris_scaled():
    from sklearn.datasets import load_iris
    from sklearn.preprocessing import StandardScaler
    X, y = load_iris(return_X_y=True)
    return StandardScaler().fit_transform(X), y

@pytest.fixture
def simple_3d():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 3))
    return X

@pytest.fixture
def fitted_iris(iris_scaled):
    X, _ = iris_scaled
    model = pca(n_components=2)
    model.fit(X)
    return model, X


# ─────────────────────────────────────────────────────────────────────────────
# 1. Constructor
# ─────────────────────────────────────────────────────────────────────────────

class TestInit:
    def test_default_n_components_none(self):
        assert pca().n_components is None

    def test_int_n_components(self):
        assert pca(n_components=2).n_components == 2

    def test_float_n_components(self):
        assert pca(n_components=0.95).n_components == 0.95

    def test_components_none_before_fit(self):
        assert pca().components_ is None

    def test_mean_none_before_fit(self):
        assert pca().mean_ is None


# ─────────────────────────────────────────────────────────────────────────────
# 2. fit() — attribute shapes and values
# ─────────────────────────────────────────────────────────────────────────────

class TestFit:
    def test_fit_returns_self(self, iris_scaled):
        X, _ = iris_scaled
        m = pca(n_components=2)
        assert m.fit(X) is m

    def test_components_shape_int(self, iris_scaled):
        X, _ = iris_scaled
        m = pca(n_components=2)
        m.fit(X)
        assert m.components_.shape == (2, 4)

    def test_components_shape_none(self, simple_3d):
        m = pca(n_components=None)
        m.fit(simple_3d)
        assert m.components_.shape[1] == 3

    def test_mean_shape(self, iris_scaled):
        X, _ = iris_scaled
        m = pca(n_components=2)
        m.fit(X)
        assert m.mean_.shape == (4,)

    def test_mean_close_to_zero_after_scale(self, iris_scaled):
        """StandardScaler centres X, so PCA mean should be near zero."""
        X, _ = iris_scaled
        m = pca(n_components=2)
        m.fit(X)
        np.testing.assert_allclose(m.mean_, 0, atol=1e-10)

    def test_explained_variance_ratio_sums_leq_1(self, iris_scaled):
        X, _ = iris_scaled
        m = pca(n_components=None)
        m.fit(X)
        assert m.explained_variance_ratio_.sum() <= 1.0 + 1e-10

    def test_explained_variance_ratio_descending(self, iris_scaled):
        X, _ = iris_scaled
        m = pca(n_components=None)
        m.fit(X)
        evr = m.explained_variance_ratio_
        assert all(evr[i] >= evr[i + 1] for i in range(len(evr) - 1))

    def test_singular_values_stored(self, iris_scaled):
        X, _ = iris_scaled
        m = pca(n_components=2)
        m.fit(X)
        assert m.singular_values_ is not None
        assert len(m.singular_values_) == 2

    def test_n_components_stored(self, iris_scaled):
        X, _ = iris_scaled
        m = pca(n_components=2)
        m.fit(X)
        assert m.n_components_ == 2

    def test_float_n_components_selects_enough(self, iris_scaled):
        """n_components=0.95 should select enough components for ≥95% variance."""
        X, _ = iris_scaled
        m = pca(n_components=0.95)
        m.fit(X)
        assert m.explained_variance_ratio_.sum() >= 0.95

    def test_too_few_samples_raises(self):
        X = np.array([[1.0, 2.0]])   # only 1 sample
        with pytest.raises(ValueError):
            pca(n_components=1).fit(X)

    def test_invalid_n_components_raises(self, iris_scaled):
        X, _ = iris_scaled
        with pytest.raises(ValueError):
            pca(n_components="bad").fit(X)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Orthogonality of components
# ─────────────────────────────────────────────────────────────────────────────

class TestOrthogonality:
    def test_components_orthonormal(self, iris_scaled):
        X, _ = iris_scaled
        m = pca(n_components=None)
        m.fit(X)
        gram = m.components_ @ m.components_.T
        np.testing.assert_allclose(gram, np.eye(len(gram)), atol=1e-10)


# ─────────────────────────────────────────────────────────────────────────────
# 4. transform()
# ─────────────────────────────────────────────────────────────────────────────

class TestTransform:
    def test_transform_shape(self, fitted_iris):
        m, X = fitted_iris
        T = m.transform(X)
        assert T.shape == (len(X), 2)

    def test_transform_before_fit_raises(self, iris_scaled):
        X, _ = iris_scaled
        with pytest.raises(RuntimeError):
            pca(n_components=2).transform(X)

    def test_transform_wrong_features_raises(self, fitted_iris):
        m, X = fitted_iris
        with pytest.raises(ValueError):
            m.transform(np.ones((10, 10)))

    def test_transformed_mean_near_zero(self, fitted_iris):
        """Projected data should have near-zero mean (data was centred)."""
        m, X = fitted_iris
        T = m.transform(X)
        np.testing.assert_allclose(T.mean(axis=0), 0, atol=1e-10)

    def test_transform_returns_ndarray(self, fitted_iris):
        m, X = fitted_iris
        assert isinstance(m.transform(X), np.ndarray)


# ─────────────────────────────────────────────────────────────────────────────
# 5. fit_transform()
# ─────────────────────────────────────────────────────────────────────────────

class TestFitTransform:
    def test_fit_transform_equals_fit_then_transform(self, iris_scaled):
        X, _ = iris_scaled
        m1 = pca(n_components=2)
        T1 = m1.fit_transform(X)

        m2 = pca(n_components=2)
        m2.fit(X)
        T2 = m2.transform(X)

        np.testing.assert_allclose(T1, T2, atol=1e-10)

    def test_fit_transform_shape(self, iris_scaled):
        X, _ = iris_scaled
        T = pca(n_components=3).fit_transform(X)
        assert T.shape == (len(X), 3)


# ─────────────────────────────────────────────────────────────────────────────
# 6. inverse_transform()
# ─────────────────────────────────────────────────────────────────────────────

class TestInverseTransform:
    def test_roundtrip_perfect_with_all_components(self, simple_3d):
        m = pca(n_components=None)
        T = m.fit_transform(simple_3d)
        X_back = m.inverse_transform(T)
        np.testing.assert_allclose(X_back, simple_3d, atol=1e-10)

    def test_roundtrip_shape(self, fitted_iris):
        m, X = fitted_iris
        T = m.transform(X)
        X_back = m.inverse_transform(T)
        assert X_back.shape == X.shape

    def test_inverse_before_fit_raises(self):
        with pytest.raises(RuntimeError):
            pca(n_components=2).inverse_transform(np.ones((5, 2)))

    def test_wrong_columns_raises(self, fitted_iris):
        m, X = fitted_iris
        with pytest.raises(ValueError):
            m.inverse_transform(np.ones((5, 10)))


# ─────────────────────────────────────────────────────────────────────────────
# 7. reconstruction_error() and score()
# ─────────────────────────────────────────────────────────────────────────────

class TestReconstructionError:
    def test_error_zero_all_components(self, simple_3d):
        m = pca(n_components=None)
        m.fit(simple_3d)
        assert m.reconstruction_error(simple_3d) < 1e-10

    def test_error_increases_fewer_components(self, iris_scaled):
        X, _ = iris_scaled
        errors = []
        for n in [4, 3, 2, 1]:
            m = pca(n_components=n)
            m.fit(X)
            errors.append(m.reconstruction_error(X))
        assert errors == sorted(errors)  # ascending as components decrease

    def test_score_is_negative_error(self, fitted_iris):
        m, X = fitted_iris
        assert m.score(X) == pytest.approx(-m.reconstruction_error(X), rel=1e-10)

    def test_score_negative(self, fitted_iris):
        m, X = fitted_iris
        assert m.score(X) <= 0


# ─────────────────────────────────────────────────────────────────────────────
# 8. get_covariance()
# ─────────────────────────────────────────────────────────────────────────────

class TestCovariance:
    def test_covariance_shape(self, fitted_iris):
        m, X = fitted_iris
        cov = m.get_covariance()
        assert cov.shape == (4, 4)

    def test_covariance_symmetric(self, fitted_iris):
        m, X = fitted_iris
        cov = m.get_covariance()
        np.testing.assert_allclose(cov, cov.T, atol=1e-12)

    def test_covariance_before_fit_raises(self):
        with pytest.raises(RuntimeError):
            pca(n_components=2).get_covariance()


# ─────────────────────────────────────────────────────────────────────────────
# 9. Integration — Iris separability
# ─────────────────────────────────────────────────────────────────────────────

class TestIrisIntegration:
    def test_2d_projection_separates_setosa(self, iris_scaled):
        X, y = iris_scaled
        m = pca(n_components=2)
        T = m.fit_transform(X)
        # Setosa (class 0) has distinctly different PC1 values
        setosa_pc1 = T[y == 0, 0]
        other_pc1  = T[y != 0, 0]
        # The means should be clearly separated
        assert abs(setosa_pc1.mean() - other_pc1.mean()) > 1.0

    def test_variance_retained_2_components(self, iris_scaled):
        X, _ = iris_scaled
        m = pca(n_components=2)
        m.fit(X)
        assert m.explained_variance_ratio_.sum() > 0.9
