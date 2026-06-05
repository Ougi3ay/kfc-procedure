import numpy as np
import pytest

from cobra.core.cv import CVFactory
from cobra.core.cv import KFoldCV


# =========================================================
# Fixtures
# =========================================================
@pytest.fixture
def sample_data():
    X = np.arange(50).reshape(50, 1)
    y = np.arange(50)
    return X, y


# =========================================================
# Factory tests
# =========================================================
def test_cv_factory_create():
    cv = CVFactory.create("kfold", n_splits=5)
    assert isinstance(cv, KFoldCV)


def test_cv_factory_available():
    names = CVFactory.available()
    assert "kfold" in names


# =========================================================
# Basic structure tests
# =========================================================
def test_kfold_number_of_splits(sample_data):
    X, y = sample_data

    cv = KFoldCV(n_splits=5)

    folds = list(cv.split(X, y))

    assert len(folds) == 5


# =========================================================
# Disjoint validation sets
# =========================================================
def test_kfold_disjoint_validation(sample_data):
    X, y = sample_data

    cv = KFoldCV(n_splits=5, shuffle=True, random_state=42)

    folds = list(cv.split(X, y))

    for i in range(len(folds)):
        for j in range(i + 1, len(folds)):
            vi = set(folds[i].eval_idx)
            vj = set(folds[j].eval_idx)

            assert vi.isdisjoint(vj)


# =========================================================
# Full coverage test
# =========================================================
def test_kfold_full_coverage(sample_data):
    X, y = sample_data

    cv = KFoldCV(n_splits=5, shuffle=True, random_state=42)

    folds = list(cv.split(X, y))

    all_val = np.concatenate([f.eval_idx for f in folds])

    assert set(all_val) == set(range(len(X)))


# =========================================================
# No overlap train vs validation (within fold)
# =========================================================
def test_kfold_no_train_val_leakage(sample_data):
    X, y = sample_data

    cv = KFoldCV(n_splits=5, shuffle=True, random_state=42)

    folds = list(cv.split(X, y))

    for f in folds:
        train = set(f.train_idx)
        val = set(f.eval_idx)

        assert train.isdisjoint(val)


# =========================================================
# Deterministic behavior
# =========================================================
def test_kfold_reproducibility(sample_data):
    X, y = sample_data

    cv1 = KFoldCV(n_splits=5, shuffle=True, random_state=123)
    cv2 = KFoldCV(n_splits=5, shuffle=True, random_state=123)

    f1 = list(cv1.split(X, y))
    f2 = list(cv2.split(X, y))

    for a, b in zip(f1, f2):
        assert np.array_equal(a.train_idx, b.train_idx)
        assert np.array_equal(a.eval_idx, b.eval_idx)


# =========================================================
# Fold size sanity check
# =========================================================
def test_kfold_fold_sizes(sample_data):
    X, y = sample_data

    cv = KFoldCV(n_splits=5, shuffle=True, random_state=42)

    folds = list(cv.split(X, y))

    total = len(X)

    # each sample appears exactly once in validation
    val_counts = np.zeros(total, dtype=int)

    for f in folds:
        for idx in f.eval_idx:
            val_counts[idx] += 1

    assert np.all(val_counts == 1)