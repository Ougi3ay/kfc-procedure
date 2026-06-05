import numpy as np
import pytest

from cobra.core.splitters import (
    OverlapSplitter,
    RandomHoldoutSplitter,
    SplitterFactory,
)


# -------------------------
# Fixtures
# -------------------------
@pytest.fixture
def sample_data():
    X = np.arange(100).reshape(100, 1)
    y = np.arange(100)
    return X, y


# -------------------------
# Holdout Splitter Tests
# -------------------------
def test_holdout_split_shapes(sample_data):
    X, y = sample_data

    splitter = RandomHoldoutSplitter(calibration_size=0.3, random_state=42)
    result = splitter.split(X, y)

    n = len(X)

    assert len(result.train_idx) + len(result.eval_idx) == n
    assert set(result.train_idx).isdisjoint(set(result.eval_idx))


def test_holdout_reproducibility(sample_data):
    X, y = sample_data

    s1 = RandomHoldoutSplitter(calibration_size=0.3, random_state=42)
    s2 = RandomHoldoutSplitter(calibration_size=0.3, random_state=42)

    r1 = s1.split(X, y)
    r2 = s2.split(X, y)

    assert np.array_equal(r1.train_idx, r2.train_idx)
    assert np.array_equal(r1.eval_idx, r2.eval_idx)


# -------------------------
# Overlap Splitter Tests
# -------------------------
def test_overlap_split_basic(sample_data):
    X, y = sample_data

    splitter = OverlapSplitter(
        split_ratio=0.5,
        overlap=0.2,
        shuffle=True,
        random_state=42,
    )

    result = splitter.split(X, y)

    assert len(result.train_idx) > 0
    assert len(result.eval_idx) > 0
    assert len(result.train_idx) <= len(X)
    assert len(result.eval_idx) <= len(X)


def test_overlap_exists(sample_data):
    X, y = sample_data

    splitter = OverlapSplitter(
        split_ratio=0.5,
        overlap=0.4,
        shuffle=False,
    )

    result = splitter.split(X, y)

    train = set(result.train_idx)
    eval_ = set(result.eval_idx)

    assert len(train.intersection(eval_)) > 0


def test_overlap_zero(sample_data):
    X, y = sample_data

    splitter = OverlapSplitter(
        split_ratio=0.5,
        overlap=0.0,
        shuffle=False,
    )

    result = splitter.split(X, y)

    assert set(result.train_idx).isdisjoint(set(result.eval_idx))


def test_overlap_invalid_params():
    with pytest.raises(ValueError):
        OverlapSplitter(split_ratio=1.2)

    with pytest.raises(ValueError):
        OverlapSplitter(overlap=-0.1)
    
    with pytest.raises(ValueError):
        OverlapSplitter(split_ratio=0.3, overlap=0.5)


# -------------------------
# Factory Tests
# -------------------------
def test_factory_create_holdout():
    splitter = SplitterFactory.create(
        "holdout",
        calibration_size=0.3,
        random_state=42,
    )
    assert isinstance(splitter, RandomHoldoutSplitter)


def test_factory_create_overlap():
    splitter = SplitterFactory.create("split_overlap")
    assert isinstance(splitter, OverlapSplitter)


def test_factory_available():
    names = SplitterFactory.available()
    assert "holdout" in names
    assert "split_overlap" in names


def test_factory_case_insensitive():
    s1 = SplitterFactory.create("HoLdOuT", calibration_size=0.3)
    s2 = SplitterFactory.create("holdout", calibration_size=0.3)

    assert type(s1) == type(s2)


# -------------------------
# Debug sanity check
# -------------------------
def test_split_sanity(sample_data):
    X, y = sample_data

    splitter = OverlapSplitter(split_ratio=0.6, overlap=0.2, shuffle=True)

    result = splitter.split(X, y)

    assert np.min(result.train_idx) >= 0
    assert np.min(result.eval_idx) >= 0
    assert np.max(result.train_idx) < len(X)
    assert np.max(result.eval_idx) < len(X)
