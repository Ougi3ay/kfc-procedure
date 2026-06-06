import numpy as np
import pytest

from kfc_procedure.cobra.core.aggregators.weighted_mean import WeightedMeanAggregator
from kfc_procedure.cobra.core.aggregators.weighted_vote import WeightedVoteAggregator


# ======================================================
# WEIGHTED MEAN TESTS
# ======================================================

def test_weighted_mean_basic():
    agg = WeightedMeanAggregator()

    values = np.array([1.0, 2.0, 3.0])
    weights = np.array([1.0, 1.0, 1.0])

    result = agg.aggregate(values, weights)

    assert np.isclose(result, 2.0)


def test_weighted_mean_zero_weights_fallback():
    agg = WeightedMeanAggregator()

    values = np.array([1.0, 2.0, 3.0])
    weights = np.array([0.0, 0.0, 0.0])

    result = agg.aggregate(values, weights)

    assert np.isclose(result, np.mean(values))


def test_weighted_mean_nan_weights():
    agg = WeightedMeanAggregator()

    values = np.array([1.0, 2.0, 3.0])
    weights = np.array([np.nan, 1.0, 1.0])

    result = agg.aggregate(values, weights)

    assert np.isclose(result, (2 + 3) / 2)


def test_weighted_mean_batch_consistency():
    agg = WeightedMeanAggregator()

    values = np.array([1.0, 2.0, 3.0])
    weights = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])

    batch = agg.aggregate_matrix(values, weights)

    expected = np.array([1.0, 2.0, 3.0])

    np.testing.assert_allclose(batch.astype(float), expected)


# ======================================================
# WEIGHTED VOTE TESTS
# ======================================================

def test_weighted_vote_basic():
    agg = WeightedVoteAggregator()

    values = np.array([0, 1, 1])
    weights = np.array([0.2, 0.3, 0.5])

    result = agg.aggregate(values, weights)

    assert result == 1


def test_weighted_vote_no_weights_majority():
    agg = WeightedVoteAggregator()

    values = np.array([0, 0, 1, 1, 1])

    result = agg.aggregate(values)

    assert result == 1


def test_weighted_vote_nan_weights():
    agg = WeightedVoteAggregator()

    values = np.array([0, 1, 1])
    weights = np.array([np.nan, 0.3, 0.5])

    result = agg.aggregate(values, weights)

    assert result == 1


def test_weighted_vote_batch():
    agg = WeightedVoteAggregator()

    values = np.array([0, 1, 1])
    weights = np.array([
        [1.0, 0.0, 0.0],  # -> 0
        [0.0, 1.0, 1.0],  # -> 1
    ])

    result = agg.aggregate_matrix(values, weights)

    assert np.array_equal(result, np.array([0, 1]))


# ======================================================
# EDGE CASE TESTS
# ======================================================

def test_weighted_vote_empty_values():
    agg = WeightedVoteAggregator()

    with pytest.raises(ValueError):
        agg.aggregate(np.array([]), np.array([]))


def test_weighted_mean_empty_values():
    agg = WeightedMeanAggregator()

    with pytest.raises(ValueError):
        agg.aggregate(np.array([]), np.array([]))
