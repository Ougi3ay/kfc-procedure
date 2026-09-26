import numpy as np
import pytest

from kfc_procedure.core.clustering.bregman import BregmanKMeans
from kfc_procedure.core.clustering.divergences.euclidean import SquaredEuclidean


def test_bregman_kmeans_fits_two_clusters():
    X = np.array([
        [0.0, 0.0], [0.1, 0.0], [0.0, 0.1],
        [5.0, 5.0], [5.1, 5.0], [5.0, 5.1],
    ])
    model = BregmanKMeans(
        n_clusters=2,
        divergence=SquaredEuclidean(),
        n_init=2,
        random_state=42,
    ).fit(X)

    assert model.labels_.shape == (6,)
    assert model.cluster_centers_.shape == (2, 2)
    assert np.isfinite(model.cluster_centers_).all()


def test_bregman_kmeans_reproducible():
    X = np.array([[0.0], [0.1], [10.0], [10.1]])
    kwargs = dict(n_clusters=2, divergence=SquaredEuclidean(), n_init=2, random_state=123)
    a = BregmanKMeans(**kwargs).fit(X)
    b = BregmanKMeans(**kwargs).fit(X)
    np.testing.assert_allclose(a.cluster_centers_, b.cluster_centers_)


def test_bregman_kmeans_rejects_wrong_divergence_type():
    with pytest.raises(TypeError):
        BregmanKMeans(n_clusters=2, divergence="euclidean")
