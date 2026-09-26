import numpy as np
import pytest

from kfc_procedure.cobra.core.distances.euclidean import EuclideanDistance
from kfc_procedure.cobra.core.distances.manhattan import ManhattanDistance
from kfc_procedure.cobra.core.distances.cosine import CosineDistance
from kfc_procedure.cobra.core.distances.hamming import HammingDistance
from kfc_procedure.cobra.core.distances.minkowski import MinkowskiDistance
from kfc_procedure.cobra.core.kernels.naive import NaiveKernel
from kfc_procedure.cobra.core.kernels.radial import RadialKernel
from kfc_procedure.cobra.core.kernels.triangular import TriangularKernel
from kfc_procedure.cobra.core.kernels.epanechnikov import EpanechnikovKernel
from kfc_procedure.cobra.core.kernels.biweight import BiweightKernel
from kfc_procedure.cobra.core.kernels.triweight import TriweightKernel
from kfc_procedure.cobra.core.kernels.cauchy import CauchyKernel
from kfc_procedure.cobra.core.kernels.exponential import ExponentialKernel
from kfc_procedure.cobra.core.kernels.reverse_cosh import ReverseCoshKernel
from kfc_procedure.cobra.core.kernels.cobra import COBRAKernel
from kfc_procedure.cobra.core.losses.mse import MSELoss
from kfc_procedure.cobra.core.losses.mae import MAELoss
from kfc_procedure.cobra.core.losses.huber import HuberLoss
from kfc_procedure.cobra.core.losses.hinge import HingeLoss
from kfc_procedure.cobra.core.losses.log_loss import LogLoss
from kfc_procedure.cobra.core.losses.quantile import QuantileLoss
from kfc_procedure.cobra.core.normalizers.minmax import MinMaxNormalizer
from kfc_procedure.cobra.core.normalizers.standard import StandardNormalizer
from kfc_procedure.cobra.core.aggregators.weighted_mean import WeightedMeanAggregator
from kfc_procedure.cobra.core.aggregators.weighted_vote import WeightedVoteAggregator

X=np.array([[0.,0.],[1.,2.],[2.,1.]])

@pytest.mark.parametrize('obj',[EuclideanDistance(),ManhattanDistance(),CosineDistance(),HammingDistance(),MinkowskiDistance(p=3)])
def test_distances_matrix(obj):
    D=obj.matrix(X,X)
    assert D.shape==(3,3)
    assert np.all(np.isfinite(D))
    if not isinstance(obj, CosineDistance):
        assert np.allclose(np.diag(D),0,atol=1e-8)

@pytest.mark.parametrize('kernel',[NaiveKernel(),RadialKernel(),TriangularKernel(),EpanechnikovKernel(),BiweightKernel(),TriweightKernel(),CauchyKernel(),ExponentialKernel(exponent=2),ReverseCoshKernel(exponent=2),COBRAKernel(threshold=.5)])
def test_kernels(kernel):
    D=np.array([[0.,.5,2.]])
    out=kernel(D)
    assert out.shape==D.shape
    assert np.all(np.isfinite(out))
    params=kernel.get_params(); kernel.set_params(**params)
    assert isinstance(kernel.is_continuous(),bool) and isinstance(kernel.is_discrete(),bool)

@pytest.mark.parametrize('loss',[MSELoss(),MAELoss(),HuberLoss(delta=1.0),HingeLoss(),LogLoss(),QuantileLoss(tau=.5)])
def test_losses(loss):
    yt=np.array([0.,1.,1.]); yp=np.array([.1,.8,.6])
    v=loss(yt,yp)
    assert np.all(np.isfinite(np.asarray(v)))

@pytest.mark.parametrize('norm',[MinMaxNormalizer(), StandardNormalizer()])
def test_normalizers(norm):
    z=norm.fit_transform(X)
    assert z.shape==X.shape
    assert np.all(np.isfinite(z))
    assert norm.transform(X).shape==X.shape


def test_weighted_mean_all_paths():
    a=WeightedMeanAggregator()
    assert a.aggregate([1,3])==2
    assert a.aggregate([1,3],[1,1])==2
    assert a.aggregate([1,3],[0,0],fallback=7)==7
    with pytest.raises(ValueError): a.aggregate([])
    p=a.aggregate_proba([[.8,.2],[.2,.8]],[1,1]); assert np.allclose(p,[.5,.5])
    p2=a.aggregate_proba([[.8,.2],[.2,.8]],None); assert np.allclose(p2,[.5,.5])


def test_weighted_vote_all_paths():
    a=WeightedVoteAggregator()
    assert a.aggregate([0,0,1])==0
    assert a.aggregate([0,1,1],[.1,.2,.9])==1
    with pytest.raises(ValueError): a.aggregate([])
    with pytest.raises(ValueError): a.aggregate([0,1],[1])
    with pytest.raises(ValueError): a.aggregate_matrix([0,1],[1,2])
    out=a.aggregate_matrix([0,1],[[1,0],[0,1]]); assert np.array_equal(out,[0,1])
    p=a.aggregate_proba([0,1,1],classes=[0,1]); assert np.isclose(p.sum(),1)
    pb=a.aggregate_proba_batch([0,1],[[1,1],[2,0]],classes=[0,1]); assert pb.shape==(2,2)
