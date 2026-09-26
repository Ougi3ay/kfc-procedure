import numpy as np
import pytest
from kfc_procedure.cobra.core.cv.kfold import KFoldCV
from kfc_procedure.cobra.core.cv.stratified_kfold import StratifiedKFoldCV
from kfc_procedure.cobra.core.cv.time_series import TimeSeriesCV
from kfc_procedure.cobra.core.splitters.holdout import RandomHoldoutSplitter
from kfc_procedure.cobra.core.splitters.overlap import OverlapSplitter
from kfc_procedure.cobra.core.adapters.one_parameter import OneParameterKernelAdapter
from kfc_procedure.cobra.core.adapters.two_parameter import TwoParameterKernelAdapter
from kfc_procedure.cobra.core.estimators.mean_regressor import MeanRegressor
from kfc_procedure.cobra.core.estimators.sklearn import SklearnEstimator
from sklearn.linear_model import LinearRegression, LogisticRegression

X=np.arange(40,dtype=float).reshape(20,2); y=np.array([0,1]*10)

@pytest.mark.parametrize('cv',[KFoldCV(n_splits=4,shuffle=True,random_state=1),StratifiedKFoldCV(n_splits=4,random_state=1),TimeSeriesCV(n_splits=4,test_size=2)])
def test_cv(cv):
    splits=list(cv.split(X,y)); assert len(splits)==cv.get_n_splits(X,y) if isinstance(cv,KFoldCV) else len(splits)>0
    for sp in splits: assert len(sp.train_idx)>0 and len(sp.eval_idx)>0


def test_splitters():
    for s in [RandomHoldoutSplitter(calibration_size=.25,random_state=1),OverlapSplitter(split_ratio=.6,overlap=.2,shuffle=True,random_state=1)]:
        out=s.split(X,y); assert out is not None


def test_adapters():
    D=np.array([[0.,1.],[2.,3.]])
    a=OneParameterKernelAdapter(bandwidth=2.0); assert a.transform(D).shape==D.shape
    assert a.get_params()['bandwidth']==2.0; a.set_params(bandwidth=1.0); assert a.parameter_vector().size>=1
    b=TwoParameterKernelAdapter(alpha=2.0,beta=1.0); assert b.transform(D).shape==D.shape; assert b.parameter_vector().size==2


def test_estimators():
    r=MeanRegressor().fit(X,np.arange(20.)); assert r.predict(X[:3]).shape==(3,)
    s=SklearnEstimator(LinearRegression).fit(X,np.arange(20.)); assert s.predict(X[:2]).shape==(2,)
    c=SklearnEstimator(LogisticRegression, max_iter=100).fit(X,y); assert c.predict_proba(X[:2]).shape==(2,2)
    p=c.get_params(); c.set_params(**{k:v for k,v in p.items() if k!='estimator_cls'})
