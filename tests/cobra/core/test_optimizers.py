import numpy as np
import pytest
from kfc_procedure.cobra.core.optimizers._utils import central_difference_gradient, forward_difference_gradient, spsa_gradient, complex_step_gradient, compute_gradient
from kfc_procedure.cobra.core.optimizers.gradient.gd import GradientDescentOptimizer
from kfc_procedure.cobra.core.optimizers.gradient.momentum import MomentumOptimizer
from kfc_procedure.cobra.core.optimizers.gradient.adam import AdamOptimizer
from kfc_procedure.cobra.core.optimizers.search.search import GridSearchOptimizer

f=lambda x: np.sum(np.asarray(x)**2)

def test_grad_utils():
    x=np.array([1.,2.])
    for fn in [central_difference_gradient,forward_difference_gradient,complex_step_gradient]:
        g=fn(f,x,1e-5); assert g.shape==x.shape
    g=spsa_gradient(f,x,1e-4); assert g.shape==x.shape
    for m in ['central','forward','spsa','complex']:
        g=compute_gradient(f,x,method=m,eps=1e-4); assert g.shape==x.shape
    with pytest.raises(ValueError): compute_gradient(f,x,method='bad')

@pytest.mark.parametrize('opt',[GradientDescentOptimizer(learning_rate=.1,max_iter=20,tol=1e-8,show_process=False),MomentumOptimizer(learning_rate=.1,max_iter=20,tol=1e-8,show_process=False),AdamOptimizer(learning_rate=.1,max_iter=20,tol=1e-8,show_process=False)])
def test_gradient_optimizers(opt):
    r=opt.optimize(f,np.array([1.,-1.])); assert 'x' in r and np.all(np.isfinite(r['x']))


def test_grid_search():
    o=GridSearchOptimizer(param_grid={'a':[0.,1.],'b':[0.,2.]},show_process=False)
    c=o.candidates(); assert len(c)==4
    r=o.optimize(lambda x: np.sum(np.asarray(x,dtype=float)**2)); assert r['best_index']>=0
    for strategy in ['mean','sum','max','min','median','l2']:
        o.risk_strategy=strategy; assert np.isfinite(o.reduce_risk([1,2]))
    o.risk_strategy='bad'
    with pytest.raises(ValueError): o.reduce_risk([1,2])
    assert o.select_best_index([2,1,1])==2
