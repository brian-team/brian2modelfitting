'''
Test the optimizer class
'''
import numpy as np
from numpy.testing import assert_equal, assert_raises
from brian2modelfitting import Optimizer, NevergradOptimizer, SkoptOptimizer, calc_bounds

from skopt import Optimizer as SOptimizer
from nevergrad.optimization.base import Optimizer as NOptimizer

def test_init():
    # Optimizer()
    NevergradOptimizer()
    SkoptOptimizer()

    NevergradOptimizer(method='DE')
    SkoptOptimizer(method='GP')


def test_init_wrong_method():
    assert_raises(AssertionError, NevergradOptimizer, method='foo')
    assert_raises(AssertionError, SkoptOptimizer, method='foo')


def test_init_kwds():
    pass


def test_calc_bounds():
    bounds = calc_bounds({'a'}, a=[0, 1])
    assert_equal(bounds, [[0, 1]])

    bounds = calc_bounds(['a', 'b'], a=[0, 1], b=[2, 3])
    assert_equal(bounds, [[0, 1], [2, 3]])


def test_initialize_nevergrad():
    n_opt = NevergradOptimizer()
    n_opt.initialize({'g'}, g=[1, 30], popsize=30, rounds=2)
    assert isinstance(n_opt.optim, NOptimizer)
    assert_equal(n_opt.optim.dimension, 1)

    n_opt.initialize(['g', 'E'], g=[1, 30], E=[2, 20], popsize=30, rounds=2)
    assert isinstance(n_opt.optim, NOptimizer)
    assert_equal(n_opt.optim.dimension, 2)

    assert_raises(AssertionError, n_opt.initialize, ['g'], g=[1], popsize=30, rounds=2)
    assert_raises(AssertionError, n_opt.initialize, ['g'], g=[[1, 2]], popsize=30, rounds=2)
    assert_raises(Exception, n_opt.initialize, ['g'], g=[1, 2], E=[1, 2], popsize=30, rounds=2)
    assert_raises(Exception, n_opt.initialize, ['g', 'E'], g=[1, 2], popsize=30, rounds=2)


def test_initialize_skopt():
    s_opt = SkoptOptimizer()
    s_opt.initialize({'g'}, g=[1, 30], popsize=30, rounds=2)
    assert isinstance(s_opt.optim, SOptimizer)
    assert_equal(len(s_opt.optim.space.dimensions), 1)

    s_opt.initialize({'g', 'E'}, g=[1, 30], E=[2, 20], popsize=30, rounds=2)
    assert isinstance(s_opt.optim, SOptimizer)
    assert_equal(len(s_opt.optim.space.dimensions), 2)

    assert_raises(TypeError, s_opt.initialize, ['g'], g=[1], popsize=30)
    assert_raises(Exception, s_opt.initialize, ['g'], g=[1, 2], E=[1, 2], popsize=30, rounds=2)
    assert_raises(Exception, s_opt.initialize, ['g', 'E'], g=[1, 2], popsize=30, rounds=2)


def test_ask_nevergrad():
    n_opt = NevergradOptimizer()
    n_opt.initialize(['a', 'b', 'c'], a=[0, 1], b=[0, 2], c=[0, 3],
                     popsize=30, rounds=2)

    n_samples = np.random.randint(1, 30)
    params = n_opt.ask(n_samples)
    assert len(params) == n_samples
    assert all(set(p.keys()) == {'a', 'b', 'c'} for p in params)

    for i, p_name in enumerate(['a', 'b', 'c']):
        assert all(p[p_name] <= i+1 for p in params), 'Values in params are bigger than required'
        assert all(p[p_name] >= 0 for p in params), 'Values in params are smaller than required'


def test_ask_skopt():
    s_opt = SkoptOptimizer()
    n_samples = 9
    s_opt.initialize(['a', 'b', 'c'], a=[0, 1], b=[0, 2], c=[0, 3],
                     popsize=n_samples, rounds=2)

    params = s_opt.ask(n_samples)
    assert_equal(np.shape(params), (n_samples, 3))

    for i in np.arange(0, 3):
        assert all(np.array(params)[:, i] <= i+1), 'Values in params are bigger than required'
        assert all(np.array(params)[:, i] >= 0), 'Values in params are smaller than required'


def test_tell_nevergrad():
    n_opt = NevergradOptimizer()
    n_opt.initialize(['a', 'b', 'c'], a=[0, 1], b=[0, 2], c=[0, 3],
                     popsize=30, rounds=2)

    n_samples = np.random.randint(1, 30)
    data = np.random.rand(n_samples, 3)

    params, candidates = [], []
    for row in data:
        values = {'a': row[0], 'b': row[1], 'c': row[2]}
        cand = n_opt.optim.parametrization.spawn_child(values)
        candidates.append(cand)
        params.append(values)

    n_opt.candidates = candidates

    errors = np.random.rand(n_samples)
    n_opt.tell(params, errors)
    assert_equal(n_opt.optim.num_tell, n_samples)


def test_tell_skopt():
    s_opt = SkoptOptimizer()
    n_samples = 9
    s_opt.initialize(['a', 'b', 'c'], a=[0, 1], b=[0, 2], c=[0, 3],
                     popsize=n_samples, rounds=2)

    params = s_opt.ask(n_samples)

    errors = np.random.rand(n_samples)
    s_opt.tell(params, errors)
    assert_equal(s_opt.optim.Xi, params)
    assert_equal(s_opt.optim.yi, errors)


def test_recommend_nevergrad():
    n_opt = NevergradOptimizer()
    n_opt.initialize(['a', 'b', 'c'], a=[0, 1], b=[0, 2], c=[0, 3],
                     popsize=30, rounds=2)

    n_samples = np.random.randint(1, 30)
    data = np.random.rand(n_samples, 3)

    params, candidates = [], []
    for row in data:
        values = {'a': row[0], 'b': row[1], 'c': row[2]}
        cand = n_opt.optim.parametrization.spawn_child(values)
        candidates.append(cand)
        params.append(values)

    errors = np.random.rand(n_samples)
    n_opt.candidates = candidates
    n_opt.tell(params, errors)

    ans = n_opt.recommend()
    er_min = (errors).argmin()
    assert_equal(params[er_min], ans)


def test_recommend_skopt():
    s_opt = SkoptOptimizer()
    n_samples = 9
    s_opt.initialize(['a', 'b', 'c'], a=[0, 1], b=[0, 2], c=[0, 3],
                     popsize=n_samples, rounds=2)

    params = s_opt.ask(n_samples)

    errors = np.random.rand(n_samples)
    s_opt.tell(params, errors)

    ans = s_opt.recommend()
    er_min = (errors).argmin()
    assert_equal(params[er_min], list(ans))
