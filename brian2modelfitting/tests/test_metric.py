'''
Test the metric class
'''
import numpy as np
from numpy.testing.utils import assert_equal, assert_raises, assert_almost_equal
from brian2 import ms
from brian2.units.fundamentalunits import DimensionMismatchError
from brian2modelfitting import Metric, MSEMetric, GammaFactor, firing_rate, get_gamma_factor


def test_firing_rate():
    assert_equal(firing_rate([1, 2, 3]), 1)
    assert_equal(firing_rate([1, 1.2, 1.4, 1.6, 1.8]), 5)
    assert_almost_equal(firing_rate([1.1, 1.2, 1.3, 1.4, 1.5]), 10)


def test_get_gamma_factor():
    # two same traces
    # two different traces
    # src = [7, 9, 11] * ms
    # src2 = [1, 2, 3] * ms
    # trg = [0, 2, 4, 6, 8] * ms
    #
    # get_gamma_factor(trg, trg, delta=12*ms, dt=0.1*ms)
    # get_gamma_factor(src2, trg, delta=12*ms, dt=0.1*ms)
    # get_gamma_factor(src, src2, delta=5*ms, dt=0.1*ms)
    pass


def test_init():
    Metric()
    MSEMetric()
    GammaFactor(10*ms, 0.1*ms)

    assert_raises(AssertionError, GammaFactor, dt=0.1*ms)


def test_calc_mse():
    mse = MSEMetric()
    out = np.random.rand(2, 20)
    inp = np.random.rand(10, 20)

    errors = mse.calc(inp, out, 2)
    assert_equal(np.shape(errors), (5,))
    assert_equal(mse.calc(out, out, 2), [0.])
    assert(np.all(mse.calc(inp, out, 2) > 0))


def test_calc_gf():
    assert_raises(TypeError, GammaFactor)
    assert_raises(TypeError, GammaFactor, delta=10*ms)
    assert_raises(AssertionError, GammaFactor, dt=0.01*ms)
    assert_raises(DimensionMismatchError, GammaFactor, delta=10*ms, dt=0.01)
    assert_raises(DimensionMismatchError, GammaFactor, delta=10, dt=0.01*ms)

    inp_gf = np.round(np.sort(np.random.rand(10, 5) * 10), 2)
    out_gf = np.round(np.sort(np.random.rand(2, 5) * 10), 2)

    gf = GammaFactor(delta=10*ms, dt=1*ms)
    errors = gf.calc(inp_gf, out_gf, 2)
    assert_equal(np.shape(errors), (5,))
    assert(np.all(errors > 0))
    errors = gf.calc(out_gf, out_gf, 2)
    assert_almost_equal(errors, [0.])


def test_get_features_mse():
    mse = MSEMetric()
    out_mse = np.random.rand(2, 20)
    inp_mse = np.random.rand(6, 20)

    mse.get_features(inp_mse, out_mse, 2)
    assert_equal(np.shape(mse.features), (6,))
    assert(np.all(np.array(mse.features) > 0))

    mse.get_features(out_mse, out_mse, 2)
    assert_equal(np.shape(mse.features), (2,))
    assert_equal(mse.features, [0., 0.])


def test_get_errors_mse():
    mse = MSEMetric()
    mse.get_errors(np.random.rand(10, 1), 2)
    assert_equal(np.shape(mse.errors), (5,))
    assert(np.all(np.array(mse.errors) > 0))

    mse.get_errors(np.zeros((10, 1)), 5)
    assert_equal(np.shape(mse.errors), (2,))
    assert_equal(mse.errors, [0., 0.])


def test_get_features_gamma():
    inp_gf = np.round(np.sort(np.random.rand(6, 5) * 10), 2)
    out_gf = np.round(np.sort(np.random.rand(2, 5) * 10), 2)

    gf = GammaFactor(delta=10*ms, dt=1*ms)
    gf.get_features(inp_gf, out_gf, 2)
    assert_equal(np.shape(gf.features), (6,))
    assert(np.all(np.array(gf.features) > 0))

    gf.get_features(out_gf, out_gf, 2)
    assert_equal(np.shape(gf.features), (2,))
    assert_almost_equal(gf.features, [0., 0.])


def test_get_errors_gamma():
    gf = GammaFactor(delta=10*ms, dt=1*ms)
    gf.get_errors(np.random.rand(10, 1), 2)
    assert_equal(np.shape(gf.errors), (5,))
    assert(np.all(np.array(gf.errors) > 0))

    gf.get_errors(np.zeros((10, 1)), 5)
    assert_equal(np.shape(gf.errors), (2,))
    assert_almost_equal(gf.errors, [0., 0.])
