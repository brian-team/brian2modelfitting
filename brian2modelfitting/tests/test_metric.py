'''
Test the metric class
'''
import numpy as np
from numpy.testing.utils import assert_equal, assert_raises, assert_almost_equal
from brian2 import ms, mV
from brian2.units.fundamentalunits import DimensionMismatchError
from brian2modelfitting import (FeatureMetric, MSEMetric, GammaFactor,
                                firing_rate, get_gamma_factor, calc_eFEL)


def test_firing_rate():
    assert_equal(firing_rate([1, 2, 3]), 1)
    assert_equal(firing_rate([1, 1.2, 1.4, 1.6, 1.8]), 5)
    assert_almost_equal(firing_rate([1.1, 1.2, 1.3, 1.4, 1.5]), 10)


def test_get_gamma_factor():
    src = [7, 9, 11] * ms
    src2 = [1, 2, 3] * ms
    trg = [0, 2, 4, 6, 8, 10] * ms

    gf0 = get_gamma_factor(trg, trg, delta=0.5*ms, time=12*ms, dt=0.1*ms)
    gf1 = get_gamma_factor(src2, trg, delta=0.5*ms, time=12*ms, dt=0.1*ms)
    gf2 = get_gamma_factor(src, src2, delta=0.5*ms, time=5*ms, dt=0.1*ms)

    assert_equal(gf0, -1)
    assert gf1 > 0  # Since data rate = 2 * model rate
    assert gf2 > -1

    gf0 = get_gamma_factor(trg, trg, delta=0.5*ms, time=12*ms, dt=0.1*ms,
                           rate_correction=False)
    gf1 = get_gamma_factor(src2, trg, delta=0.5*ms, time=12*ms, dt=0.1*ms,
                           rate_correction=False)
    gf2 = get_gamma_factor(src, src2, delta=0.5*ms, time=5*ms, dt=0.1*ms,
                           rate_correction=False)

    assert_equal(gf0, 0)
    assert gf1 > 0
    assert gf2 > 0


def test_init():
    MSEMetric()
    GammaFactor(10*ms, time=10*ms)


def test_calc_mse():
    mse = MSEMetric()
    out = np.random.rand(2, 20)
    inp = np.random.rand(5, 2, 20)

    errors = mse.calc(inp, out, 0.01*ms)
    assert_equal(np.shape(errors), (5,))
    assert_equal(mse.calc(np.tile(out, (5, 1, 1)), out, 0.1*ms),
                 np.zeros(5))
    assert(np.all(mse.calc(inp, out, 0.1*ms) > 0))


def test_calc_mse_t_start():
    mse = MSEMetric(t_start=1*ms)
    out = np.random.rand(2, 20)
    inp = np.random.rand(5, 2, 20)

    errors = mse.calc(inp, out, 0.1*ms)
    assert_equal(np.shape(errors), (5,))
    assert(np.all(errors > 0))
    # Everything before 1ms should be ignored, so having the same values for
    # the rest should give an error of 0
    inp[:, :, 10:] = out[None, :, 10:]
    assert_equal(mse.calc(inp, out, 0.1*ms), np.zeros(5))


def test_calc_gf():
    assert_raises(TypeError, GammaFactor)
    assert_raises(DimensionMismatchError, GammaFactor, delta=10*mV)
    assert_raises(DimensionMismatchError, GammaFactor, time=10)

    model_spikes = [[np.array([1, 5, 8]), np.array([2, 3, 8, 9])],  # Correct rate
                    [np.array([1, 5]), np.array([0, 2, 3, 8, 9])]]  # Wrong rate
    data_spikes = [np.array([0, 5, 9]), np.array([1, 3, 5, 6])]

    gf = GammaFactor(delta=0.5*ms, time=10*ms)
    errors = gf.calc([data_spikes]*5, data_spikes, 0.1*ms)
    assert_almost_equal(errors, np.ones(5)*-1)
    errors = gf.calc(model_spikes, data_spikes, 0.1*ms)
    assert errors[0] > -1  # correct rate
    assert errors[1] > errors[0]

    gf = GammaFactor(delta=0.5*ms, time=10*ms, rate_correction=False)
    errors = gf.calc([data_spikes]*5, data_spikes, 0.1*ms)
    assert_almost_equal(errors, np.zeros(5))
    errors = gf.calc(model_spikes, data_spikes, 0.1*ms)
    assert all(errors > 0)


def test_get_features_mse():
    mse = MSEMetric()
    out_mse = np.random.rand(2, 20)
    inp_mse = np.random.rand(5, 2, 20)

    features = mse.get_features(inp_mse, out_mse, 0.1*ms)
    assert_equal(np.shape(features), (5, 2))
    assert(np.all(np.array(features) > 0))

    features = mse.get_features(np.tile(out_mse, (5, 1, 1)), out_mse, 0.1*ms)
    assert_equal(np.shape(features), (5, 2))
    assert_equal(features, np.zeros((5, 2)))


def test_get_errors_mse():
    mse = MSEMetric()
    errors = mse.get_errors(np.random.rand(5, 10))
    assert_equal(np.shape(errors), (5,))
    assert(np.all(np.array(errors) > 0))

    errors = mse.get_errors(np.zeros((2, 10)))
    assert_equal(np.shape(errors), (2,))
    assert_equal(errors, [0., 0.])


def test_get_features_gamma():
    model_spikes = [[np.array([1, 5, 8]), np.array([2, 3, 8, 9])],  # Correct rate
                    [np.array([1, 5]), np.array([0, 2, 3, 8, 9])]]  # Wrong rate
    data_spikes = [np.array([0, 5, 9]), np.array([1, 3, 5, 6])]

    gf = GammaFactor(delta=0.5*ms, time=10*ms)
    features = gf.get_features(model_spikes, data_spikes, 0.1*ms)
    assert_equal(np.shape(features), (2, 2))
    assert(np.all(np.array(features) > -1))

    features = gf.get_features([data_spikes]*3, data_spikes, 0.1*ms)
    assert_equal(np.shape(features), (3, 2))
    assert_almost_equal(features, np.ones((3, 2))*-1)


def test_get_errors_gamma():
    gf = GammaFactor(delta=10*ms, time=10*ms)
    errors = gf.get_errors(np.random.rand(5, 10))
    assert_equal(np.shape(errors), (5,))
    assert(np.all(np.array(errors) > 0))

    errors = gf.get_errors(np.zeros((2, 10)))
    assert_equal(np.shape(errors), (2,))
    assert_almost_equal(errors, [0., 0.])


def test_calc_EFL():
    # "voltage traces" that are constant at -70*mV, -60mV, -50mV, -40mV for
    # 50ms each.
    dt = 1*ms
    voltage = np.ones((2, 200))*np.repeat([-70, -60, -50, -40], 50)*mV
    # Note that calcEFL takes times in ms
    inp_times = [[99, 150], [49, 150]]
    results = calc_eFEL(voltage, inp_times, ['voltage_base'], dt=dt)
    assert len(results) == 2
    assert all(res.keys() == {'voltage_base'} for res in results)
    assert_almost_equal(results[0]['voltage_base'], float(-60*mV))
    assert_almost_equal(results[1]['voltage_base'], float(-70*mV))


def test_get_features_feature_metric():
    # "voltage traces" that are constant at -70*mV, -60mV, -50mV, -40mV for
    # 50ms each.
    voltage_target = np.ones((2, 200)) * np.repeat([-70, -60, -50, -40], 50) * mV
    dt = 1*ms
    # The results for the first and last "parameter set" are too high/low, the
    # middle one is perfect
    voltage_model = np.ones((3, 2, 200)) * np.repeat([-70, -60, -50, -40], 50) * mV
    voltage_model[0, 0, :] += 2.5*mV
    voltage_model[0, 1, :] += 5*mV
    voltage_model[2, 0, :] -= 2.5*mV
    voltage_model[2, 1, :] -= 5*mV

    inp_times = [[99 * ms, 150 * ms], [49 * ms, 150 * ms]]

    # Default comparison: absolute difference
    feature_metric = FeatureMetric(inp_times, ['voltage_base'])
    results = feature_metric.get_features(voltage_model, voltage_target, dt=dt)
    assert len(results) == 3
    assert all(isinstance(r, dict) for r in results)
    assert all(r.keys() == {'voltage_base'} for r in results)
    assert_almost_equal(results[0]['voltage_base'], np.array([2.5*mV, 5*mV]))
    assert_almost_equal(results[1]['voltage_base'], [0, 0])
    assert_almost_equal(results[2]['voltage_base'], np.array([2.5*mV, 5*mV]))

    # Custom comparison: squared difference
    feature_metric = FeatureMetric(inp_times, ['voltage_base'],
                                   combine=lambda x, y: (x - y)**2)
    results = feature_metric.get_features(voltage_model, voltage_target, dt=dt)
    assert len(results) == 3
    assert all(isinstance(r, dict) for r in results)
    assert all(r.keys() == {'voltage_base'} for r in results)
    assert_almost_equal(results[0]['voltage_base'], np.array([(2.5*mV)**2, (5*mV)**2]))
    assert_almost_equal(results[1]['voltage_base'], [0, 0])
    assert_almost_equal(results[2]['voltage_base'], np.array([(2.5*mV)**2, (5*mV)**2]))


def test_get_errors_feature_metric():
    # Fake results
    features = [{'feature1': np.array([0, 0.5]),
                'feature2': np.array([1, 2])},
                {'feature1': np.array([0, 0]),
                 'feature2': np.array([0, 0])},
                {'feature1': np.array([1, 2]),
                 'feature2': np.array([0, 0.5])}]

    # All features are weighed the same
    inp_times = [[99*ms, 150*ms], [49*ms, 150*ms]]  # Not used
    feature_metric = FeatureMetric(inp_times, ['feature1', 'feature2'])
    results = feature_metric.get_errors(features)
    assert len(results) == 3
    assert_almost_equal(results, [3.5, 0, 3.5])

    # First feature is weighted twice as high
    feature_metric = FeatureMetric(inp_times, ['feature1', 'feature2'],
                                   weights={'feature1': 2, 'feature2': 1})
    results = feature_metric.get_errors(features)
    assert len(results) == 3
    assert_almost_equal(results, [4, 0, 6.5])
