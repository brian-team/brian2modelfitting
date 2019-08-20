'''
Test the modelfitting module
'''
import numpy as np
import pytest
from numpy.testing.utils import assert_equal

from brian2 import zeros, Equations, SpikeMonitor
from brian2 import nS, mV, volt, ms
from brian2 import NeuronGroup, StateMonitor, TimedArray

from brian2modelfitting import (NevergradOptimizer, SkoptOptimizer, TraceFitter,
                                OnlineTraceFitter, SpikeFitter, MSEMetric)
from brian2modelfitting import Simulation, Metric, Optimizer

from brian2modelfitting.modelfitting.modelfitting import (get_param_dic, get_spikes)
from abc import ABCMeta

E = 40*mV
input_traces = zeros((10,5))*volt
for i in range(5):
    input_traces[5:,i]=i*10*mV

output_traces = 10*nS*input_traces

model = Equations('''
    I = g*(v-E) : amp
    g : siemens (constant)
    ''')

n_opt = NevergradOptimizer()
metric =  MSEMetric()


def test_get_param_dic():
    d = get_param_dic([1, 2], ['a', 'b'], 2, 2)
    assert isinstance(d, dict)
    assert_equal(d, {'a': [1, 1, 1, 1], 'b': [2, 2, 2, 2]})

    d = get_param_dic([[1, 3], [2, 4]], ['a', 'b'], 1, 1)
    assert_equal(d, {'a': [1, 2], 'b': [3, 4]})

    d = get_param_dic([[1, 3], [2, 4]], ['a', 'b'], 1, 2)
    assert_equal(d, {'a': [1, 2], 'b': [3, 4]})

    d = get_param_dic([[1, 3], [2, 4]], ['a', 'b'], 2, 1)
    assert_equal(d, {'a': [1, 1, 2, 2], 'b': [3, 3, 4, 4]})


def test_get_spikes():
    # needs spike monitor to be run
    pass


def test_tracefitter_init():
    tf = TraceFitter(dt=0.1*ms,
                     model=model,
                     input_var='v',
                     output_var='I',
                     input=input_traces,
                     output=output_traces,
                     n_samples=10,)

    attr_fitter = ['dt', 'results_', 'simulator', 'parameter_names', 'n_traces',
                   'duration', 'n_neurons', 'n_samples', 'method', 'threshold',
                   'reset', 'refractory', 'input', 'output', 'output_var',
                   'best_res', 'input_traces', 'model', 'network', 'optimizer',
                   'metric',]
    for attr in attr_fitter:
        assert hasattr(tf, attr)

    assert tf.metric is None
    assert tf.optimizer is None
    assert tf.best_res is None

    attr_tracefitter = ['input_traces', 'model', 'neurons', 'network', 'simulator']
    for attr in attr_tracefitter:
        assert hasattr(tf, attr)

    assert isinstance(tf.network['neurons'], NeuronGroup)
    assert isinstance(tf.network['monitor'], StateMonitor)
    assert isinstance(tf.simulator, Simulation)
    assert isinstance(tf.input_traces, TimedArray)
    assert isinstance(tf.model, Equations)


def test_tracefitter_init_errors():
    with pytest.raises(Exception):
        TraceFitter(dt=0.1*ms, model=model, input=input_traces,
                    n_samples=10,
                    output=output_traces,
                    output_var='I',
                    input_var='Exception',)

    with pytest.raises(Exception):
        TraceFitter(dt=0.1*ms, model=model, input=input_traces,
                    n_samples=10,
                    output=output_traces,
                    input_var='v',
                    output_var='Exception',)

    with pytest.raises(Exception):
        TraceFitter(dt=0.1*ms, model=model, input=input_traces,
                    n_samples=10,
                    output=[1],
                    input_var='v',
                    output_var='I',)


def test_namespace():
    pass


def test_fitter_fit():
    tf = TraceFitter(dt=0.1*ms,
                     model=model,
                     input_var='v',
                     output_var='I',
                     input=input_traces,
                     output=output_traces,
                     n_samples=2,)

    results, errors = tf.fit(n_rounds=2,
                             optimizer=n_opt,
                             metric=metric,
                             g=[1*nS, 30*nS])

    attr_fit = ['optimizer','metric','best_res']
    for attr in attr_fit:
        assert hasattr(tf, attr)

    assert isinstance(tf.metric, Metric)
    assert isinstance(tf.optimizer, Optimizer)
    assert isinstance(tf.simulator, Simulation)

    assert isinstance(results, dict)
    assert isinstance(errors, float)
    assert 'g' in results.keys()

    assert_equal(results, tf.best_res)


def test_fitter_fit_errors():
    tf = TraceFitter(dt=0.1*ms,
                     model=model,
                     input_var='v',
                     output_var='I',
                     input=input_traces,
                     output=output_traces,
                     n_samples=2,)

    with pytest.raises(TypeError):
        tf.fit(n_rounds=2,
                optimizer=None,
                metric=metric,
                g=[1*nS, 30*nS])

    with pytest.raises(TypeError):
        tf.fit(n_rounds=2,
                optimizer=n_opt,
                metric=None,
                g=[1*nS, 30*nS])


def test_fit_restart():
    pass

def test_fit_restart_errors():
    pass


def test_fit_param_init():
    pass

def test_fitter_generate_traces():
    pass

def test_fitter_results():
    pass


# OnlineTraceFitter class
def test_onlinetracefitter_init():
    pass

def test_onlinetracefitter_fit():
    pass
