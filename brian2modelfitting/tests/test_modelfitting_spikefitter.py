'''
Test the modelfitting module
'''
import numpy as np
from numpy.testing.utils import assert_equal

from brian2 import zeros, Equations, SpikeMonitor
from brian2 import nS, mV, volt, ms
from brian2modelfitting import (NevergradOptimizer, SkoptOptimizer, TraceFitter,
                                OnlineTraceFitter, SpikeFitter, MSEMetric)
from brian2modelfitting.modelfitting.modelfitting import (get_param_dic, get_spikes)
from abc import ABCMeta

# input_traces = zeros((10,5))*volt
# for i in range(5):
#     input_traces[5:,i]=i*10*mV
#
# output_traces = 10*nS*input_traces
#
# model = Equations('''
#     I = g*(v-E) : amp
#     g : siemens (constant)
#     E : volt (constant)
#     ''')
#
# n_opt = NevergradOptimizer()
# metric =  MSEMetric()

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


def test_fitter_setup_neuron_group():
    pass

def test_fitter_optimization_iter():
    pass

def test_fitter_fit():
    pass

def test_fitter_results():
    pass

def test_fitter_generate():
    pass

# SpikeFitter class
def test_spikefitter_init():
    pass

def test_spikefitter_calc_errors():
    pass

def test_spikefitter_generate_traces():
    pass
