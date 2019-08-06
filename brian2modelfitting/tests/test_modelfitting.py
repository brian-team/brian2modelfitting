'''
Test the modelfitting module
'''
import numpy as np
from numpy.testing.utils import assert_equal

from brian2 import zeros, Equations, SpikeMonitor
from brian2 import nS, mV, volt, ms
from brian2modelfitting import fit_traces, NevergradOptimizer, SkoptOptimizer
from brian2modelfitting.modelfitting.modelfitting import (make_dic, get_param_dic,
                                                          get_spikes, setup_fit,
                                                          setup_fit, setup_neuron_group,
                                                          calc_errors_spikes, calc_errors_traces,
                                                          optim_iter)


def test_make_dic():
    names = ['a', 'b']
    values = [1, 2]
    result_dic = make_dic(names, values)

    assert isinstance(result_dic, dict)
    assert_equal(result_dic, {'a': 1, 'b': 2})


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


def test_setup_fit():
    # elicit all the errors
    # check for None and Metric
    # check for simulator setups
    pass


def test_setup_neuron_group():
    # check if Neurons is a neuron group with correct name, method, threshold, refra, rest, states
    pass


def test_calc_errors_spikes():
    # needs monitor to be run

    pass


def test_calc_error_traces():
    # needs monitor to be run

    pass


def test_optim_iter():
    #  mock everything setup
    pass


def test_fit_traces_errors():
    # just erorrs
    # separate files for functional test

    pass


def test_fit_spikes_errors():
    pass
