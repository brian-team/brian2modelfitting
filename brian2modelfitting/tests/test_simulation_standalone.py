'''
Test the simulation class - standalone
'''
import pytest
import numpy as np
from numpy.testing.utils import assert_equal, assert_raises
from brian2 import (Equations, NeuronGroup, StateMonitor, Network, ms,
                    device, start_scope)
from brian2.devices.device import Dummy
from brian2modelfitting.simulator import (initialize_neurons,
                                          initialize_parameter,
                                          Simulator, CPPStandaloneSimulator)
from brian2.devices.device import reinit_devices


model = Equations('''
    I = g*(v-E) : amp
    v = 10*mvolt :volt
    g : siemens (constant)
    E : volt (constant)
    ''')

neurons = NeuronGroup(1, model, name='neurons')
monitor = StateMonitor(neurons, 'I', record=True, name='monitor')

net = Network(neurons, monitor)
empty_net = Network()
wrong_net = Network(NeuronGroup(1, model, name='neurons2'))


@pytest.fixture()
def setup(request):
    dt = 0.1 * ms
    duration = 10 * ms

    def fin():
        reinit_devices()
    request.addfinalizer(fin)

    return dt, duration


def test_init():
    sts = CPPStandaloneSimulator()
    assert isinstance(sts, Simulator)


def test_initialize_parameter():
    g_init = initialize_parameter(neurons.__getattr__('g'), 100)
    assert(isinstance(g_init, Dummy))


def test_initialize_neurons():
    params_init = initialize_neurons(['g', 'E'], neurons, {'g': 100, 'E': 10})
    assert(isinstance(params_init, dict))
    assert(isinstance(params_init['g'], Dummy))
    assert(isinstance(params_init['E'], Dummy))


def test_initialize_simulation_standalone():
    start_scope()
    sas = CPPStandaloneSimulator()
    assert_raises(TypeError, sas.initialize)
    assert_raises(TypeError, sas.initialize, net)
    assert_raises(KeyError, sas.initialize, empty_net, None)
    assert_raises(Exception, sas.initialize, wrong_net, None)

    sas.initialize(net, var_init=None)
    assert(isinstance(sas.network, Network))


def test_run_simulation_standalone(setup):
    dt, duration = setup
    start_scope()

    neurons = NeuronGroup(1, model, name='neurons')
    monitor = StateMonitor(neurons, 'I', record=True, name='monitor')
    net = Network(neurons, monitor)

    device.reinit()
    device.activate()
    device.has_been_run = False
    sas = CPPStandaloneSimulator()
    sas.initialize(net, var_init=None)

    sas.run(duration, {'g': 100, 'E': 10}, ['g', 'E'])
    I = getattr(sas.network['monitor'], 'I')
    print(I)
    assert_equal(np.shape(I), (1, duration/dt))

    sas.run(duration, {'g': 100, 'E': 10}, ['g', 'E'])
    I = getattr(sas.network['monitor'], 'I')
    assert_equal(np.shape(I), (1, 2 * duration/dt))
