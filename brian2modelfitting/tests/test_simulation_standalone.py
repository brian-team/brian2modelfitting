'''
Test the simulation class - standalone
'''
import pytest
import numpy as np
from numpy.testing import assert_equal, assert_raises
from brian2 import (Equations, NeuronGroup, StateMonitor, Network, ms,
                    device, start_scope)
from brian2.devices.device import Dummy
from brian2modelfitting.simulator import (initialize_neurons,
                                          initialize_parameter,
                                          Simulator, CPPStandaloneSimulator)
from brian2.devices.device import reinit_devices, set_device


model = Equations('''
    I = g*(v-E) : amp
    v = 10*mvolt :volt
    g : siemens (constant)
    E : volt (constant)
    ''')

empty_net = Network()


@pytest.fixture
def setup(request):
    dt = 0.1 * ms
    duration = 10 * ms

    neurons = NeuronGroup(1, model, name='neurons')
    monitor = StateMonitor(neurons, 'I', record=True, name='statemonitor')

    net = Network(neurons, monitor)

    def fin():
        reinit_devices()
    request.addfinalizer(fin)

    return net, dt, duration


@pytest.fixture
def setup_standalone(request):
    # Workaround to avoid issues with Network instances still around
    Network.__instances__().clear()
    set_device('cpp_standalone', directory=None)
    dt = 0.1 * ms
    duration = 10 * ms
    neurons = NeuronGroup(1, model + Equations('iteration: integer (constant, shared)'), name='neurons')
    monitor = StateMonitor(neurons, 'I', record=True, name='statemonitor')

    net = Network(neurons, monitor)

    def fin():
        reinit_devices()
        set_device('runtime')
    request.addfinalizer(fin)

    return net, dt, duration


def test_init():
    sts = CPPStandaloneSimulator()
    assert isinstance(sts, Simulator)


def test_initialize_parameter():
    neurons = NeuronGroup(1, model, name='neurons')
    g_init = initialize_parameter(neurons.__getattr__('g'), 100)
    assert(isinstance(g_init, Dummy))


def test_initialize_neurons():
    neurons = NeuronGroup(1, model, name='neurons')
    params_init = initialize_neurons(['g', 'E'], neurons, {'g': 100, 'E': 10})
    assert(isinstance(params_init, dict))
    assert(isinstance(params_init['g'], Dummy))
    assert(isinstance(params_init['E'], Dummy))


def test_initialize_simulation_standalone(setup):
    start_scope()
    net, _, _ = setup
    sas = CPPStandaloneSimulator()
    assert_raises(TypeError, sas.initialize)
    assert_raises(TypeError, sas.initialize, net)
    assert_raises(KeyError, sas.initialize, empty_net, None)
    wrong_net = Network(NeuronGroup(1, model, name='neurons2'))
    assert_raises(Exception, sas.initialize, wrong_net, None)

    sas.initialize(net, var_init=None, name='test')
    assert(isinstance(sas.networks['test'], Network))


def test_run_simulation_standalone(setup_standalone):
    net, dt, duration = setup_standalone

    sas = CPPStandaloneSimulator()
    sas.initialize(net, var_init=None)

    sas.run(duration, {'g': 100, 'E': 10}, ['g', 'E'], iteration=0)
    I = getattr(sas.statemonitor, 'I')
    assert_equal(np.shape(I), (1, duration/dt))
