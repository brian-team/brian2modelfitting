'''
Test the simulation class - runtime
'''
import pytest
import numpy as np
from numpy.testing.utils import assert_equal, assert_raises
from brian2 import (Equations, NeuronGroup, StateMonitor, Network, ms,
                    start_scope)
from brian2.devices.device import Dummy
from brian2modelfitting import Simulation, RuntimeSimulation
from brian2modelfitting.modelfitting.simulation import (initialize_neurons,
                                                        initialize_parameter)
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
    rts = RuntimeSimulation()
    assert isinstance(rts, Simulation)


def test_initialize_parameter():
    g_init = initialize_parameter(neurons.__getattr__('g'), 100)
    assert(isinstance(g_init, Dummy))


def test_initialize_neurons():
    params_init = initialize_neurons(['g', 'E'], neurons, {'g': 100, 'E': 10})
    assert(isinstance(params_init, dict))
    assert(isinstance(params_init['g'], Dummy))
    assert(isinstance(params_init['E'], Dummy))


def test_initialize_simulation_runtime():
    start_scope()
    rts = RuntimeSimulation()
    assert_raises(TypeError, rts.initialize)

    rts.initialize(net)
    assert(isinstance(rts.network, Network))
    assert_raises(KeyError, rts.initialize, empty_net)
    assert_raises(Exception, rts.initialize, wrong_net)


def test_run_simulation_runtime(setup):
    dt, duration = setup
    start_scope()

    neurons = NeuronGroup(1, model, name='neurons')
    monitor = StateMonitor(neurons, 'I', record=True, name='monitor')
    net = Network(neurons, monitor)

    rts = RuntimeSimulation()
    rts.initialize(net)

    rts.run(duration, {'g': 100, 'E': 10}, ['g', 'E'])
    I = getattr(rts.network['monitor'], 'I')
    assert_equal(np.shape(I), (1, duration/dt))
