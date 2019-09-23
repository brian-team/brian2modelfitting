'''
Test the simulation class - runtime
'''
import pytest
import numpy as np
from numpy.testing.utils import assert_equal, assert_raises
from brian2 import (Equations, NeuronGroup, StateMonitor, Network, ms,
                    start_scope, mV)
from brian2.devices.device import Dummy
from brian2modelfitting.simulator import (initialize_neurons,
                                          initialize_parameter,
                                          Simulator, RuntimeSimulator)
from brian2.devices.device import reinit_devices


model = Equations('''
    I = g*(v-E) : amp
    v = 10*mvolt :volt
    g : siemens (constant)
    E : volt (constant)
    ''')

model2 = Equations('''
                  dv/dt = (gL*(EL-v)+gL*DeltaT*exp((v-VT)/DeltaT) + I)/C : volt
                  I = 20* nA :amp
                  gL: siemens (constant)
                  C: farad (constant)
                  ''',
                  EL=-70*mV,
                  VT=-50*mV,
                  DeltaT=2*mV,)

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
    rts = RuntimeSimulator()
    assert isinstance(rts, Simulator)


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
    rts = RuntimeSimulator()
    assert_raises(TypeError, rts.initialize)

    rts.initialize(net, var_init=None)
    assert(isinstance(rts.network, Network))
    assert_raises(KeyError, rts.initialize, empty_net, None)
    assert_raises(Exception, rts.initialize, wrong_net, None)
    assert_raises(TypeError, rts.initialize, Network)


def test_run_simulation_runtime(setup):
    dt, duration = setup
    start_scope()

    neurons = NeuronGroup(1, model, name='neurons')
    monitor = StateMonitor(neurons, 'I', record=True, name='monitor')
    net = Network(neurons, monitor)

    rts = RuntimeSimulator()
    rts.initialize(net, var_init=None)

    rts.run(duration, {'g': 100, 'E': 10}, ['g', 'E'])
    I = getattr(rts.network['monitor'], 'I')
    assert_equal(np.shape(I), (1, duration/dt))


def test_run_simulation_runtime_var_init(setup):
    dt, duration = setup
    start_scope()

    neurons = NeuronGroup(1, model2, name='neurons')
    monitor = StateMonitor(neurons, 'v', record=True, name='monitor')
    net = Network(neurons, monitor)

    rts = RuntimeSimulator()
    rts.initialize(net, var_init={'v': -60*mV})

    rts.run(duration, {'gL': 100, 'C': 10}, ['gL', 'C'])
    v = getattr(rts.network['monitor'], 'v')
    assert_equal(np.shape(v), (1, duration/dt))
