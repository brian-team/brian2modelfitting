'''
Test the simulation class
'''
import numpy as np
from numpy.testing.utils import assert_equal, assert_raises
from brian2 import (Equations, NeuronGroup, StateMonitor, Network, ms,
                    defaultclock, device, start_scope)
from brian2.devices.device import Dummy
from brian2modelfitting import Simulation, RuntimeSimulation, CPPStandaloneSimulation
from brian2modelfitting.modelfitting.simulation import (initialize_neurons,
                                                        initialize_parameter)


model = Equations('''
    I = g*(v-E) : amp
    v = 10*mvolt :volt
    g : siemens (constant)
    E : volt (constant)
    ''')

dt = 0.1 * ms
duration = 10 * ms
defaultclock.dt = dt

neurons = NeuronGroup(1, model, name='neurons')
monitor = StateMonitor(neurons, 'I', record=True, name='monitor')

net = Network(neurons, monitor)
empty_net = Network()
wrong_net = Network(NeuronGroup(1, model, name='neurons2'))


def test_init():
    Simulation()
    RuntimeSimulation()
    CPPStandaloneSimulation()


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
    sas = CPPStandaloneSimulation()
    assert_raises(TypeError, sas.initialize)
    assert_raises(KeyError, sas.initialize, empty_net)
    assert_raises(Exception, sas.initialize, wrong_net)

    sas.initialize(net)
    assert(isinstance(sas.network, Network))


def test_initialize_simulation_runtime():
    start_scope()
    rts = RuntimeSimulation()
    assert_raises(TypeError, rts.initialize)

    rts.initialize(net)
    assert(isinstance(rts.network, Network))
    assert_raises(KeyError, rts.initialize, empty_net)
    assert_raises(Exception, rts.initialize, wrong_net)


def test_run_simulation_standalone():
    start_scope()

    neurons = NeuronGroup(1, model, name='neurons')
    monitor = StateMonitor(neurons, 'I', record=True, name='monitor')
    net = Network(neurons, monitor)

    device.has_been_run = False
    sas = CPPStandaloneSimulation()
    sas.initialize(net)

    sas.run(duration, {'g': 100, 'E': 10}, ['g', 'E'])
    I = getattr(sas.network['monitor'], 'I')
    assert_equal(np.shape(I), (1, duration/dt))

    # check the re-run
    sas.run(duration, {'g': 100, 'E': 10}, ['g', 'E'])
    I = getattr(sas.network['monitor'], 'I')
    assert_equal(np.shape(I), (1, 2 * duration/dt))


def test_run_simulation_runtime():
    start_scope()

    neurons = NeuronGroup(1, model, name='neurons')
    monitor = StateMonitor(neurons, 'I', record=True, name='monitor')
    net = Network(neurons, monitor)

    rts = RuntimeSimulation()
    rts.initialize(net)

    rts.run(duration, {'g': 100, 'E': 10}, ['g', 'E'])
    I = getattr(rts.network['monitor'], 'I')
    assert_equal(np.shape(I), (1, duration/dt))
