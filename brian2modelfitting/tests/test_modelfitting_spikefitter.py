'''
Test the modelfitting module
'''
import pytest
import numpy as np
from numpy.testing.utils import assert_equal
from brian2 import (Equations, NeuronGroup, SpikeMonitor, TimedArray,
                    nS, nF, mV, ms, nA, amp, start_scope)
from brian2modelfitting import (NevergradOptimizer, SpikeFitter, GammaFactor,
                                Simulation, Metric, Optimizer)
from brian2modelfitting.modelfitting.modelfitting import get_spikes
from brian2.devices.device import reinit_devices

dt_def = 0.01 * ms
input_current = np.hstack([np.zeros(int(5*ms/dt_def)), np.ones(int(5*ms/dt_def)*5), np.zeros(5*int(5*ms/dt_def))])* 5 * nA
inp_trace = np.array([input_current])
output = [np.array([ 9.26, 13.54, 17.82, 22.1, 26.38])]

EL = -70*mV
VT = -50*mV
DeltaT = 2*mV
model = Equations('''
                  dv/dt = (gL*(EL-v)+gL*DeltaT*exp((v-VT)/DeltaT) + I)/C : volt
                  gL: siemens (constant)
                  C: farad (constant)
                  ''')

n_opt = NevergradOptimizer()
metric = GammaFactor(dt_def, 60*ms)


@pytest.fixture()
def setup(request):
    dt = 0.01 * ms

    def fin():
        reinit_devices()
    request.addfinalizer(fin)

    return dt

def test_get_spikes():
    # needs spike monitor to be run
    pass

def test_spikefitter_init(setup):
    dt = setup
    start_scope()
    sf = SpikeFitter(model=model, input_var='I', dt=dt,
                     input=inp_trace*amp, output=output,
                     n_samples=2,
                     threshold='v > -50*mV',
                     reset='v = -70*mV',)

    attr_fitter = ['dt', 'results_', 'simulator', 'parameter_names', 'n_traces',
                   'duration', 'n_neurons', 'n_samples', 'method', 'threshold',
                   'reset', 'refractory', 'input', 'output', 'output_var',
                   'best_res', 'input_traces', 'model', 'network', 'optimizer',
                   'metric']
    for attr in attr_fitter:
        assert hasattr(sf, attr)

    assert sf.metric is None
    assert sf.optimizer is None
    assert sf.best_res is None

    attr_spikefitter = ['input_traces', 'model', 'neurons', 'network',
                        'simulator']
    for attr in attr_spikefitter:
        assert hasattr(sf, attr)

    assert isinstance(sf.network['neurons'], NeuronGroup)
    assert isinstance(sf.network['monitor'], SpikeMonitor)
    assert isinstance(sf.simulator, Simulation)
    assert isinstance(sf.input_traces, TimedArray)
    assert isinstance(sf.model, Equations)


def test_tracefitter_init_errors(setup):
    dt = setup
    with pytest.raises(Exception):
        SpikeFitter(model=model, input_var='Exception', dt=dt,
                    input=inp_trace*amp, output=output,
                    n_samples=2,
                    threshold='v > -50*mV',
                    reset='v = -70*mV',)


def test_spikefitter_fit(setup):
    dt = setup
    sf = SpikeFitter(model=model, input_var='I', dt=dt,
                     input=inp_trace*amp, output=output,
                     n_samples=2,
                     threshold='v > -50*mV',
                     reset='v = -70*mV',)

    results, errors = sf.fit(n_rounds=2,
                             optimizer=n_opt,
                             metric=metric,
                             gL=[20*nS, 40*nS],
                             C=[0.5*nF, 1.5*nF])

    attr_fit = ['optimizer', 'metric', 'best_res']
    for attr in attr_fit:
        assert hasattr(sf, attr)

    assert isinstance(sf.metric, Metric)
    assert isinstance(sf.optimizer, Optimizer)
    assert isinstance(sf.simulator, Simulation)

    assert isinstance(results, dict)
    assert isinstance(errors, float)
    assert 'gL' in results.keys()
    assert 'C' in results.keys()

    assert_equal(results, sf.best_res)


def test_spikefitter_calc_errors():
    pass


def test_spikefitter_generate_traces():
    pass