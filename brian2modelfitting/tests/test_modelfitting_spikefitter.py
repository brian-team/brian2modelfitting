'''
Test the modelfitting module
'''
import pytest
import numpy as np
from numpy.testing import assert_equal
from brian2 import (Equations, NeuronGroup, SpikeMonitor, TimedArray,
                    nS, nF, mV, ms, nA, amp, run)
from brian2modelfitting import (NevergradOptimizer, SpikeFitter, GammaFactor,
                                Simulator, Metric, Optimizer, MSEMetric)
from brian2modelfitting.fitter import get_spikes
from brian2.devices.device import reinit_devices

dt_def = 0.01 * ms
input_current = np.hstack([np.zeros(int(5*ms/dt_def)), np.ones(int(5*ms/dt_def)*5), np.zeros(5*int(5*ms/dt_def))])* 5 * nA
inp_trace = np.array([input_current])
output = [np.array([9.26, 13.54, 17.82, 22.1, 26.38])]

EL = -70*mV
VT = -50*mV
DeltaT = 2*mV
model = Equations('''
                  dv/dt = (gL*(EL-v)+gL*DeltaT*exp((v-VT)/DeltaT) + I)/C : volt
                  gL: siemens (constant)
                  C: farad (constant)
                  ''')

n_opt = NevergradOptimizer()
metric = GammaFactor(60*ms, 60*ms)


@pytest.fixture
def setup(request):
    dt = 0.01 * ms
    sf = SpikeFitter(model=model, input_var='I', dt=dt,
                     input=inp_trace*amp, output=output,
                     n_samples=2,
                     threshold='v > -50*mV',
                     reset='v = -70*mV',)

    def fin():
        reinit_devices()
    request.addfinalizer(fin)

    return dt, sf


@pytest.fixture
def setup_spikes(request):
    def fin():
        reinit_devices()
    request.addfinalizer(fin)
    EL = -70*mV
    VT = -50*mV
    DeltaT = 2*mV
    C = 1*nF
    gL = 30*nS
    I = TimedArray(input_current, dt=0.01 * ms)
    model = Equations('''
                      dv/dt = (gL*(EL-v)+gL*DeltaT*exp((v-VT)/DeltaT) + I(t))/C : volt
                      ''')
    group = NeuronGroup(1, model,
                        threshold='v > -50*mV',
                        reset='v = -70*mV',
                        method='exponential_euler')
    group.v = -70 * mV
    spike_mon = SpikeMonitor(group)
    run(60*ms)
    spikes = getattr(spike_mon, 't_')

    return spike_mon, spikes


def test_get_spikes(setup_spikes):
    spike_mon, spikes = setup_spikes
    gs = get_spikes(spike_mon, 1, 1)
    assert isinstance(gs, list)
    assert isinstance(gs[0][0], np.ndarray)
    assert_equal(gs, [[np.array(spikes)]])


def test_spikefitter_init(setup):
    dt, sf = setup
    attr_fitter = ['dt', 'simulator', 'parameter_names', 'n_traces',
                   'duration', 'n_neurons', 'n_samples', 'method', 'threshold',
                   'reset', 'refractory', 'input', 'output', 'output_var',
                   'best_params', 'input_traces', 'model', 'optimizer',
                   'metric']
    for attr in attr_fitter:
        assert hasattr(sf, attr)

    assert sf.metric is None
    assert sf.optimizer is None
    assert sf.best_params is None

    attr_spikefitter = ['input_traces', 'model', 'simulator']
    for attr in attr_spikefitter:
        assert hasattr(sf, attr)

    assert isinstance(sf.input_traces, TimedArray)
    assert isinstance(sf.model, Equations)


def test_spikefitter_init_errors(setup):
    dt, _ = setup
    with pytest.raises(Exception):
        SpikeFitter(model=model, input_var='Exception', dt=dt,
                    input=inp_trace*amp, output=output,
                    n_samples=2,
                    threshold='v > -50*mV',
                    reset='v = -70*mV')

    with pytest.raises(ValueError):
        sf = SpikeFitter(model=model, input_var='I', dt=dt,
                         input=inp_trace * amp, output=output,
                         n_samples=2,
                         threshold='v > -50*mV',
                         reset='v = -70*mV',
                         param_init={'V': -70*mV})  # name is "v" not "V"


def test_spikefitter_fit(setup):
    dt, sf = setup
    results, errors = sf.fit(n_rounds=2,
                             optimizer=n_opt,
                             metric=metric,
                             gL=[20*nS, 40*nS],
                             C=[0.5*nF, 1.5*nF])
    assert sf.simulator.neurons.iteration == 1
    attr_fit = ['optimizer', 'metric', 'best_params']
    for attr in attr_fit:
        assert hasattr(sf, attr)

    assert len(sf.metric) == 1 and isinstance(sf.metric[0], Metric)
    assert isinstance(sf.optimizer, Optimizer)

    assert isinstance(results, dict)
    assert isinstance(errors, float)
    assert 'gL' in results.keys()
    assert 'C' in results.keys()

    assert_equal(results, sf.best_params)


def test_spikefitter_fit_errors(setup):
    dt, sf = setup
    class NaiveOptimizer:
        def __init__(self):
            self.best = []
    naive_opt = NaiveOptimizer()
    with pytest.raises(TypeError):
        results, errors = sf.fit(n_rounds=2,
                                 optimizer=n_opt,
                                 metric=MSEMetric(), #testing a wrong metric
                                 gL=[20*nS, 40*nS],
                                 C=[0.5*nF, 1.5*nF])
    with pytest.raises(TypeError):
        results, errors = sf.fit(n_rounds=2,
                                 optimizer=naive_opt, #testing a Non-Optimizer child
                                 metric=metric,
                                 gL=[20*nS, 40*nS],
                                 C=[0.5*nF, 1.5*nF])

def test_fitter_fit_default_optimizer(setup):
    dt, sf = setup
    results, errors = sf.fit(n_rounds=2,
                             optimizer=None,
                             metric=metric,
                             gL=[20*nS, 40*nS],
                             C=[0.5*nF, 1.5*nF])
    assert sf.simulator.neurons.iteration == 1
    attr_fit = ['optimizer', 'metric', 'best_params']
    for attr in attr_fit:
        assert hasattr(sf, attr)

    assert isinstance(sf.optimizer, NevergradOptimizer) #default optimizer
    assert isinstance(sf.simulator, Simulator)

    assert_equal(results, sf.best_params)
    assert_equal(errors, sf.best_error)


def test_spikefitter_fit_default_metric(setup):
    dt, sf = setup
    results, errors = sf.fit(n_rounds=2,
                             optimizer=n_opt,
                             metric=None,
                             gL=[20*nS, 40*nS],
                             C=[0.5*nF, 1.5*nF])
    assert sf.simulator.neurons.iteration == 1
    attr_fit = ['optimizer', 'metric', 'best_params']
    for attr in attr_fit:
        assert hasattr(sf, attr)
    assert len(sf.metric) == 1 and isinstance(sf.metric[0],
                                              GammaFactor) #default spike metric
    assert isinstance(sf.simulator, Simulator)

    assert_equal(results, sf.best_params)
    assert_equal(errors, sf.best_error)


def test_spikefitter_param_init(setup):
    dt, _ = setup
    SpikeFitter(model=model, input_var='I', dt=dt,
                input=inp_trace*amp, output=output,
                n_samples=2,
                threshold='v > -50*mV',
                reset='v = -70*mV',
                param_init={'v': -60*mV})

    with pytest.raises(ValueError):
        SpikeFitter(model=model, input_var='I', dt=dt,
                    input=inp_trace*amp, output=output,
                    n_samples=2,
                    threshold='v > -50*mV',
                    reset='v = -70*mV',
                    param_init={'Error': -60*mV})


def test_spikefitter_generate_spikes(setup):
    dt, sf = setup
    results, errors = sf.fit(n_rounds=2,
                             optimizer=n_opt,
                             metric=metric,
                             gL=[20*nS, 40*nS],
                             C=[0.5*nF, 1.5*nF])
    spikes = sf.generate_spikes()
    assert isinstance(spikes[0], np.ndarray)
    assert_equal(np.shape(spikes)[0], np.shape(inp_trace)[0])


def test_spikefitter_generate(setup):
    dt, sf = setup
    results, errors = sf.fit(n_rounds=2,
                             optimizer=n_opt,
                             metric=metric,
                             gL=[20*nS, 40*nS],
                             C=[0.5*nF, 1.5*nF])
    traces = sf.generate(params=None,
                         output_var='v',
                         param_init={'v': -70*mV})
    assert isinstance(traces, np.ndarray)
    assert_equal(np.shape(traces), np.shape(inp_trace))


def test_spikefitter_generate_multiple_variables(setup):
    dt, sf = setup
    results, errors = sf.fit(n_rounds=2,
                             optimizer=n_opt,
                             metric=metric,
                             gL=[20*nS, 40*nS],
                             C=[0.5*nF, 1.5*nF])
    recordings = sf.generate(params=None,
                             output_var=['v', 'spikes'],
                             param_init={'v': -70*mV})
    assert isinstance(recordings, dict)
    assert set(recordings.keys()) == {'v', 'spikes'}
    assert_equal(np.shape(recordings['v']), np.shape(inp_trace))
    assert_equal(np.shape(recordings['spikes'])[0], np.shape(inp_trace)[0])
