'''
Test the modelfitting module
'''
import pytest
import pandas as pd
import scipy.optimize

from numpy.testing import assert_equal, assert_almost_equal
from brian2 import (zeros, Equations, NeuronGroup, StateMonitor, TimedArray,
                    nS, mV, volt, ms, pA, pF, Quantity, set_device, get_device,
                    Network, have_same_dimensions, DimensionMismatchError)
from brian2.equations.equations import DIFFERENTIAL_EQUATION, SUBEXPRESSION
import brian2.numpy_ as np  # for unit-awareness
from brian2modelfitting import (NevergradOptimizer, TraceFitter, MSEMetric,
                                OnlineTraceFitter, Simulator, Metric,
                                Optimizer, GammaFactor, FeatureMetric)
from brian2.devices.device import reinit_devices, reset_device
from brian2modelfitting.fitter import get_param_dic


E = 40*mV
input_traces = zeros((10, 5))*volt
for i in range(5):
    input_traces[5:, i] = i*10*mV

output_traces = 10*nS*input_traces

model = Equations('''
    I = g*(v-E) : amp
    g : siemens (constant)
    ''')

strmodel = '''
    I = g*(v-E) : amp
    g : siemens (constant)
    '''

constant_model = Equations('''
    v = c + x: volt
    v2 = 2*c + x : volt
    c : volt (constant)''')

all_constant_model = Equations('''
    v = 10*mV + x: volt
    c : volt (constant)
    penalty_fixed = 10*mV**2: volt**2
    penalty_wrong_unit = 10*mV : volt''')

multiobjective_model = '''
dvar1/dt = (target1 - var1 + x)/(5*ms) : 1
dvar2/dt = (target2 - var2)/(5*ms) : volt
target1 : 1 (constant)
target2 : volt (constant)'''

n_opt = NevergradOptimizer()
metric = MSEMetric()


@pytest.fixture
def setup(request):
    dt = 0.01 * ms
    tf = TraceFitter(dt=dt,
                     model=model,
                     input={'v': input_traces},
                     output={'I': output_traces},
                     n_samples=30)

    def fin():
        reinit_devices()
    request.addfinalizer(fin)

    return dt, tf

@pytest.fixture
def setup_no_units(request):
    dt = 0.01 * ms
    tf = TraceFitter(dt=dt,
                     model=model,
                     input_var='v',
                     output_var='I',
                     input=input_traces,
                     output=output_traces,
                     n_samples=2,
                     use_units=False)

    def fin():
        reinit_devices()
    request.addfinalizer(fin)

    return dt, tf

@pytest.fixture
def setup_constant(request):
    dt = 0.1 * ms
    # Membrane potential is constant at 10mV for first 50 steps, then at 20mV
    out_trace = np.hstack([np.ones(50) * 10, np.ones(50) * 20])*mV
    tf = TraceFitter(dt=dt,
                     model=constant_model,
                     input_var='x',
                     output_var='v',
                     input=(np.zeros(100)*mV)[None, :],
                     output=out_trace[None, :],
                     n_samples=70)

    def fin():
        reinit_devices()
    request.addfinalizer(fin)

    return dt, tf


@pytest.fixture
def setup_constant_multiobjective(request):
    dt = 0.1 * ms
    # Membrane potential is constant at 10mV for first 50 steps, then at 20mV
    out_trace = np.hstack([np.ones(50) * 10, np.ones(50) * 20])*mV
    tf = TraceFitter(dt=dt,
                     model=constant_model,
                     input={'x': (np.zeros(100)*mV)[None, :]},
                     output={'v': out_trace[None, :],
                             'v2': 2*out_trace[None, :]},
                     n_samples=70)

    def fin():
        reinit_devices()
    request.addfinalizer(fin)

    return dt, tf


@pytest.fixture
def setup_all_constant(request):
    dt = 0.1 * ms
    # Membrane potential is constant at 10mV all the time and
    # does not depend on the parameter
    out_trace = np.ones((2, 100)) * 10 * mV
    tf = TraceFitter(dt=dt,
                     model=all_constant_model,
                     input_var='x',
                     output_var='v',
                     input=(np.zeros((2, 100)) * mV),
                     output=out_trace,
                     n_samples=70)

    def fin():
        reinit_devices()

    request.addfinalizer(fin)

    return dt, tf


@pytest.fixture
def setup_online(request):
    dt = 0.01 * ms

    otf = OnlineTraceFitter(dt=dt,
                            model=strmodel,
                            input_var='v',
                            output_var='I',
                            input=input_traces,
                            output=output_traces,
                            n_samples=10)

    def fin():
        reinit_devices()
    request.addfinalizer(fin)

    return dt, otf

@pytest.fixture
def setup_online_constant(request):
    dt = 0.1 * ms
    # Membrane potential is constant at 10mV for first 50 steps, then at 20mV
    out_trace = np.hstack([np.ones(50) * 10, np.ones(50) * 20])*mV
    otf = OnlineTraceFitter(dt=dt,
                            model=constant_model,
                            input_var='x',
                            output_var='v',
                            input=(np.zeros(100) * mV)[None, :],
                            output=out_trace[None, :],
                            n_samples=100)

    def fin():
        reinit_devices()
    request.addfinalizer(fin)

    return dt, otf


@pytest.fixture
def setup_standalone(request):
    # Workaround to avoid issues with Network instances still around
    Network.__instances__().clear()
    set_device('cpp_standalone', directory=None)
    dt = 0.01 * ms
    tf = TraceFitter(dt=dt,
                     model=model,
                     input_var='v',
                     output_var='I',
                     input=input_traces,
                     output=output_traces,
                     n_samples=2)

    def fin():
        reinit_devices()
        set_device('runtime')
    request.addfinalizer(fin)

    return dt, tf

@pytest.fixture
def setup_multiobjective(request):
    dt = 0.1 * ms
    out_trace1 = np.ones(1000)*5
    out_trace2 = np.ones(1000)*-7*mV
    tf = TraceFitter(dt=dt,
                     model=multiobjective_model,
                     input={'x': np.zeros_like(out_trace1)[None, :]},
                     output={'var1': out_trace1[None, :],
                             'var2': out_trace2[None, :]},
                     n_samples=70)

    def fin():
        reinit_devices()
    request.addfinalizer(fin)

    return dt, tf


@pytest.fixture
def setup_multiobjective_no_units(request):
    dt = 0.1 * ms

    out_trace1 = np.ones(1000)*5
    out_trace2 = np.ones(1000)*-7*mV
    tf = TraceFitter(dt=dt,
                     model=multiobjective_model,
                     input={'x': np.zeros_like(out_trace1)[None, :]},
                     output={'var1': out_trace1[None, :],
                             'var2': out_trace2[None, :]},
                     n_samples=70, use_units=False)

    def fin():
        reinit_devices()
    request.addfinalizer(fin)

    return dt, tf


def test_get_param_dic():
    d = get_param_dic([{'a': 1, 'b': 2}], ['a', 'b'], 4, 1)
    assert isinstance(d, dict)
    assert_equal(d, {'a': [1, 1, 1, 1], 'b': [2, 2, 2, 2]})

    d = get_param_dic([{'a': 1, 'b': 3}, {'a': 2, 'b': 4}], ['a', 'b'], 1, 2)
    assert_equal(d, {'a': [1, 2], 'b': [3, 4]})

    d = get_param_dic([{'a': 1, 'b': 3}, {'a': 2, 'b': 4}], ['a', 'b'], 2, 2)
    assert_equal(d, {'a': [1, 1, 2, 2], 'b': [3, 3, 4, 4]})


def test_tracefitter_init(setup):
    dt, tf = setup
    attr_fitter = ['dt', 'simulator', 'parameter_names', 'n_traces',
                   'duration', 'n_neurons', 'n_samples', 'method', 'threshold',
                   'reset', 'refractory', 'input', 'output', 'output_var',
                   'best_params', 'input_traces', 'model', 'optimizer',
                   'metric']
    for attr in attr_fitter:
        assert hasattr(tf, attr)

    assert tf.metric is None
    assert tf.optimizer is None
    assert tf.best_params is None

    attr_tracefitter = ['input_traces', 'model', 'simulator']
    for attr in attr_tracefitter:
        assert hasattr(tf, attr)

    assert isinstance(tf.input_traces, TimedArray)
    assert isinstance(tf.model, Equations)

    target_var = '{}_target'.format(tf.output_var[0])
    assert target_var in tf.model
    assert tf.model[target_var].dim is tf.output_dim[0]


def test_tracefitter_init_errors(setup):
    dt, _ = setup
    with pytest.raises(Exception):
        TraceFitter(dt=dt, model=model, input=input_traces,
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

    with pytest.raises(DimensionMismatchError):
        tf = TraceFitter(dt=dt,
                         model=model,
                         input_var='v',
                         output_var='I',
                         input=input_traces,
                         output=np.array(output_traces),  # no units
                         n_samples=2)


def test_tracefitter_fit_default_metric(setup):
    dt, tf = setup
    results, errors = tf.fit(n_rounds=2,
                             optimizer=n_opt,
                             metric=None,
                             g=[1*nS, 30*nS],
                             callback=None)
    assert tf.simulator.neurons.iteration == 1
    attr_fit = ['optimizer', 'metric', 'best_params']
    for attr in attr_fit:
        assert hasattr(tf, attr)
    assert len(tf.metric) == 1 and isinstance(tf.metric[0],
                                              MSEMetric) #default trace metric
    assert isinstance(tf.simulator, Simulator)

    assert_equal(results, tf.best_params)
    assert_equal(errors, tf.best_error)


from nevergrad.optimization import registry as nevergrad_registry
@pytest.mark.parametrize('method', sorted(nevergrad_registry.keys()))
def test_fitter_fit_methods(method):
    dt = 0.01 * ms
    model = Equations('''
        I = g*(v-E) : amp
        g : siemens (constant)
        E : volt (constant)
        ''')
    tf = TraceFitter(dt=dt,
                     model=model,
                     input_var='v',
                     output_var='I',
                     input=input_traces,
                     output=output_traces,
                     n_samples=30)
    # Skip a few methods that seem to hang due to multi-threading deadlocks (?) or simply take very long
    skip = ['BO', 'ParaPortfolio', 'BAR', 'MultiBFGS', 'MultiCobyla', 'MultiSQP', 'NgIohRW', 'F3SQPCMA']
    if any(s in method for s in skip):
        pytest.skip(f'Skipping method {method}')

    try:
        optimizer = NevergradOptimizer(method)  # set high budget to avoid problems for some methods
        # Just make sure that it can run at all
        tf.fit(n_rounds=2,
               optimizer=optimizer,
               metric=metric,
               g=[1*nS, 30*nS],
               E=[-60*mV, -20*mV],
               callback=None)
    except (ImportError, RuntimeError) as ex:
        # Skip methods that need additional packages
        if isinstance(ex, ImportError):
            pytest.skip(f"Could not test method '{method}', raised '{str(ex)}'.")
        # Import errors might be wrapped and propagated as a RuntimeError
        elif isinstance(getattr(ex, '__cause__', None), ImportError):
            pytest.skip(f"Could not test method '{method}', raised '{str(ex.__cause__)}'.")
        else:
            raise ex


def test_fitter_fit(setup):
    dt, tf = setup
    results, errors = tf.fit(n_rounds=2,
                             optimizer=n_opt,
                             metric=metric,
                             g=[1*nS, 30*nS],
                             callback=None)

    attr_fit = ['optimizer', 'metric', 'best_params']
    for attr in attr_fit:
        assert hasattr(tf, attr)
    assert tf.simulator.neurons.iteration == 1

    assert len(tf.metric) == 1 and isinstance(tf.metric[0], Metric)
    assert isinstance(tf.optimizer, Optimizer)
    assert isinstance(tf.simulator, Simulator)

    assert isinstance(results, dict)
    assert all(isinstance(v, Quantity) for v in results.values())
    assert isinstance(errors, Quantity)
    assert 'g' in results.keys()

    assert_equal(results, tf.best_params)
    assert_equal(errors, tf.best_error)


def test_fitter_fit_no_units(setup_no_units):
    dt, tf = setup_no_units
    results, errors = tf.fit(n_rounds=2,
                             optimizer=n_opt,
                             metric=metric,
                             g=[1*nS, 30*nS],
                             callback=None)

    attr_fit = ['optimizer', 'metric', 'best_params']
    for attr in attr_fit:
        assert hasattr(tf, attr)

    assert len(tf.metric) == 1 and isinstance(tf.metric[0], Metric)
    assert isinstance(tf.optimizer, Optimizer)
    assert isinstance(tf.simulator, Simulator)

    assert isinstance(results, dict)
    assert all(isinstance(v, float) for v in results.values())
    assert isinstance(errors, float)
    assert 'g' in results.keys()

    assert_equal(results, tf.best_params)
    assert_equal(errors, tf.best_error)


def test_fitter_fit_default_optimizer(setup):
    dt, tf = setup
    results, errors = tf.fit(n_rounds=2,
                             optimizer=None,
                             metric=metric,
                             g=[1*nS, 30*nS],
                             callback=None)
    assert tf.simulator.neurons.iteration == 1
    attr_fit = ['optimizer', 'metric', 'best_params']
    for attr in attr_fit:
        assert hasattr(tf, attr)

    assert isinstance(tf.optimizer, NevergradOptimizer) #default optimizer
    assert isinstance(tf.simulator, Simulator)

    assert_equal(results, tf.best_params)
    assert_equal(errors, tf.best_error)


def test_fitter_fit_callback(setup):
    dt, tf = setup

    calls = []
    def our_callback(params, errors, best_params, best_error, index,
                     additional_info):
        calls.append(index)
        assert all(isinstance(p, dict) for p in params)
        assert isinstance(errors, np.ndarray)
        assert isinstance(best_params, dict)
        assert isinstance(best_error, Quantity)
        assert isinstance(index, int)
        assert isinstance(additional_info, dict)
    results, errors = tf.fit(n_rounds=2,
                             optimizer=n_opt,
                             metric=metric,
                             g=[1*nS, 30*nS],
                             callback=our_callback)
    assert len(calls) == 2

    # Stop a fit via the callback

    calls = []
    def our_callback(params, errors, best_params, best_error, index,
                     additional_info):
        calls.append(index)
        assert all(isinstance(p, dict) for p in params)
        assert isinstance(errors, np.ndarray)
        assert isinstance(best_params, dict)
        assert isinstance(best_error, Quantity)
        assert isinstance(index, int)
        assert isinstance(additional_info, dict)
        return True  # stop

    results, errors = tf.fit(n_rounds=2,
                             optimizer=n_opt,
                             metric=metric,
                             g=[1*nS, 30*nS],
                             callback=our_callback)
    assert len(calls) == 1


def test_fitter_fit_errors(setup):
    dt, tf = setup
    class NaiiveOptimizer:
        def __init__(self):
            self.best = []
    opt = NaiiveOptimizer()
    with pytest.raises(TypeError):
        tf.fit(n_rounds=2,
               optimizer=opt, #testing a Non-Optimizer child
               metric=metric,
               g=[1*nS, 30*nS])

    with pytest.raises(TypeError):
        tf.fit(n_rounds=2,
               optimizer=n_opt,
               metric=GammaFactor(3*ms, 60*ms),  # spike metric
               g=[1*nS, 30*nS])


def test_fitter_fit_tstart(setup_constant):
    dt, tf = setup_constant

    # Ignore the first 50 steps at 10mV
    params, result = tf.fit(n_rounds=10, optimizer=n_opt,
                            metric=MSEMetric(t_start=50*dt),
                            c=[0 * mV, 30 * mV])
    # Fit should be close to 20mV
    assert np.abs(params['c'] - 20*mV) < 1*mV


def test_fitter_fit_tsteps(setup_constant):
    dt, tf = setup_constant

    with pytest.raises(ValueError):
        # Incorrect weight size
        tf.fit(n_rounds=10, optimizer=n_opt,
               metric=MSEMetric(t_weights=np.ones(101)),
               c=[0 * mV, 30 * mV])

    # Ignore the first 50 steps at 10mV
    weights = np.ones(100)
    weights[:50] = 0
    params, result = tf.fit(n_rounds=10, optimizer=n_opt,
                            metric=MSEMetric(t_weights=weights),
                            c=[0 * mV, 30 * mV])
    # Fit should be close to 20mV
    assert np.abs(params['c'] - 20*mV) < 1*mV


def test_fitter_fit_penalty(setup_all_constant):
    dt, tf = setup_all_constant

    params, result = tf.fit(n_rounds=2, optimizer=n_opt,
                            metric=metric,
                            c=[19.9*mV, 20.1*mV],  # real error is minimal
                            callback=None,
                            penalty='penalty_fixed')
    assert abs(float(result - 10*mV**2)) < 1e-6

    with pytest.raises(DimensionMismatchError):
        params, result = tf.fit(n_rounds=2, optimizer=n_opt,
                                c=[19.9 * mV, 20.1 * mV],  # real error is minimal
                                metric=metric,
                                callback=None,
                                penalty='penalty_wrong_unit')


def test_fitter_refine(setup):
    dt, tf = setup
    results, errors = tf.fit(n_rounds=2,
                             optimizer=n_opt,
                             metric=metric,
                             g=[1*nS, 30*nS],
                             callback=None)
    # Run refine after running fit
    params, result = tf.refine()
    assert isinstance(params, dict)
    assert isinstance(result, scipy.optimize.OptimizeResult)


def test_fitter_refine_standalone(setup_standalone):
    dt, tf = setup_standalone
    results, errors = tf.fit(n_rounds=2,
                             optimizer=n_opt,
                             metric=metric,
                             g=[1*nS, 30*nS],
                             callback=None)
    # Run refine after running fit
    params, result = tf.refine()
    assert isinstance(params, dict)
    assert isinstance(result, scipy.optimize.OptimizeResult)


def test_fitter_refine_direct(setup):
    dt, tf = setup
    # Run refine without running fit before
    params, result = tf.refine({'g': 5*nS}, g=[1*nS, 30*nS])
    error = result.cost
    assert isinstance(params, dict)
    assert isinstance(result, scipy.optimize.OptimizeResult)
    # The algorithm is deterministic and should therefore give the same result
    # for the second run
    params, result = tf.refine({'g': 5 * nS}, g=[1 * nS, 30 * nS],
                               metric=MSEMetric(normalization=1/2))
    assert result.cost == 4 * error


def test_fitter_refine_calc_gradient():
    tau = 5*ms
    Cm = 100*pF
    inputs = (np.ones((100, 2))*np.array([1, 2])).T*100*pA
    # The model results can be approximated with exponentials
    def exp_fit(x, a, b):
        return a * np.exp(x / b) -70 - a * np.exp(0)
    outputs = np.vstack([exp_fit(np.arange(100), 1.2836869755582263, 51.41761887704586),
                         exp_fit(np.arange(100), 2.567374463239943,  51.417624003833076)])*volt

    model = '''
    dv/dt = (g_L * (E_L - v) + I_e)/Cm : volt
    dI_e/dt = -I/tau : amp
    g_L : siemens (constant)
    E_L : volt (constant)
    '''
    tf = TraceFitter(dt=0.1*ms,
                     model=model,
                     input_var='I',
                     output_var='v',
                     input=inputs,
                     output=outputs,
                     n_samples=2,
                     param_init={'v': 'E_L'})
    params, result = tf.refine({'g_L': 5 * nS, 'E_L': -65*mV},
                               g_L=[1 * nS, 30 * nS],
                               E_L=[-80*mV, -50*mV],
                               calc_gradient=True)
    assert 'S_v_g_L' in tf.simulator.neurons.equations
    assert 'S_I_e_g_L' in tf.simulator.neurons.equations
    assert tf.simulator.neurons.equations['S_v_g_L'].type == DIFFERENTIAL_EQUATION
    assert tf.simulator.neurons.equations['S_I_e_g_L'].type == SUBEXPRESSION  # optimized away
    params, result = tf.refine({'g_L': 5 * nS, 'E_L': -65*mV},
                               g_L=[1 * nS, 30 * nS],
                               E_L=[-80*mV, -50*mV],
                               calc_gradient=True, optimize=False)
    assert 'S_v_g_L' in tf.simulator.neurons.equations
    assert 'S_I_e_g_L' in tf.simulator.neurons.equations
    assert tf.simulator.neurons.equations['S_v_g_L'].type == DIFFERENTIAL_EQUATION
    assert tf.simulator.neurons.equations['S_I_e_g_L'].type == DIFFERENTIAL_EQUATION


def test_fitter_refine_reuse_tstart(setup_constant):
    dt, tf = setup_constant

    # Ignore the first 50 steps at 10mV but do not actually fit (0 rounds)
    params, result = tf.fit(n_rounds=0, optimizer=n_opt,
                            metric=MSEMetric(t_start=50*dt),
                            c=[0 * mV, 30 * mV])
    # t_start should be reused
    params, result = tf.refine({'c': 5 * mV}, c=[0 * mV, 30 * mV])

    # Fit should be close to 20mV
    assert np.abs(params['c'] - 20 * mV) < 1 * mV


def test_fitter_refine_reuse_tstart_multiobjective(setup_constant_multiobjective):
    dt, tf = setup_constant_multiobjective

    # Ignore the first 50 steps at 10mV only for v2, but do not actually fit (0 rounds)
    params, result = tf.fit(n_rounds=0, optimizer=n_opt,
                            metric={'v': MSEMetric(t_start=0*dt),
                                    'v2': MSEMetric(t_start=50*dt)},
                            c=[0 * mV, 30 * mV])
    # t_start should be reused
    params, result = tf.refine({'c': 5 * mV}, c=[0 * mV, 30 * mV])

    # Fit should be close to 17.5 mV (15mV for v, 20mV for v2)
    assert np.abs(params['c'] - 17.5 * mV) < 1 * mV


def test_fitter_refine_reuse_tsteps(setup_constant):
    dt, tf = setup_constant
    weights = np.ones(100)
    weights[:50] = 0
    # Ignore the first 50 steps at 10mV but do not actually fit (0 rounds)
    params, result = tf.fit(n_rounds=0, optimizer=n_opt,
                            metric=MSEMetric(t_weights=weights),
                            c=[0 * mV, 30 * mV])
    # t_start should be reused
    params, result = tf.refine({'c': 5 * mV}, c=[0 * mV, 30 * mV])

    # Fit should be close to 20mV
    assert np.abs(params['c'] - 20 * mV) < 1 * mV


def test_fitter_refine_reuse_tsteps_multiobjective(setup_constant_multiobjective):
    dt, tf = setup_constant_multiobjective
    weights = np.ones(100)
    weights[:50] = 0

    # Ignore the first 50 steps at 10mV only for v2, but do not actually fit (0 rounds)
    params, result = tf.fit(n_rounds=0, optimizer=n_opt,
                            metric={'v': MSEMetric(t_start=0*dt),
                                    'v2': MSEMetric(t_start=50*dt)},
                            c=[0 * mV, 30 * mV])
    # t_start should be reused
    params, result = tf.refine({'c': 5 * mV}, c=[0 * mV, 30 * mV])

    # Fit should be close to 17.5 mV (15mV for v, 20mV for v2)
    assert np.abs(params['c'] - 17.5 * mV) < 1 * mV


def test_fitter_refine_errors(setup):
    dt, tf = setup
    with pytest.raises(TypeError):
        # Missing start parameter
        tf.refine(g=[1*nS, 30*nS])

    with pytest.raises(TypeError):
        # Missing bounds
        tf.refine({'g': 5*nS})

    with pytest.raises(TypeError):
        # Wrong metric
        stim_times = [[5 * ms, 10 * ms]]
        feat_list = ['voltage_base']
        weights = {'voltage_base': 1}
        metric = FeatureMetric(stim_times, feat_list, weights=weights)
        tf.refine({'g': 5*nS}, g=[1*nS, 30*nS], metric=metric)


def test_fitter_callback(setup, caplog):
    dt, tf = setup

    calls = []
    def our_callback(params, errors, best_params, best_error, index,
                     additional_info):
        calls.append(index)
        assert isinstance(params, dict)
        assert isinstance(errors, list)
        assert isinstance(best_params, dict)
        assert isinstance(best_error, Quantity)
        assert isinstance(index, int)
        assert isinstance(additional_info, dict)

    tf.refine({'g': 5 * nS}, g=[1 * nS, 30 * nS], callback=our_callback)
    assert len(calls)


def test_fit_restart(setup):
    dt, tf = setup
    results, errors = tf.fit(n_rounds=2,
                             optimizer=n_opt,
                             metric=metric,
                             g=[1*nS, 30*nS])
    assert tf.simulator.neurons.iteration == 1

    results, errors = tf.fit(n_rounds=2,
                             optimizer=n_opt,
                             metric=metric,
                             g=[1*nS, 30*nS])
    assert tf.simulator.neurons.iteration == 3

    results, errors = tf.fit(n_rounds=2,
                             restart=True,
                             optimizer=n_opt,
                             metric=metric,
                             g=[1*nS, 30*nS])
    assert tf.simulator.neurons.iteration == 1


def test_fit_set_start_iteration(setup):
    dt, tf = setup
    results, errors = tf.fit(n_rounds=2,
                             optimizer=n_opt,
                             metric=metric,
                             g=[1 * nS, 30 * nS],
                             start_iteration=17)
    assert tf.simulator.neurons.iteration == 18

    results, errors = tf.fit(n_rounds=2,
                             optimizer=n_opt,
                             metric=metric,
                             g=[1 * nS, 30 * nS])
    assert tf.simulator.neurons.iteration == 20

    results, errors = tf.fit(n_rounds=2,
                             restart=True,
                             optimizer=n_opt,
                             metric=metric,
                             g=[1 * nS, 30 * nS],
                             start_iteration=5)
    assert tf.simulator.neurons.iteration == 6


def test_fit_continue_with_generate(setup):
    dt, tf = setup
    results, error = tf.fit(n_rounds=2,
                            optimizer=n_opt,
                            metric=metric,
                            g=[1*nS, 30*nS])

    fits = tf.generate_traces()

    results, error = tf.fit(n_rounds=2,
                            optimizer=n_opt,
                            metric=metric,
                            g=[1*nS, 30*nS])

    fits = tf.generate_traces()


def test_fit_restart_errors(setup):
    dt, tf = setup
    results, errors = tf.fit(n_rounds=2,
                             optimizer=n_opt,
                             metric=metric,
                             g=[1*nS, 30*nS],
                             restart=False,)

    n_opt2 = NevergradOptimizer('PSO')
    with pytest.raises(Exception):
        tf.fit(n_rounds=2,
               optimizer=n_opt2,
               metric=metric,
               g=[1*nS, 30*nS],
               restart=False,)

    metric2 = GammaFactor(40*ms, 40*ms)
    with pytest.raises(Exception):
        tf.fit(n_rounds=2,
               optimizer=n_opt,
               metric=metric2,
               g=[1*nS, 30*nS],
               restart=False,)


def test_fit_restart_change(setup):
    dt, tf = setup
    results, errors = tf.fit(n_rounds=2,
                             optimizer=n_opt,
                             metric=metric,
                             g=[1*nS, 30*nS],
                             restart=False)

    n_opt2 = NevergradOptimizer('PSO')
    results2, errors2 = tf.fit(n_rounds=2,
                               optimizer=n_opt2,
                               metric=metric,
                               g=[1*nS, 30*nS],
                               restart=True)


def test_fitter_generate_traces(setup):
    dt, tf = setup
    results, errors = tf.fit(n_rounds=2,
                             optimizer=n_opt,
                             metric=metric,
                             g=[1*nS, 30*nS],
                             restart=False,)
    traces = tf.generate_traces()
    assert isinstance(traces, np.ndarray)
    assert_equal(np.shape(traces), np.shape(output_traces))


def test_fitter_generate_traces_multiple_vars(setup):
    dt, tf = setup
    results, errors = tf.fit(n_rounds=2,
                             optimizer=n_opt,
                             metric=metric,
                             g=[1*nS, 30*nS],
                             restart=False,)
    traces = tf.generate(output_var=['I', 'g'])
    assert isinstance(traces, dict)
    assert set(traces.keys()) == {'I', 'g'}
    assert_equal(np.shape(traces['I']), np.shape(output_traces))
    assert_equal(np.shape(traces['g']), np.shape(output_traces))


def test_fitter_generate_traces_standalone(setup_standalone):
    dt, tf = setup_standalone
    results, errors = tf.fit(n_rounds=2,
                             optimizer=n_opt,
                             metric=metric,
                             g=[1*nS, 30*nS],
                             restart=False,)

    traces = tf.generate_traces()
    assert isinstance(traces, np.ndarray)
    assert_equal(np.shape(traces), np.shape(output_traces))


def test_fitter_results(setup):
    dt, tf = setup
    best_params, errors = tf.fit(n_rounds=2,
                              optimizer=n_opt,
                              metric=metric,
                              g=[1*nS, 30*nS],
                              restart=False)

    params_list = tf.results(format='list')
    assert isinstance(params_list, list)
    assert isinstance(params_list[0], dict)
    assert isinstance(params_list[0]['g'], Quantity)
    assert 'g' in params_list[0].keys()
    assert 'error' in params_list[0].keys()
    assert_equal(np.shape(params_list), (tf.n_samples*2,))
    assert_equal(len(params_list[0]), 2)
    assert have_same_dimensions(params_list[0]['g'].dim, nS)

    params_dic = tf.results(format='dict')
    assert isinstance(params_dic, dict)
    assert 'g' in params_dic.keys()
    assert 'error' in params_dic.keys()
    assert isinstance(params_dic['g'], Quantity)
    assert_equal(len(params_dic), 2)
    assert_equal(np.shape(params_dic['g']), (tf.n_samples * 2,))
    assert_equal(np.shape(params_dic['error']), (tf.n_samples * 2,))

    # Should raise a warning because dataframe cannot have units
    # Skip this check for now since Brian's logger mechanism has changed recently
    # assert len(caplog.records) == 0
    params_df = tf.results(format='dataframe')
    # assert len(caplog.records) == 1
    assert isinstance(params_df, pd.DataFrame)
    assert_equal(params_df.shape, (tf.n_samples * 2, 2))
    assert 'g' in params_df.keys()
    assert 'error' in params_df.keys()


def test_fitter_results_no_units(setup_no_units, caplog):
    dt, tf = setup_no_units
    tf.fit(n_rounds=2,
           optimizer=n_opt,
           metric=metric,
           g=[1*nS, 30*nS],
           restart=False)

    params_list = tf.results(format='list')
    assert isinstance(params_list, list)
    assert isinstance(params_list[0], dict)
    assert isinstance(params_list[0]['g'], float)
    assert 'g' in params_list[0].keys()
    assert 'error' in params_list[0].keys()
    assert_equal(np.shape(params_list), (tf.n_samples * 2,))
    assert_equal(len(params_list[0]), 2)

    params_dic = tf.results(format='dict')
    assert isinstance(params_dic, dict)
    assert 'g' in params_dic.keys()
    assert 'error' in params_dic.keys()
    assert isinstance(params_dic['g'], np.ndarray)
    assert_equal(len(params_dic), 2)
    assert_equal(np.shape(params_dic['g']), (tf.n_samples * 2,))
    assert_equal(np.shape(params_dic['error']), (tf.n_samples * 2,))

    params_df = tf.results(format='dataframe')
    assert isinstance(params_df, pd.DataFrame)
    assert_equal(params_df.shape, (tf.n_samples * 2, 2))
    assert 'g' in params_df.keys()
    assert 'error' in params_df.keys()


# OnlineTraceFitter
def test_onlinetracefitter_init(setup_online):
    dt, otf = setup_online
    attr_fitter = ['dt', 'simulator', 'parameter_names', 'n_traces',
                   'duration', 'n_neurons', 'n_samples', 'method', 'threshold',
                   'reset', 'refractory', 'input', 'output', 'output_var',
                   'best_params', 'input_traces', 'model', 'optimizer',
                   'metric', 't_start']
    for attr in attr_fitter:
        assert hasattr(otf, attr)

    assert otf.metric is None
    assert otf.optimizer is None
    assert otf.best_params is None
    assert_equal(otf.t_start, 0*ms)

    attr_tracefitter = ['input_traces', 'model', 'simulator']
    for attr in attr_tracefitter:
        assert hasattr(otf, attr)

    assert isinstance(otf.input_traces, TimedArray)
    assert isinstance(otf.model, Equations)


def test_onlinetracefitter_init_errors(setup_online):
    dt, _ = setup_online
    with pytest.raises(Exception):
        OnlineTraceFitter(dt=dt, model=model, input=input_traces,
                          n_samples=10,
                          output=output_traces,
                          output_var='I',
                          input_var='Exception')

    with pytest.raises(Exception):
        OnlineTraceFitter(dt=0.1*ms, model=model, input=input_traces,
                          n_samples=10,
                          output=output_traces,
                          input_var='v',
                          output_var='Exception')

    with pytest.raises(Exception):
        OnlineTraceFitter(dt=0.1*ms, model=model, input=input_traces,
                          n_samples=10,
                          output=[1],
                          input_var='v',
                          output_var='I',)


def test_onlinetracefitter_fit(setup_online):
    dt, otf = setup_online
    results, errors = otf.fit(n_rounds=2,
                              optimizer=n_opt,
                              g=[1*nS, 30*nS],
                              restart=False,)
    assert otf.simulator.neurons.iteration == 1
    attr_fit = ['optimizer', 'metric', 'best_params']
    for attr in attr_fit:
        assert hasattr(otf, attr)

    assert len(otf.metric) == 1 and isinstance(otf.metric[0], MSEMetric)
    assert isinstance(otf.optimizer, Optimizer)

    assert isinstance(results, dict)
    assert isinstance(errors, Quantity)
    assert 'g' in results.keys()

    assert_equal(results, otf.best_params)


def test_onlinetracefitter_generate_traces(setup_online):
    dt, otf = setup_online
    results, errors = otf.fit(n_rounds=2,
                              optimizer=n_opt,
                              g=[1 * nS, 30 * nS],
                              restart=False, )
    traces = otf.generate_traces()
    assert isinstance(traces, np.ndarray)
    assert_equal(np.shape(traces), np.shape(output_traces))


def test_onlinetracefitter_fit_tstart():
    dt = 0.1 * ms
    # Membrane potential is constant at 10mV for first 50 steps, then at 20mV
    out_trace = np.hstack([np.ones(50) * 10, np.ones(50) * 20]) * mV
    otf = OnlineTraceFitter(dt=dt,
                            model=constant_model,
                            input_var='x',
                            output_var='v',
                            input=(np.zeros(100) * mV)[None, :],
                            output=out_trace[None, :],
                            n_samples=100,
                            t_start=50*dt)

    # Ignore the first 50 steps at 10mV
    params, result = otf.fit(n_rounds=10, optimizer=n_opt,
                             c=[0 * mV, 30 * mV])
    # Fit should be close to 20mV
    assert np.abs(params['c'] - 20*mV) < 1*mV


def test_multiobjective_basic(setup_multiobjective):
    dt, tf = setup_multiobjective
    result, error = tf.fit(n_rounds=20,
                       metric={'var1': MSEMetric(t_start=50*ms),
                               'var2': MSEMetric(t_start=50*ms, normalization=1*mV)},
                       optimizer=n_opt,
                       target1=(-10, 10),
                       target2=(-10*mV, 10*mV),
                       callback=None)
    # Since the problem is simple, we should be reasonably close to the correct solution
    assert abs(result['target1'] - 5) < 0.1
    assert abs(result['target2'] + 7*mV) < 0.1*mV

    # The variables relax exponentially to the target values
    t_values = np.arange(1000)*0.1*ms
    simulated_var1 = result['target1'] * (1 - np.exp(-t_values/(5*ms)))
    simulated_var2 = result['target2'] * (1 - np.exp(-t_values/(5*ms)))
    error_var1 = np.mean((simulated_var1[t_values>=50*ms] - 5)**2)
    normed_error_var1 = error_var1
    error_var2 = np.mean((simulated_var2[t_values>=50*ms] + 7*mV) ** 2)
    normed_error_var2 = np.mean(((simulated_var2[t_values>=50 * ms] + 7 * mV)/(1*mV)) ** 2)
    # Check that the errors returned by the fitter are correct
    assert_almost_equal(error, tf.best_error)
    assert_almost_equal(error, normed_error_var1 + normed_error_var2)
    assert set(tf.best_objective_errors.keys()) == {'var1', 'var2'}
    assert have_same_dimensions(tf.best_objective_errors['var1'], error_var1)
    assert_almost_equal(float(tf.best_objective_errors['var1']), float(error_var1))
    assert have_same_dimensions(tf.best_objective_errors['var2'], error_var2)
    assert_almost_equal(float(tf.best_objective_errors['var2']), float(error_var2))
    assert set(tf.best_objective_errors_normalized.keys()) == {'var1', 'var2'}

    assert have_same_dimensions(tf.best_objective_errors_normalized['var1'], normed_error_var1)
    assert_almost_equal(float(tf.best_objective_errors_normalized['var1']), float(normed_error_var1))
    assert have_same_dimensions(tf.best_objective_errors_normalized['var2'], normed_error_var2)
    assert_almost_equal(float(tf.best_objective_errors_normalized['var2']), float(normed_error_var2))

    # Check that the objective errors are included in the results
    list_results = tf.results(format='list')
    assert all('objective_errors' for r in list_results)
    assert all('objective_errors_normalized' for r in list_results)
    assert all('var1' in r['objective_errors'] and 'var2' in r['objective_errors']
               for r in list_results)
    assert all('var1' in r['objective_errors_normalized'] and
               'var2' in r['objective_errors_normalized']
               for r in list_results)
    assert all((have_same_dimensions(r['objective_errors']['var1'], 1) and
                have_same_dimensions(r['objective_errors']['var2'], mV**2))
               for r in list_results)
    assert all((have_same_dimensions(r['objective_errors_normalized']['var1'], 1) and
                have_same_dimensions(r['objective_errors_normalized']['var2'], 1))
               for r in list_results)

    dict_results = tf.results(format='dict')
    assert ('objective_errors' in dict_results and
            isinstance(dict_results['objective_errors'], dict))
    assert ('objective_errors_normalized' in dict_results and
            isinstance(dict_results['objective_errors_normalized'], dict))
    assert ('var1' in dict_results['objective_errors'] and
            have_same_dimensions(dict_results['objective_errors']['var1'], 1))
    assert ('var2' in dict_results['objective_errors'] and
            have_same_dimensions(dict_results['objective_errors']['var2'], mV**2))
    assert ('var1' in dict_results['objective_errors_normalized'] and
            have_same_dimensions(dict_results['objective_errors_normalized']['var1'], 1))
    assert ('var2' in dict_results['objective_errors'] and
            have_same_dimensions(dict_results['objective_errors_normalized']['var2'], 1))

    try:
        import pandas
        pandas_results = tf.results(format='dataframe')
        assert 'error_var1' in pandas_results.columns
        assert 'error_var2' in pandas_results.columns
        assert 'normalized_error_var1' in pandas_results.columns
        assert 'normalized_error_var2' in pandas_results.columns
    except ImportError:
        pass

    # Fits should still be could after refinement
    refined, _ = tf.refine(calc_gradient=True, callback=None)
    assert abs(refined['target1'] - 5) < 0.1
    assert abs(refined['target2'] + 7*mV) < 0.1*mV


def test_multiobjective_no_units(setup_multiobjective_no_units):
    dt, tf = setup_multiobjective_no_units
    result, error = tf.fit(n_rounds=20,
                       metric={'var1': MSEMetric(t_start=50*ms),
                               'var2': MSEMetric(t_start=50*ms, normalization=0.001)},
                       optimizer=n_opt,
                       target1=(-10, 10),
                       target2=(-10*mV, 10*mV),
                       callback=None)
    # Since the problem is simple, we should be reasonably close to the correct solution
    assert abs(result['target1'] - 5) < 0.1
    assert abs(result['target2'] + float(7*mV)) < float(0.1*mV)

    # The variables relax exponentially to the target values
    t_values = np.arange(1000)*0.1*ms
    simulated_var1 = result['target1'] * (1 - np.exp(-t_values/(5*ms)))
    simulated_var2 = result['target2'] * (1 - np.exp(-t_values/(5*ms)))
    error_var1 = np.mean((simulated_var1[t_values>=50*ms] - 5)**2)
    normed_error_var1 = error_var1
    error_var2 = np.mean((simulated_var2[t_values>=50*ms] + float(7*mV)) ** 2)
    normed_error_var2 = np.mean(((simulated_var2[t_values>=50 * ms] + float(7*mV))/float(1*mV)) ** 2)
    # Check that the errors returned by the fitter are correct
    assert_almost_equal(error, tf.best_error)
    assert_almost_equal(error, normed_error_var1 + normed_error_var2)
    assert set(tf.best_objective_errors.keys()) == {'var1', 'var2'}
    assert have_same_dimensions(tf.best_objective_errors['var1'], error_var1)
    assert_almost_equal(float(tf.best_objective_errors['var1']), float(error_var1))
    assert have_same_dimensions(tf.best_objective_errors['var2'], error_var2)
    assert_almost_equal(float(tf.best_objective_errors['var2']), float(error_var2))
    assert set(tf.best_objective_errors_normalized.keys()) == {'var1', 'var2'}

    assert have_same_dimensions(tf.best_objective_errors_normalized['var1'], normed_error_var1)
    assert_almost_equal(float(tf.best_objective_errors_normalized['var1']), float(normed_error_var1))
    assert have_same_dimensions(tf.best_objective_errors_normalized['var2'], normed_error_var2)
    assert_almost_equal(float(tf.best_objective_errors_normalized['var2']), float(normed_error_var2))

    # Check that the objective errors are included in the results
    list_results = tf.results(format='list')
    assert all('objective_errors' for r in list_results)
    assert all('objective_errors_normalized' for r in list_results)
    assert all('var1' in r['objective_errors'] and 'var2' in r['objective_errors']
               for r in list_results)
    assert all('var1' in r['objective_errors_normalized'] and
               'var2' in r['objective_errors_normalized']
               for r in list_results)
    assert all((have_same_dimensions(r['objective_errors']['var1'], 1) and
                have_same_dimensions(r['objective_errors']['var2'], 1))
               for r in list_results)
    assert all((have_same_dimensions(r['objective_errors_normalized']['var1'], 1) and
                have_same_dimensions(r['objective_errors_normalized']['var2'], 1))
               for r in list_results)

    dict_results = tf.results(format='dict')
    assert ('objective_errors' in dict_results and
            isinstance(dict_results['objective_errors'], dict))
    assert ('objective_errors_normalized' in dict_results and
            isinstance(dict_results['objective_errors_normalized'], dict))
    assert ('var1' in dict_results['objective_errors'] and
            have_same_dimensions(dict_results['objective_errors']['var1'], 1))
    assert ('var2' in dict_results['objective_errors'] and
            have_same_dimensions(dict_results['objective_errors']['var2'], 1))
    assert ('var1' in dict_results['objective_errors_normalized'] and
            have_same_dimensions(dict_results['objective_errors_normalized']['var1'], 1))
    assert ('var2' in dict_results['objective_errors'] and
            have_same_dimensions(dict_results['objective_errors_normalized']['var2'], 1))

    try:
        import pandas
        pandas_results = tf.results(format='dataframe')
        assert 'error_var1' in pandas_results.columns
        assert 'error_var2' in pandas_results.columns
        assert 'normalized_error_var1' in pandas_results.columns
        assert 'normalized_error_var2' in pandas_results.columns
    except ImportError:
        pass

    # Fits should still be could after refinement
    refined, _ = tf.refine(calc_gradient=True, callback=None)
    assert abs(refined['target1'] - 5) < 0.1
    assert abs(refined['target2'] + float(7*mV)) < float(0.1*mV)