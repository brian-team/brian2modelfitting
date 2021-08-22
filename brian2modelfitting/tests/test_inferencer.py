from brian2 import nS, siemens, mV, volt, ms, TimedArray, Equations
from brian2.devices.device import reinit_devices
from brian2modelfitting.inferencer import (Inferencer,
                                           get_param_dict,
                                           calc_prior)
import numpy as np
from numpy.testing import assert_equal, assert_almost_equal
import pytest


def test_get_param_dict():
    param_names = ['a', 'b']
    n_samples = 3
    n_traces = 3
    n_neurons = n_samples * n_traces

    # check theta
    theta = np.array([[1, 2],
                      [3, 4],
                      [5, 6]])
    _theta = np.array([[1, 2],
                       [1, 2],
                       [1, 2],
                       [3, 4],
                       [3, 4],
                       [3, 4],
                       [5, 6],
                       [5, 6],
                       [5, 6]])
    assert_equal(_theta, np.repeat(theta, repeats=n_traces, axis=0))

    # check dictionary type and values
    theta = np.repeat(theta, repeats=n_traces, axis=0)
    d_param = {'a': np.array([1., 1., 1., 3., 3., 3., 5., 5., 5.]),
               'b': np.array([2., 2., 2., 4., 4., 4., 6., 6., 6.])}
    d_param_ret = get_param_dict(param_names=param_names,
                                 param_values=_theta,
                                 n_values=n_neurons)
    assert isinstance(d_param_ret, dict)
    assert_equal(d_param, d_param_ret)


def test_calc_prior():
    n_samples = 10
    param_names = ['a', 'b']
    a_lower_bound = 1
    a_upper_bound = 10
    a_mean = (a_lower_bound + a_upper_bound) / 2.
    b_lower_bound = 0
    b_upper_bound = 1
    b_mean = (b_lower_bound + b_upper_bound) / 2.
    prior_mean = np.array([a_mean, b_mean], dtype=np.float32)
    a_std = (a_upper_bound - a_lower_bound) / np.sqrt(12)
    b_std = (b_upper_bound - b_lower_bound) / np.sqrt(12)
    prior_std = np.array([a_std, b_std], dtype=np.float32)

    prior_sbi = calc_prior(param_names=param_names,
                           a=[a_lower_bound*volt, a_upper_bound*volt],
                           b=[b_lower_bound*volt, b_upper_bound*volt])

    # check mean value
    prior_sbi_mean = prior_sbi.mean.numpy()
    assert_equal(prior_mean, prior_sbi_mean)

    # check standard deviation
    prior_sbi_std = prior_sbi.stddev.numpy()
    assert_equal(prior_std, prior_sbi_std)

    # check the type and shape of the prior
    prior_sbi_sampled = prior_sbi.sample((n_samples, )).numpy()
    assert isinstance(prior_sbi_sampled[0, 0], np.float32)
    assert_equal(prior_sbi_sampled.shape, (n_samples, len(param_names)))


E = 40*mV
input_traces = np.zeros((10, 5))*volt
for i in range(5):
    input_traces[5:, i] = i * 10*mV
input_traces = input_traces.T
output_traces = input_traces * 10*nS
output_traces = output_traces.T
model = '''
    I = g * (v - E) : amp
    g : siemens (constant)
    '''
model_full = '''
    I = g * (v - 40*mV) : amp
    g : siemens (constant)
    '''


@pytest.fixture
def setup(request):
    dt = 0.01 * ms
    inferencer = Inferencer(dt=dt,
                            model=model,
                            input={'v': input_traces},
                            output={'I': output_traces})

    def fin():
        reinit_devices()
    request.addfinalizer(fin)

    return dt, inferencer


@pytest.fixture
def setup_features(request):
    dt = 0.01 * ms
    inferencer = Inferencer(dt=dt,
                            model=model,
                            input={'v': input_traces},
                            output={'I': output_traces},
                            features={'I': [lambda x: x.mean()]})

    def fin():
        reinit_devices()
    request.addfinalizer(fin)

    return dt, inferencer


@pytest.fixture
def setup_full(request):
    dt = 0.01 * ms
    inferencer = Inferencer(dt=dt,
                            model=model_full,
                            input={'v': input_traces},
                            output={'I': output_traces},
                            features={'I': [lambda x: x.mean()]})

    def fin():
        reinit_devices()
    request.addfinalizer(fin)

    return dt, inferencer


def test_inferencer_init(setup):
    _, inferencer = setup
    attr_inferencer = ['dt', 'output_var', 'output', 'param_names',
                       'n_traces', 'sim_time', 'output_dim', 'model',
                       'input_traces', 'param_init', 'method',
                       'threshold', 'reset', 'refractory', 'x_o',
                       'features', 'params', 'n_samples', 'samples',
                       'posterior', 'theta', 'x', 'sbi_device']
    for attr in attr_inferencer:
        assert hasattr(inferencer, attr)

    assert inferencer.params is None
    assert inferencer.n_samples is None
    assert inferencer.samples is None
    assert inferencer.posterior is None
    assert inferencer.theta is None
    assert inferencer.x is None

    assert isinstance(inferencer.input_traces, TimedArray)
    assert isinstance(inferencer.model, Equations)

    target_var = f'{inferencer.output_var[0]}_target'
    assert target_var in inferencer.model
    assert inferencer.model[target_var].dim is inferencer.output_dim[0]

    assert inferencer.features is None

    observations = []
    for o in output_traces:
        o = np.array(o)
        observations.append(o.ravel().astype(np.float32))
    x_o = np.concatenate(observations)
    assert_almost_equal(inferencer.x_o, x_o)


def test_inferencer_init_features(setup_features):
    _, inferencer = setup_features
    attr_inferencer = ['dt', 'output_var', 'output', 'param_names',
                       'n_traces', 'sim_time', 'output_dim', 'model',
                       'input_traces', 'param_init', 'method',
                       'threshold', 'reset', 'refractory', 'x_o',
                       'features', 'params', 'n_samples', 'samples',
                       'posterior', 'theta', 'x', 'sbi_device']
    for attr in attr_inferencer:
        assert hasattr(inferencer, attr)

    assert inferencer.params is None
    assert inferencer.n_samples is None
    assert inferencer.samples is None
    assert inferencer.posterior is None
    assert inferencer.theta is None
    assert inferencer.x is None

    assert isinstance(inferencer.input_traces, TimedArray)
    assert isinstance(inferencer.model, Equations)

    target_var = f'{inferencer.output_var[0]}_target'
    assert target_var in inferencer.model
    assert inferencer.model[target_var].dim is inferencer.output_dim[0]

    assert callable(inferencer.features['I'][0])

    observations = []
    for o in output_traces:
        o = np.array(o)
        print(o)
        observations.append(o.mean())
    x_o = np.array(observations, dtype=np.float32)
    print(x_o)
    print(inferencer.x_o)
    assert_almost_equal(inferencer.x_o, x_o)


def test_init_prior(setup):
    _, inferencer = setup
    lower_bound = 1e-9
    upper_bound = 100e-9
    g_mean = np.array((lower_bound + upper_bound) / 2, dtype=np.float32)
    g_std = np.array((upper_bound - lower_bound) / np.sqrt(12),
                     dtype=np.float32)
    g_std = g_std.astype(np.float32)
    prior = inferencer.init_prior(g=[lower_bound*siemens, upper_bound*siemens])

    # check mean value
    prior_mean = prior.mean.numpy()
    assert_almost_equal(g_mean, prior_mean)

    # check standard deviation
    prior_std = prior.stddev.numpy()
    assert_almost_equal(g_std, prior_std)


def test_generate_training_data(setup):
    _, inferencer = setup
    lower_bound = 1*nS
    upper_bound = 100*nS
    n_samples = 100
    prior = inferencer.init_prior(g=[lower_bound, upper_bound])
    theta = inferencer.generate_training_data(n_samples, prior)
    assert theta.shape == (100, 1)


def test_extract_summary_statistics(setup):
    _, inferencer = setup
    lower_bound = 1*nS
    upper_bound = 100*nS
    n_samples = 100
    n_traces = 5
    prior = inferencer.init_prior(g=[lower_bound, upper_bound])
    theta = inferencer.generate_training_data(n_samples, prior)
    x = inferencer.extract_summary_statistics(theta)
    assert x.shape == (100, n_traces * 10)


def test_extract_summary_statistics_errors(setup):
    _, inferencer = setup
    lower_bound = 1*nS
    upper_bound = 100*nS
    n_samples = 100
    prior = inferencer.init_prior(g=[lower_bound, upper_bound])
    theta = inferencer.generate_training_data(n_samples, prior)
    with pytest.raises(Exception):
        _ = inferencer.extract_summary_statistics(theta, level=2)


def test_save_summary_statistics_errors(setup):
    _, inferencer = setup
    with pytest.raises(AttributeError):
        inferencer.save_summary_statistics('summary_stats')

    theta = np.array([[1, 2], [3, 4]])
    x = np.arange(2).reshape(2, 1)
    with pytest.raises(TypeError):
        inferencer.save_summary_statistics(1, theta=theta, x=x)
        inferencer.save_summary_statistics(1, theta='theta', x=x)
        inferencer.save_summary_statistics(1, theta=theta, x='x')
        inferencer.save_summary_statistics(1, theta='theta', x='x')


def test_load_summary_statistics(setup):
    _, inferencer = setup
    theta = np.array([[1, 2], [3, 4]])
    x = np.arange(2).reshape(2, 1)
    inferencer.save_summary_statistics('summ_stats.npz', theta=theta, x=x)
    theta_load, x_load = inferencer.load_summary_statistics('summ_stats.npz')
    assert_equal(theta_load, theta)
    assert_equal(x_load, x)


def test_init_inference_errors(setup):
    _, inferencer = setup
    prior = inferencer.init_prior(g=[1*nS, 100*nS])
    with pytest.raises(NameError):
        _ = inferencer.init_inference(inference_method='test',
                                      density_estimator_model='mdn',
                                      prior=prior,
                                      sbi_device='cpu')
        _ = inferencer.init_inference(inference_method='SNPE',
                                      density_estimator_model='test',
                                      prior=prior,
                                      sbi_device='cpu')

    # sbi_device should be automatically set to 'cpu'
    _ = inferencer.init_inference(inference_method='SNPE',
                                  density_estimator_model='mdn',
                                  prior=prior,
                                  sbi_device='test')
    assert inferencer.sbi_device == 'cpu'


def test_infer_step(setup_full):
    _, inferencer = setup_full
    prior = inferencer.init_prior(g=[1*nS, 100*nS])
    inference = inferencer.init_inference(inference_method='SNPE',
                                          density_estimator_model='mdn',
                                          prior=prior,
                                          sbi_device='cpu')
    posterior = inferencer.infer_step(proposal=prior,
                                      n_samples=10,
                                      inference=inference)
    assert posterior._method_family == 'snpe'
    assert_equal(np.array(posterior._x_shape), np.array([1, 5]))


def test_infer_step_errors(setup_full):
    _, inferencer = setup_full
    prior = inferencer.init_prior(g=[1*nS, 100*nS])
    inference = inferencer.init_inference(inference_method='SNPE',
                                          density_estimator_model='mdn',
                                          prior=prior,
                                          sbi_device='cpu')
    with pytest.raises(ValueError):
        _ = inferencer.infer_step(proposal=prior,
                                  inference=inference)


def test_infer(setup):
    _, inferencer = setup
    posterior = inferencer.infer(n_samples=10, g=[1*nS, 100*nS])
    assert posterior == inferencer.posterior

    posterior_multi_round = inferencer.infer(n_samples=10, n_rounds=2,
                                             restart=True, g=[1*nS, 100*nS])
    assert posterior_multi_round != posterior
    assert posterior_multi_round == inferencer.posterior


def test_load_posterior(setup):
    _, inferencer = setup
    _ = inferencer.infer(n_samples=10, g=[1*nS, 100*nS])
    inferencer.save_posterior('posterior.pth')
    posterior_load = inferencer.load_posterior('posterior.pth')
    assert inferencer.posterior.net == posterior_load.net
