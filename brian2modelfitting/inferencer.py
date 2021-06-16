from numbers import Number
from typing import Mapping

from brian2.core.functions import Function
from brian2.core.namespace import get_local_namespace
from brian2.core.network import Network
from brian2.devices.cpp_standalone.device import CPPStandaloneDevice
from brian2.devices.device import get_device, device
from brian2.equations.equations import Equations
from brian2.groups.neurongroup import NeuronGroup
from brian2.input.timedarray import TimedArray
from brian2.monitors.spikemonitor import SpikeMonitor
from brian2.monitors.statemonitor import StateMonitor
from brian2.units.allunits import *  # all physical units
from brian2.units.fundamentalunits import (DIMENSIONLESS, get_dimensions,
                                           Quantity)
import matplotlib.pyplot as plt
import numpy as np
from sbi.utils.get_nn_models import posterior_nn
from sbi.utils.torchutils import BoxUniform
import sbi.analysis
import sbi.inference
import torch

from .simulator import RuntimeSimulator, CPPStandaloneSimulator


def setup_simulator():
    """Return the configured simulator, which can be either
    `RuntimeSimulator`, object for use with `RuntimeDevice`, or
    `CPPStandaloneSimulator`, object for use with `CPPStandaloneDevice.

    Parameters
    ----------
    None

    Returns
    -------
    brian2modelfitting.simulator.Simulator
        Either `RuntimeSimulator` or `CPPStandaloneSimulator` depending
        on the currently active `Device` object describing the
        available computational engine.
    """
    simulators = {'CPPStandaloneDevice': CPPStandaloneSimulator(),
                  'RuntimeDevice': RuntimeSimulator()}
    if isinstance(get_device(), CPPStandaloneDevice):
        if device.has_been_run is True:
            build_options = dict(device.build_options)
            get_device().reinit()
            get_device().activate(**build_options)
    return simulators[get_device().__class__.__name__]


def get_full_namespace(additional_namespace, level=0):
    """Return the namespace with added `additional_namespace`, in which
    references to external parameters or functions are stored.

    Parameters
    ----------
    additional_namespace : dict
        References to external parameters or functions, where key is
        the name and value is the value of the external param/func.
    level : int, optional
        How far to go back to get the locals/globals.

    Returns
    -------
    dict
        Namespace with additional references to external parameters or
        functions.
    """
    namespace = {key: value
                 for key, value in get_local_namespace(level=level + 1).items()
                 if isinstance(value, (Number, Quantity, Function))}
    namespace.update(additional_namespace)
    return namespace


def get_param_dict(param_values, param_names, n_samples):
    """Return a dictionary compiled of parameter names and values.

    Parameters
    ----------
    param_values : iterable
        Iterable of size (`n_samples`, `len(param_names)` containing
        parameter values.
    param_names : iterable
        Iterable containing parameter names
    n_samples : int
        Total number of given values for a single parameter.

    Returns
    -------
    dict
        Dictionary containing key-value pairs thet correspond to a
        parameter name and value(s)
    """
    param_values = np.array(param_values)
    param_dict = dict()
    for name, value in zip(param_names, param_values.T):
        param_dict[name] = (np.ones((n_samples, )) * value)
    return param_dict


def calc_prior(param_names, **params):
    """Return prior distribution over given parameters. Note that the
    only available prior distribution currently supported is
    multidimensional uniform distribution defined on a box.

    Parameters
    ----------
    param_names : iterable
        Iterable containing parameter names.
    params : dict
        Dictionary with keys that correspond to parameter names, and
        values should be a single dimensional lists or arrays

    Return
    ------
    sbi.utils.torchutils.BoxUniform
        `sbi` compatible object that contains a uniform prior
        distribution over a given set of parameter
    """
    for param_name in param_names:
        if param_name not in params:
            raise TypeError(f'"Bounds must be set for parameter {param_name}')
    prior_min = []
    prior_max = []
    for param_name in param_names:
        prior_min.append(min(params[param_name]).item())
        prior_max.append(max(params[param_name]).item())
    prior = BoxUniform(low=torch.as_tensor(prior_min),
                       high=torch.as_tensor(prior_max))
    return prior


class Inferencer(object):
    """Class for simulation-based inference.

    It offers an interface similar to that of `Fitter` class but
    instead of fitting, neural density estimator is trained using a
    generative model. This class serves as a wrapper for `sbi` library
    for inferencing posterior over unknown parameters of a given model.

    Parameters
    ----------
    dt : brian2.units.fundamentalunits.Quantity
        Integration time step.
    model : str or brian2.equations.equations.Equations
        Single cell model equations.
    input : dict
        Input traces in dictionary format, where key corresponds to the
        name of the input variable as defined in `model` and value
        corresponds to a single dimensional array of data traces.
    output : dict
        Dictionary of recorded (or simulated) output data traces, where
        key corresponds to the name of the output variable as defined
        in `model` and value corresponds to a single dimensional array
        of recorded data traces.
    method : str, optional
        Integration method.
    threshold : str, optional
        The condition which produces spikes. Should be a single line
        boolean expression.
    reset : str, optional
        The (possibly multi-line) string with the code to execute on
        reset.
    refractory : str, optional
        Either the length of the refractory period (e.g., `2*ms`), a
        string expression that evaluates to the length of the
        refractory period after each spike (e.g., `'(1 + rand())*ms'`),
        or a string expression evaluating to a boolean value, given the
        condition under which the neuron stays refractory after a spike
        (e.g., `'v > -20*mV'`).
    param_init : dict, optional
        Dictionary of state variables to be initialized with respective
        values.
    """
    def __init__(self, dt, model, input, output, method=None, threshold=None,
                 reset=None, refractory=None, param_init=None):
        self.dt = dt
        if isinstance(model, str):
            self.model = Equations(model)
        else:
            raise TypeError('Equations must be appropriately formatted.')
        if not isinstance(input, Mapping):
            raise TypeError('`input` argument must be a dictionary mapping'
                            ' the name of the input variable and `input`.')
        if len(input) > 1:
            raise NotImplementedError('Only a single input is supported.')
        self.input_var = list(input.keys())[0]
        self.input = input[self.input_var]
        if not isinstance(output, Mapping):
            raise TypeError('`output` argument must be a dictionary mapping'
                            ' the name of the output variable and `output`')
        if len(output) > 1:
            raise NotImplementedError('Only a single output is supported')
        self.output_var = list(output.keys())[0]
        self.output = output[self.output_var]
        input_dim = get_dimensions(self.input)
        input_dim = '1' if input_dim is DIMENSIONLESS else repr(input_dim)
        input_eqs = f'{self.input_var} = input_var(t) : {input_dim}'
        self.model += input_eqs
        self.input_traces = TimedArray(self.input.transpose(), dt=self.dt)
        n_steps = self.input.size
        self.sim_time = self.dt * n_steps
        if not param_init:
            param_init = {}
        for param, val in param_init.items():
            if not (param in self.model.diff_eq_names or
                    param in self.model.parameter_names):
                raise ValueError(f'{param} is not a model variable or a'
                                 ' parameter in the model')
        self.param_init = param_init
        self.param_names = self.model.parameter_names
        self.method = method
        self.threshold = threshold
        self.reset = reset
        self.refractory = refractory

    def setup_simulator(self, n_samples, output_var, param_init, level=1):
        simulator = setup_simulator()
        namespace = get_full_namespace({'input_var': self.input_traces},
                                       level=level+1)
        namespace['output_var'] = TimedArray(self.output.transpose(),
                                             dt=self.dt)
        kwargs = {}
        if self.method is not None:
            kwargs['method'] = self.method
        model = (self.model
                 + Equations('iteration : integer (constant, shared)'))
        neurons = NeuronGroup(N=n_samples,
                              model=model,
                              threshold=self.threshold,
                              reset=self.reset,
                              refractory=self.refractory,
                              dt=self.dt,
                              namespace=namespace,
                              name='neurons',
                              **kwargs)
        network = Network(neurons)
        network.add(StateMonitor(source=neurons, variables=output_var,
                                 record=True, dt=self.dt, name='statemonitor'))
        simulator.initialize(network, param_init)
        return simulator

    def generate(self, n_samples, level=0, **params):
        try:
            n_samples = int(n_samples)
        except ValueError as e:
            print(e)
        for param in params:
            if param not in self.param_names:
                raise ValueError(f'Parameter {param} must be defined as a'
                                 ' model\'s parameter')
        self.prior = calc_prior(self.param_names, **params)
        self.theta = self.prior.sample((n_samples, ))
        theta = np.atleast_2d(self.theta.numpy())
        self.simulator = self.setup_simulator(n_samples=n_samples,
                                              output_var=self.output_var,
                                              param_init=self.param_init,
                                              level=1)
        d_param = get_param_dict(theta, self.param_names, n_samples)
        self.simulator.run(self.sim_time, d_param, self.param_names, 0)

    def _create_sum_stats(self, obs, obs_dim):
        obs_mean = obs.mean(axis=0)
        obs_std = obs.std(axis=0)
        obs_ptp = obs.ptp(axis=0)
        obs_mean = obs_mean / Quantity(np.ones_like(obs_mean), obs_dim)
        obs_std = obs_std / Quantity(np.ones_like(obs_std), obs_dim)
        obs_ptp = obs_ptp / Quantity(np.ones_like(obs_ptp), obs_dim)
        return np.hstack((np.asarray(obs_mean).reshape(-1, 1),
                          np.asarray(obs_std).reshape(-1, 1),
                          np.asarray(obs_ptp).reshape(-1, 1)))

    def train(self):
        obs_val = self.simulator.statemonitor.recorded_variables
        obs = obs_val[self.output_var].get_value_with_unit()
        obs_dim = self.simulator.statemonitor.recorded_variables
        dim = get_dimensions(obs_dim[self.output_var])
        x = torch.tensor(self._create_sum_stats(obs, dim), dtype=torch.float32)
        density_esimator = posterior_nn(model='maf')
        self.inference = sbi.inference.SNPE(self.prior, density_esimator)
        self.inference.append_simulations(self.theta, x).train()
        self.posterior = self.inference.build_posterior()

    def sample(self, size, viz=False, **kwargs):
        dim = get_dimensions(self.output)
        x_o = torch.tensor(self._create_sum_stats(self.output.ravel(), dim),
                           dtype=torch.float32)
        samples = self.posterior.sample((size, ), x_o)
        if viz:
            sbi.analysis.pairplot(samples, **kwargs)
            plt.tight_layout()
            plt.show()
        return samples
