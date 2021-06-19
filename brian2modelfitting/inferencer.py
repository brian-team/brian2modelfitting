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
from brian2.monitors.statemonitor import StateMonitor
from brian2.units.allunits import *  # all physical units
from brian2.units.fundamentalunits import (DIMENSIONLESS,
                                           fail_for_dimension_mismatch,
                                           get_dimensions,
                                           Quantity)
import matplotlib.pyplot as plt
import numpy as np
from sbi.utils.get_nn_models import posterior_nn
from sbi.utils.torchutils import BoxUniform
import sbi.analysis
import sbi.inference
import torch

from .simulator import RuntimeSimulator, CPPStandaloneSimulator


def configure_simulator():
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


def get_param_dict(param_values, param_names, n_values):
    """Return a dictionary compiled of parameter names and values.

    Parameters
    ----------
    param_values : iterable
        Iterable of size (`n_samples`, `len(param_names)` containing
        parameter values.
    param_names : iterable
        Iterable containing parameter names
    n_values : int
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
        param_dict[name] = (np.ones((n_values, )) * value)
    return param_dict


def calc_prior(param_names, **params):
    """Return prior distributparion over given parameters. Note that the
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
        # time scale
        self.dt = dt

        # model equations
        if isinstance(model, str):
            model = Equations(model)
        else:
            raise TypeError('Equations must be appropriately formatted.')

        # input data traces
        if not isinstance(input, Mapping):
            raise TypeError('`input` argument must be a dictionary mapping'
                            ' the name of the input variable and `input`.')
        if len(input) > 1:
            raise NotImplementedError('Only a single input is supported.')
        input_var = list(input.keys())[0]
        input = input[input_var]
        if input_var not in model.identifiers:
            raise NameError(f'{input_var} is not an identifier in the model.')

        # output data traces
        if not isinstance(output, Mapping):
            raise TypeError('`output` argument must be a dictionary mapping'
                            ' the name of the output variable and `output`')
        output_var = list(output.keys())
        output = list(output.values())
        for o_var in output_var:
            if o_var not in model.names:
                raise NameError(f'{o_var} is not a model variable')
        self.output_var = output_var
        self.output = output

        # create variable for parameter names
        self.param_names = model.parameter_names

        # set the simulation time for a given time scale
        self.n_traces, n_steps = input.shape
        self.sim_time = dt * n_steps

        # handle multiple output variables
        self.output_dim = []
        for o_var, out in zip(self.output_var, self.output):
            self.output_dim.append(model[o_var].dim)
            fail_for_dimension_mismatch(out, self.output_dim[-1],
                                        'The provided target values must have'
                                        ' the same units as the variable'
                                        f' {o_var}')

        # add input to equations
        self.model = model
        input_dim = get_dimensions(input)
        input_dim = '1' if input_dim is DIMENSIONLESS else repr(input_dim)
        input_eqs = f'{input_var} = input_var(t, i % n_traces) : {input_dim}'
        self.model += input_eqs

        # add output to equations
        counter = 0
        for o_var, o_dim in zip(self.output_var, self.output_dim):
            counter += 1
            output_expr = f'output_var_{counter}(t, i % n_traces)'
            output_dim = ('1' if o_dim is DIMENSIONLESS else repr(o_dim))
            output_eqs = f'{o_var}_target = {output_expr} : {output_dim}'
            self.model += output_eqs

        # create the `TimedArray` object for input w.r.t. a given time scale
        self.input_traces = TimedArray(input.transpose(), dt=self.dt)

        # handle initial values for the ODE system
        if not param_init:
            param_init = {}
        for param in param_init.keys():
            if not (param in self.model.diff_eq_names or
                    param in self.model.parameter_names):
                raise ValueError(f'{param} is not a model variable or a'
                                 ' parameter in the model')
        self.param_init = param_init

        # handle the rest of optional parameters for the `NeuronGroup` class
        self.method = method
        self.threshold = threshold
        self.reset = reset
        self.refractory = refractory

        # placeholder for the number of samples
        self.n_samples = None

    @property
    def n_neurons(self):
        """Return the number of neurons that are used in `NeuronGroup`
        class while generating data for training the neural density
        estimator.

        Unlike the `Fitter` class, `Inferencer` does not take the total
        number of samples in the constructor. Thus, this property
        becomes available only after the simulation is performed.

        Parameters
        ----------
        None

        Returns
        -------
        int
            Total number of neurons.
        """
        if self.n_samples is None:
            raise ValueError('Number of samples is not yet defined.'
                             'Call `generate_training_data` method first.')
        return self.n_traces * self.n_samples

    def setup_simulator(self, network_name, n_neurons, output_var, param_init,
                        level=1):
        """Return configured simulator.

        Parameters
        ----------
        network_name : str
            Network name.
        n_neurons : int
            Number of neurons which equals to the number of samples
            times the number of input/output traces.
        output_var : str
            Name of the output variable.
        param_init : dict
            Dictionary of state variables to be initialized with
            respective values.
        level : int, optional
            How far to go back to get the locals/globals.

        Returns
        -------
        brian2modelfitting.simulator.Simulator
            Configured simulator w.r.t. to the available device.
        """
        # configure the simulator
        simulator = configure_simulator()

        # update the local namespace
        namespace = get_full_namespace({'input_var': self.input_traces,
                                        'n_traces': self.n_traces},
                                       level=level+1)
        counter = 0
        for out in self.output:
            counter += 1
            namespace[f'output_var_{counter}'] = TimedArray(out.transpose(),
                                                            dt=self.dt)

        # setup neuron group
        kwds = {}
        if self.method is not None:
            kwds['method'] = self.method
        model = (self.model
                 + Equations('iteration : integer (constant, shared)'))
        neurons = NeuronGroup(N=n_neurons,
                              model=model,
                              threshold=self.threshold,
                              reset=self.reset,
                              refractory=self.refractory,
                              dt=self.dt,
                              namespace=namespace,
                              name='neurons',
                              **kwds)
        network = Network(neurons)
        network.add(StateMonitor(source=neurons, variables=output_var,
                                 record=True, dt=self.dt, name='statemonitor'))

        # initialize the simulator
        simulator.initialize(network, param_init, name=network_name)
        return simulator

    def initialize_prior(self, **params):
        """Return the prior uniform distribution over parameters.

        Parameters
        ----------
        params : dict
            Bounds for each parameter.

        Returns
        -------
        sbi.utils.BoxUniform
            Uniformly distributed prior over given parameters.
        """
        for param in params:
            if param not in self.param_names:
                raise ValueError(f'Parameter {param} must be defined as a'
                                 ' model\'s parameter')
        prior = calc_prior(self.param_names, **params)
        return prior

    def generate_training_data(self, n_samples, prior, level=1):
        """Return sampled prior and executed simulator containing
        recorded variables to be used for training the neural density
        estimator.

        Parameter
        ---------
        n_samples : int
            The number of samples.
        prior : sbi.utils.BoxUniform
            Uniformly distributed prior over given parameters.
        level : int, optional
            How far to go back to get the locals/globals.

        Returns
        -------
        numpy.ndarray
            Sampled prior of shape (`n_samples`, -1)
        brian2modelfitting.simulator.Simulator
            Executed simulator.
        """
        # set n_samples to class variable to be able to call self.n_neurons
        self.n_samples = n_samples

        # sample from prior
        theta = prior.sample((n_samples, ))
        theta = np.atleast_2d(theta.numpy())

        # repeat each row for how many input/output different trace are there
        _theta = np.repeat(theta, repeats=self.n_traces, axis=0)

        # create a dictionary with repeated sampled prior
        d_param = get_param_dict(_theta, self.param_names, self.n_neurons)

        # setup and run the simulator
        network_name = 'infere'
        simulator = self.setup_simulator(network_name=network_name,
                                         n_neurons=self.n_neurons,
                                         output_var=self.output_var,
                                         param_init=self.param_init,
                                         level=level+1)
        simulator.run(self.sim_time, d_param, self.param_names, iteration=0,
                      name=network_name)
        return (theta, simulator)

    def train(self, n_samples, n_rounds=1, estimation_method='SNPE',
              density_estimator_model='maf', **params):
        """Return the trained neural density estimator.

        Currently only sequential neural posterior estimator is
        supported.

        Parameter
        ---------
        n_samples : int
            The number of samples.
        n_rounds : int or str, optional
            If `n_rounds`is set to 1, amortized inference will be
            performed. Otherwise, if `n_rounds` is integer larger than
            1, multi-round inference will be performed.
        estimation_method : str
            Inference method. Either of SNPE, SNLE or SNRE. Currently,
            only SNPE is supported.
        density_estimator_model : str
            The type of density estimator to be created. Either `mdn`,
            `made`, `maf` or `nsf`.
        params : dict
            Bounds for each parameter.

        Returns
        -------
        sbi.inference.posteriors.direct_posterior.DirectPosterior
            Trained posterior.
        """
        if not isinstance(n_rounds, int):
            raise ValueError('Number of rounds must be a positive integer.')
        if str.upper(estimation_method) != 'SNPE':
            raise NotImplementedError('Only SNPE estimator is supported.')

        # observation the focus is on
        x_o = []
        for o in self.output:
            o_dim = get_dimensions(o)
            o_obs = self.extract_features(o.transpose(), o_dim)
            x_o.append(o_obs.flatten())
        x_o = torch.tensor(x_o, dtype=torch.float32)
        self.x_o = x_o

        # initialize prior, density estimator and inference method
        prior = self.initialize_prior(**params)
        density_esimator = posterior_nn(density_estimator_model)
        inference = sbi.inference.SNPE(prior, density_esimator)

        # allocate empty list of posteriors, for multi-round inference only
        posteriors = []
        proposal = prior
        for _ in range(n_rounds):
            # extract the data and make adjustments for `sbi`
            theta, simulator = self.generate_training_data(n_samples, proposal)
            theta = torch.tensor(theta, dtype=torch.float32)
            obs = simulator.statemonitor.recorded_variables
            x_val = obs[self.output_var[0]].get_value_with_unit()
            x_dim = get_dimensions(obs[self.output_var[0]])
            x = torch.tensor(self.extract_features(x_val, x_dim),
                             dtype=torch.float32)
            x = x.reshape(self.n_samples, -1)

            # pass the simulated data to the inference object and train it
            de = inference.append_simulations(theta, x, proposal).train()

            # use the density estimator to build the posterior
            posterior = inference.build_posterior(de)

            # append the current posterior to the list of posteriors
            posteriors.append(posterior)

            # update the proposal given the observation
            proposal = posterior.set_default_x(x_o)
        self.posterior = posterior
        return posterior

    def extract_features(self, obs, obs_dim):
        obs_mean = obs.mean(axis=0)
        obs_std = obs.std(axis=0)
        obs_ptp = obs.ptp(axis=0)
        obs_mean = obs_mean / Quantity(np.ones_like(obs_mean), obs_dim)
        obs_std = obs_std / Quantity(np.ones_like(obs_std), obs_dim)
        obs_ptp = obs_ptp / Quantity(np.ones_like(obs_ptp), obs_dim)
        return np.hstack((np.array(obs_mean).reshape(-1, 1),
                          np.array(obs_std).reshape(-1, 1),
                          np.array(obs_ptp).reshape(-1, 1)))

    def sample(self, size, viz=False, **kwargs):
        samples = self.posterior.sample((size, ))
        if viz:
            sbi.analysis.pairplot(samples, **kwargs)
            plt.tight_layout()
            plt.show()
        return samples