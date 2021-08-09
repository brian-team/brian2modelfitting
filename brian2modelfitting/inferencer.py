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
from brian2.units.fundamentalunits import (DIMENSIONLESS,
                                           fail_for_dimension_mismatch,
                                           get_dimensions,
                                           Quantity)
import numpy as np
from sbi.utils.get_nn_models import (posterior_nn,
                                     likelihood_nn,
                                     classifier_nn)
from sbi.utils.torchutils import BoxUniform
import sbi.analysis
import sbi.inference
import torch

from .simulator import RuntimeSimulator, CPPStandaloneSimulator


def configure_simulator():
    """Return the configured simulator, which can be either
    `.RuntimeSimulator`, object for the use with `.RuntimeDevice`, or
    `.CPPStandaloneSimulator`, object for the use with
    `.CPPStandaloneDevice`.

    Parameters
    ----------
    None

    Returns
    -------
    brian2modelfitting.simulator.Simulator
        Either `.RuntimeSimulator` or `.CPPStandaloneSimulator`
        depending on the currently active ``Device`` object describing
        the available computational engine.
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
    """Return the namespace with added ``additional_namespace``, in
    which references to external parameters or functions are stored.

    Parameters
    ----------
    additional_namespace : dict
        References to external parameters or functions, where key is
        the name and value is the value of the external parameter or
        function.
    level : int, optional
        How far to go back to get the locals/globals.

    Returns
    -------
    dict
        Namespace with additional references to the external parameters
        or functions.
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
        Iterable of size (``n_samples``, ``len(param_names)``)
        containing parameter values.
    param_names : iterable
        Iterable containing parameter names
    n_values : int
        Total number of given values for a single parameter.

    Returns
    -------
    dict
        Dictionary containing key-value pairs that correspond to a
        parameter name and value(s).
    """
    param_values = np.array(param_values)
    param_dict = dict()
    for name, value in zip(param_names, param_values.T):
        param_dict[name] = (np.ones((n_values, )) * value)
    return param_dict


def calc_prior(param_names, **params):
    """Return the prior distribution over given parameters.

    N.B. The only currently supported prior distribution is
    the multi-dimensional uniform distribution defined on a box.

    Parameters
    ----------
    param_names : iterable
        Iterable containing parameter names.
    params : dict
        Dictionary with keys that correspond to parameter names, values
        are a single dimensional lists or arrays.

    Return
    ------
    sbi.utils.torchutils.BoxUniform
        ``sbi`` compatible object that contains a uniform prior
        distribution over a given set of parameters.
    """
    for param_name in param_names:
        if param_name not in params:
            raise TypeError(f'Bounds must be set for parameter {param_name}')
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

    It offers an interface similar to that of the `.Fitter` class but
    instead of fitting, neural density estimator is trained using a
    generative model which ultimately provides the posterior
    distribution over free parameters.
    This class serves as a wrapper for ``sbi`` library for inferencing
    posterior over unknown parameters of a given model.

    Parameters
    ----------
    dt : brian2.units.fundamentalunits.Quantity
        Integration time step.
    model : str or brian2.equations.equations.Equations
        Single cell model equations.
    input : dict
        Input traces in dictionary format where key corresponds to the
        name of the input variable as defined in ``model`` and value
        corresponds to an array of input traces.
    output : dict
        Dictionary of recorded (or simulated) output data traces, where
        key corresponds to the name of the output variable as defined
        in ``model`` and value corresponds to a an array of recorded
        traces.
    features : dict, optional
        Dictionary of callables that take the 1-dimensional voltage
        trace and output summary statistics. Keys correspond to output
        variable names, while values are lists of callables.
        If features are not provided, automatic feature extraction will
        be performed either by using the default multi-layer perceptron
        or by using the user-provided embedding network.
    method : str, optional
        Integration method.
    threshold : str, optional
        The condition which produces spikes. Should be a single line
        boolean expression.
    reset : str, optional
        The (possibly multi-line) string with the code to execute on
        reset.
    refractory : bool or str, optional
        Either the length of the refractory period (e.g., ``2*ms``), a
        string expression that evaluates to the length of the
        refractory period after each spike, e.g., ``'(1 + rand())*ms'``,
        or a string expression evaluating to a boolean value, given the
        condition under which the neuron stays refractory after a spike,
        e.g., ``'v > -20*mV'``.
    param_init : dict, optional
        Dictionary of state variables to be initialized with respective
        values.
    """
    def __init__(self, dt, model, input, output, features=None, method=None,
                 threshold=None, reset=None, refractory=False,
                 param_init=None):
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
                            ' the name of the output variable and `output`.')
        output_var = list(output.keys())
        output = list(output.values())
        for o_var in output_var:
            if o_var not in model.names:
                raise NameError(f'{o_var} is not a model variable')
        self.output_var = output_var
        self.output = output

        # create variable for parameter names
        self.param_names = sorted(model.parameter_names)

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

        # create ``TimedArray`` object for input w.r.t. a given time scale
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

        # handle the rest of optional parameters for the ``NeuronGroup`` class
        self.method = method
        self.threshold = threshold
        self.reset = reset
        self.refractory = refractory

        # observation the focus is on
        for ov, o in zip(self.output_var, self.output):
            o = np.array(o)
            if features:
                obs = []
                for _o in o:
                    for feature in features[ov]:
                        obs.append(feature(_o))
                x_o = np.array(obs, dtype=np.float32)
            else:
                x_o = o.ravel().astype(np.float32)
        self.x_o = x_o
        self.features = features

        # additional placeholders
        self.params = None
        self.n_samples = None
        self.samples = None
        self.posterior = None
        self.theta = None
        self.x = None

    @property
    def n_neurons(self):
        """Return the number of neurons that are used in `.NeuronGroup`
        class while generating data for training the neural density
        estimator.

        Unlike the `.Fitter` class, `.Inferencer` does not take the
        total number of samples in the constructor. Thus, this property
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
            raise ValueError('Number of samples have not been yet defined.'
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
            Configured simulator w.r.t. the available device.
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

    def init_prior(self, **params):
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
        self.params = params
        return prior

    def generate_training_data(self, n_samples, prior):
        """Return sampled prior given the total number of samples.

        Parameter
        ---------
        n_samples : int
            The number of samples.
        prior : sbi.utils.BoxUniform
            Uniformly distributed prior over given parameters.

        Returns
        -------
        numpy.ndarray
            Sampled prior of shape (``n_samples``, -1).
        """
        # set n_samples to class variable to be able to call self.n_neurons
        self.n_samples = n_samples

        # sample from the prior distribution
        theta = prior.sample((n_samples, ))
        theta = np.atleast_2d(theta.numpy())
        return theta

    def extract_summary_statistics(self, theta, level=0):
        """Return summary statistics for training the neural density
        estimator.

        Parameters
        ----------
        theta : numpy.ndarray
            Sampled prior with ``n_samples`` rows, while then umber of
            columns is equal to the number of free parameters.
        level : int, optional
            How far to go back to get the locals/globals.

        Returns
        -------
        numpy.ndarray
            Summary statistics.
        """
        # repeat each row for how many input/output different trace are there
        _theta = np.repeat(theta, repeats=self.n_traces, axis=0)

        # create a dictionary with repeated sampled prior
        d_param = get_param_dict(_theta, self.param_names, self.n_neurons)

        # set up and run the simulator
        network_name = 'infere'
        simulator = self.setup_simulator(network_name=network_name,
                                         n_neurons=self.n_neurons,
                                         output_var=self.output_var,
                                         param_init=self.param_init,
                                         level=level+1)
        simulator.run(self.sim_time, d_param, self.param_names, iteration=0,
                      name=network_name)

        # extract features for each output variable and each trace
        obs = simulator.statemonitor.recorded_variables
        for ov in self.output_var:
            o = obs[ov].get_value()
            o = o.T
            if self.features:
                summary_statistics = []
                # TODO: should be vectorized
                for _o in o:
                    for feature in self.features[ov]:
                        summary_statistics.append(feature(_o))
                x = np.array(summary_statistics, dtype=np.float32)
                x = x.reshape(self.n_samples, -1)
            else:
                x = o.reshape(self.n_samples, -1).astype(np.float32)
        return x

    def save_summary_statistics(self, f, theta=None, x=None):
        """Save sampled prior, and extracted summary statistics into a
        single file in compressed ``.npz`` format.

        Parameters
        ----------
        f : str or os.PathLike
            Path to a file either as string or ``os.PathLike`` object
            that contains file name.
        theta : numpy.ndarray, optional
            Sampled prior.
        x : numpy.ndarray, optional
            Summary statistics.

        Returns
        -------
        None
        """
        if theta is not None:
            t = theta
        elif self.theta is not None:
            t = self.theta
        else:
            raise AttributeError('Provide sampled prior or call'
                                 ' `infer_step` method first.')
        if x is not None:
            pass
        elif self.x is not None:
            x = self.x
        else:
            raise AttributeError('Provide summary feautures or call '
                                 ' `infer_step` method first.')
        np.savez_compressed(f, theta=t, x=x)

    def load_summary_statistics(self, f):
        """Load samples from a prior and extracted summary statistics
        from a compressed ``.npz`` file.

        Parameters
        ----------
        f : str or os.PathLike
            Path to file either as string or ``os.PathLike`` object
            that contains file name.

        Returns
        -------
        tuple
            Sampled prior and summary statistics arrays.
        """
        loaded = np.load(f, allow_pickle=True)
        if set(loaded.files) == {'theta', 'x'}:
            theta = loaded['theta']
            x = loaded['x']
        self.theta = theta
        self.x = x
        return (theta, x)

    def init_inference(self, inference_method, density_estimator_model, prior,
                       **inference_kwargs):
        """Return instantiated inference object.

        Parameters
        ----------
        inference_method : str
            Inference method. Either of SNPE, SNLE or SNRE.
        density_estimator_model : str
            The type of density estimator to be created. Either
            ``mdn``, ``made``, ``maf``, ``nsf`` for SNPE and SNLE, or
            ``linear``, ``mlp``, ``resnet`` for SNRE.
        prior : sbi.utils.BoxUniform
            Uniformly distributed prior over given parameters.
        inference_kwargs : dict, optional
            Additional keyword arguments for
            ``sbi.utils.get_nn_models.posterior_nn`` method. The user
            is free to provide own embedding network to learn features
            from potentially high-dimensional simulation outputs. By
            default multi-layer perceptron is used if no user defined
            embedding network is provided.

        Returns
        -------
        sbi.inference.NeuralInference
            Instantiated inference object.
        """
        try:
            inference_method = str.upper(inference_method)
            inference_method_fun = getattr(sbi.inference, inference_method)
        except AttributeError:
            raise NameError(f'Inference method {inference_method} is not'
                            ' supported. Choose between SNPE, SNLE or SNRE.')
        finally:
            if inference_method == 'SNPE':
                density_estimator_builder = posterior_nn(
                    model=density_estimator_model, **inference_kwargs)
            elif inference_method == 'SNLE':
                density_estimator_builder = likelihood_nn(
                    model=density_estimator_model, **inference_kwargs)
            else:
                density_estimator_builder = classifier_nn(
                    model=density_estimator_model, **inference_kwargs)
        inference = inference_method_fun(prior, density_estimator_builder,
                                         device='cpu',
                                         show_progress_bars=True)
        return inference

    def train(self, inference, theta, x, *args, **train_kwargs):
        """Return trained neural inference object.

        Parameters
        ----------
        inference : sbi.inference.NeuralInference
            Instantiated inference object with stored paramaters and
            simulation outputs prepared for training.
        theta : torch.tensor
            Sampled prior.
        x : torch.tensor
            Summary statistics.
        args : tuple, optional
            Contains a uniformly distributed sbi.utils.BoxUniform
            proposal. Used only for SNPE, for SNLE and SNRE,
            ``proposal`` should not be passed to ``append_simulations``
            method, thus ``args`` should not be passed. The args should
            be provided only if the parameters were not sampled from
            the prior, e.g., during the multi-round inference.
            For SNLE and SNRE, args can hold the number of round from
            which the data is stemmed from. Round 0 means from prior.
            This is used only if the `discard_prior_samples` are set to
            True inside the `train_kwargs`.
        train_kwargs : dict, optional
            Additional keyword arguments for ``train`` method of
            ``sbi.inference.NeuralInference`` object.

        Returns
        -------
        sbi.inference.NeuralInference
            Trained inference object.
        """
        inference = inference.append_simulations(theta, x, *args)
        _ = inference.train(**train_kwargs)
        return inference

    def build_posterior(self, inference, **posterior_kwargs):
        """Return inference and neural posterior objects.

        Parameters
        ----------
        inference : sbi.inference.NeuralInference
            Instantiated inference object with stored paramaters and
            simulation outputs prepared for training.
        posterior_kwargs : dict, optional
            Additional keyword arguments for ``build_posterior`` method
            of ``sbi.inference.NeuralInference`` object.

        Returns
        -------
        tuple
            ``sbi.inference.NeuralInference`` object with stored
            paramaters and simulation outputs prepared for training and
            ``sbi.inference.NeuralPosterior`` object.
        """
        posterior = inference.build_posterior(**posterior_kwargs)
        return (inference, posterior)

    def infer_step(self, proposal, inference,
                   n_samples=None, theta=None, x=None,
                   train_kwargs={}, posterior_kwargs={}, *args):
        """Return the trained neural density estimator.

        Parameter
        ---------
        proposal : ``sbi.utils.torchutils.BoxUniform``
            Prior over parameters for current inference round.
        inference : ``sbi.inference.NeuralInference``
            Inference object obtained via `.init_inference` method.
        n_samples : int, optional
            The number of samples.
        theta : numpy.ndarray, optional
            Sampled prior.
        x : numpy.ndarray, optional
            Summary statistics.
        train_kwargs : dict, optional
            Additional keyword arguments for training the posterior
            estimator.
        posterior_kwargs : dict, optional
            Dictionary of arguments for `.build_posterior` method.
        args : list, optional
            Additional arguments for `.train` method if SNPE is used as
            an inference method.

        Returns
        -------
        sbi.inference.NeuralPosterior
            Trained posterior.
        """
        # extract the training data and make adjustments for ``sbi``
        if theta is None:
            if n_samples is None:
                raise ValueError('Either provide `theta` or `n_samples`.')
            else:
                theta = self.generate_training_data(n_samples, proposal)
        self.theta = theta
        theta = torch.tensor(theta, dtype=torch.float32)

        # extract the summary statistics and make adjustments for ``sbi``
        if x is None:
            if n_samples is None:
                raise ValueError('Either provide `x` or `n_samples`.')
            else:
                x = self.extract_summary_statistics(theta, level=2)
        self.x = x
        x = torch.tensor(x)

        # pass the simulated data to the inference object and train it
        inference = self.train(inference, theta, x, *args, **train_kwargs)

        # use the density estimator to build the posterior
        inference, posterior = self.build_posterior(inference,
                                                    **posterior_kwargs)
        self.inference = inference
        self.posterior = posterior
        return posterior

    def infer(self, n_samples=None, theta=None, x=None, n_rounds=1,
              inference_method='SNPE', density_estimator_model='maf',
              inference_kwargs={}, train_kwargs={}, posterior_kwargs={},
              restart=False, **params):
        """Return the trained posterior.

        If ``theta`` and ``x`` are not provided, ``n_samples`` has to
        be defined. Otherwise, if ``n_samples`` is provided, neither
        ``theta`` nor ``x`` needs to be provided.

        Parameter
        ---------
        n_samples : int, optional
            The number of samples.
        theta : numpy.ndarray, optional
            Sampled prior.
        x : numpy.ndarray, optional
            Summary statistics.
        n_rounds : int, optional
            If ``n_rounds`` is set to 1, amortized inference will be
            performed. Otherwise, if ``n_rounds`` is integer larger
            than 1, multi-round inference will be performed. This is
            only valid if posterior has not been defined manually.
            Otherwise, if this method is called after posterior has
            already been built, multi-round inference is performed.
        inference_method : str, optional
            Inference method. Either of SNPE, SNLE or SNRE.
        density_estimator_model : str, optional
            The type of density estimator to be created. Either
            ``mdn``, ``made``, ``maf``, ``nsf`` for SNPE and SNLE, or
            ``linear``, ``mlp``, ``resnet`` for SNRE.
        inference_kwargs : dict, optional
            Additional keyword arguments for the inferencer method
            definition.
        train_kwargs : dict, optional
            Additional keyword arguments for training the posterior
            estimator.
        posterior_kwargs : dict, optional
            Additional keyword arguments for builing the posterior.
        restart : bool, optional
            When the method is called for a second time, set to True if
            amortized inference should be performed. If False,
            multi-round inference with the existing posterior will be
            performed.
        params : dict
            Bounds for each parameter. Keys should correspond to names
            of parameters as defined in the model equaions, values
            are lists with lower and upper bounds with quantities of
            respective parameter.

        Returns
        -------
        sbi.inference.NeuralPosterior
            Approximated posterior distribution over parameters.
        """
        if restart:
            self.posterior = None
        if self.posterior is None:
            # handle the number of rounds
            if not isinstance(n_rounds, int):
                raise ValueError('`n_rounds` has to be a positive integer.')

            # handle inference methods
            try:
                inference_method = str.upper(inference_method)
            except ValueError as e:
                print(e, '\nInference method should be defined as string.')
            if inference_method not in ['SNPE', 'SNLE', 'SNRE']:
                raise ValueError(f'Inference method {inference_method} is not'
                                 ' supported.')

            # initialize prior
            prior = self.init_prior(**params)

            # extract the training data and make adjustments for ``sbi``
            if theta is None:
                if n_samples is None:
                    raise ValueError('Either provide `theta` or `n_samples`.')
                else:
                    theta = self.generate_training_data(n_samples, prior)
            self.theta = theta

            # extract the summary statistics and make adjustments for ``sbi``
            if x is None:
                if n_samples is None:
                    raise ValueError('Either provide `x` or `n_samples`.')
                else:
                    x = self.extract_summary_statistics(theta, level=1)
            self.x = x

            # initialize inference object
            self.inference = self.init_inference(inference_method,
                                                 density_estimator_model,
                                                 prior,
                                                 **inference_kwargs)

            # args for SNPE in `.train`
            # args for SNRE and SNLE are not supported here, if needed the user
            # could provide them by using more flexible inferface via
            # `.infer_step` method
            if inference_method == 'SNPE':
                args = [prior, ]
            else:
                args = [None, ]
        else:  # `.infer_step` has been called manually
            x_o = torch.tensor(self.x_o, dtype=torch.float32)
            prior = self.posterior.set_default_x(x_o)
            if self.posterior._method_family == 'snpe':
                args = [prior, ]
            else:
                args = [None, ]

        # allocate empty list of posteriors
        posteriors = []

        # set a proposal
        proposal = prior

        # main inference loop
        for round in range(n_rounds):
            print(f'Round {round + 1}/{n_rounds}.')

            # inference step
            posterior = self.infer_step(proposal, self.inference,
                                        n_samples, self.theta, self.x,
                                        train_kwargs, posterior_kwargs, *args)

            # append the current posterior to the list of posteriors
            posteriors.append(posterior)

            # update the proposal given the observation
            if n_rounds > 1:
                x_o = torch.tensor(self.x_o, dtype=torch.float32)
                proposal = posterior.set_default_x(x_o)
        self.posterior = posterior
        return posterior

    def save_posterior(self, f):
        """Saves the density estimator state dictionary to a disk file.

        Parameters
        ----------
        posterior : sbi.inference.NeuralPosterior, optional
            Posterior distribution.
        f : str or os.PathLike
            Path to file either as string or ``os.PathLike`` object
            that contains file name.

        Returns
        -------
        None
        """
        torch.save(self.posterior, f)

    def load_posterior(self, f):
        """Loads the density estimator state dictionary from a disk file.

        Parameters
        ----------
        f : str or os.PathLike
            Path to file either as string or ``os.PathLike`` object
            that contains file name.

        Returns
        -------
        sbi.inference.NeuralPosterior
            Loaded neural posterior with defined method family, density
            estimator state dictionary, prior over parameters and
            output shape of the simulator.
        """
        p = torch.load(f)
        self.posterior = p
        return p

    def sample(self, shape, posterior=None, **kwargs):
        """Return samples from posterior distribution.

        Parameters
        ----------
        shape : tuple
            Desired shape of samples that are drawn from posterior.
        posterior : sbi.inference.NeuralPosterior, optional
            Posterior distribution.
        kwargs : dict, optional
            Additional keyword arguments for ``sample`` method of
            ``sbi.inference.NeuralPosterior`` object.
        Returns
        -------
        numpy.ndarray
            Samples from posterior of the shape as given in ``shape``.
        """
        if posterior:
            p = posterior
        elif posterior is None and self.posterior:
            p = self.posterior
        else:
            raise ValueError('Need to provide posterior argument if no'
                             ' posterior has computed built yet.')
        x_o = torch.tensor(self.x_o, dtype=torch.float32)
        samples = p.sample(shape, x=x_o, **kwargs)
        self.samples = samples
        return samples.numpy()

    def pairplot(self, samples=None, points=None, limits=None, subset=None,
                 labels=None, ticks=None, **kwargs):
        """Plot samples in a 2-D grid with marginals and pairwise
        marginals.

        Check ``sbi.analysis.plot.pairplot`` for more details.

        Parameters
        ----------
        samples : iterable, optional
            Samples used to build the pairplot.
        points : dict, optional
            Additional points to scatter, e.g., true parameter values,
            if known.
        limits : dict, optional
            Limits for each parameter. If None, min and max of the
            given samples will be used.
        subset : list, optional
            Which parameters to plot.
        labels : dict, optional
            Names for each parameter.
        ticks : dict, optional
            Position of the ticks. If None, default ticks positions
            will be used.
        kwargs : dict, optional
            Additional keyword arguments for the
            ``sbi.analysis.pairplot`` function.

        Returns
        -------
        tuple
            Figure and axis of posterior distribution plot.
        """
        if samples is not None:
            s = samples
        else:
            if self.samples is not None:
                s = self.samples
            else:
                raise ValueError('Samples are not available.')
        if points:
            for param_name in points.keys():
                if param_name not in self.param_names:
                    raise AttributeError(f'Invalid parameter: {param_name}')
            points = np.array([[points[param_name].item()
                               for param_name in self.param_names]])
        if limits:
            for param_name, lim_vals in limits.items():
                if param_name not in self.param_names:
                    raise AttributeError(f'Invalid parameter: {param_name}')
                if len(lim_vals) != 2:
                    raise ValueError('Invalid limits for parameter: '
                                     f'{param_name}')
            limits = [[limits[param_name][0].item(),
                       limits[param_name][1].item()]
                      for param_name in self.param_names]
        if subset:
            for param_name in subset:
                if param_name not in self.param_names:
                    raise AttributeError(f'Invalid parameter: {param_name}')
            subset = [self.param_names.index(param_name)
                      for param_name in subset]
        if ticks:
            for param_name, lim_vals in ticks.items():
                if param_name not in self.param_names:
                    raise AttributeError(f'Invalid parameter: {param_name}')
                if len(lim_vals) != 2:
                    raise ValueError('Invalid limits for parameter: '
                                     f'{param_name}')
            ticks = [[ticks[param_name][0].item(),
                      ticks[param_name][1].item()]
                     for param_name in self.param_names]
        else:
            ticks = []
        if labels:
            for param_name in labels.keys():
                if param_name not in self.param_names:
                    raise AttributeError(f'Invalid parameter: {param_name}')
            labels = [labels[param_name] for param_name in self.param_names]
        fig, axes = sbi.analysis.pairplot(samples=s,
                                          points=points,
                                          limits=limits,
                                          subset=subset,
                                          labels=labels,
                                          ticks=ticks,
                                          **kwargs)
        return fig, axes

    def conditional_pairplot(self, condition, density=None, points=None,
                             limits=None, subset=None, labels=None,
                             ticks=None, **kwargs):
        """Plot conditional distribution given all other parameters.

        Check ``sbi.analysis.plot.conditional_pairplot`` for more
        details.

        Parameters
        ----------
        condition : numpy.ndarray
            Condition that all but the one/two regarded parameters are
            fixed to.
        density : sbi.inference.NeuralPosterior, optional
            Posterior probability density.
        points : dict, optional
            Additional points to scatter, e.g., true parameter values,
            if known.
        limits : dict, optional
            Limits for each parameter. If None, min and max of the
            given samples will be used.
        subset : list, optional
            Which parameters to plot.
        labels : dict, optional
            Names for each parameter.
        ticks : dict, optional
            Position of the ticks. If None, default ticks positions
            will be used.
        kwargs : dict, optional
            Additional keyword arguments for the
            ``sbi.analysis.conditional_pairplot`` function.

        Returns
        -------
        tuple
            Figure and axis of conditional pairplot.
        """
        condition = torch.from_numpy(condition)
        if density is not None:
            d = density
        else:
            if self.posterior is not None:
                if self.posterior.default_x is None:
                    x_o = torch.tensor(self.x_o, dtype=torch.float32)
                    d = self.posterior.set_default_x(x_o)
                else:
                    d = self.posterior
            else:
                raise ValueError('Density is not available.')
        if points:
            for param_name in points.keys():
                if param_name not in self.param_names:
                    raise AttributeError(f'Invalid parameter: {param_name}')
            points = np.array([[points[param_name].item()
                               for param_name in self.param_names]])
        if limits:
            for param_name, lim_vals in limits.items():
                if param_name not in self.param_names:
                    raise AttributeError(f'Invalid parameter: {param_name}')
                if len(lim_vals) != 2:
                    raise ValueError('Invalid limits for parameter: '
                                     f'{param_name}')
            limits = [[limits[param_name][0].item(),
                       limits[param_name][1].item()]
                      for param_name in self.param_names]
        else:
            limits = [[self.params[param_name][0].item(),
                       self.params[param_name][1].item()]
                      for param_name in self.param_names]
        if subset:
            for param_name in subset:
                if param_name not in self.param_names:
                    raise AttributeError(f'Invalid parameter: {param_name}')
            subset = [self.param_names.index(param_name)
                      for param_name in subset]
        if ticks:
            for param_name, lim_vals in ticks.items():
                if param_name not in self.param_names:
                    raise AttributeError(f'Invalid parameter: {param_name}')
                if len(lim_vals) != 2:
                    raise ValueError('Invalid limits for parameter: '
                                     f'{param_name}')
            ticks = [[ticks[param_name][0].item(),
                      ticks[param_name][1].item()]
                     for param_name in self.param_names]
        else:
            ticks = []
        if labels:
            for param_name in labels.keys():
                if param_name not in self.param_names:
                    raise AttributeError(f'Invalid parameter: {param_name}')
            labels = [labels[param_name] for param_name in self.param_names]
        fig, axes = sbi.analysis.conditional_pairplot(density=d,
                                                      condition=condition,
                                                      limits=limits,
                                                      points=points,
                                                      subset=subset,
                                                      labels=labels,
                                                      ticks=ticks,
                                                      **kwargs)
        return fig, axes

    def conditional_corrcoeff(self, condition, density=None, limits=None,
                              subset=None, **kwargs):
        """Plot conditional distribution given all other parameters.

        Check ``sbi.analysis.conditional_density.conditional_corrcoeff``
        for more details.

        Parameters
        ----------
        condition : numpy.ndarray
            Condition that all but the one/two regarded parameters are
            fixed to.
        density : sbi.inference.NeuralPosterior, optional
            Posterior probability density.
        limits : dict, optional
            Limits for each parameter. If None, min and max of the
            given samples will be used.
        subset : list, optional
            Parameters that are taken for conditional distribution, if
            None all parameters are considered.
        kwargs : dict, optional
            Additional keyword arguments for the
            ``sbi.analysis.conditional_corrcoeff`` function.

        Returns
        -------
        numpy.ndarray
            Average conditional correlation matrix.
        """
        condition = torch.from_numpy(condition)
        if density is not None:
            d = density
        else:
            if self.posterior is not None:
                if self.posterior.default_x is None:
                    x_o = torch.tensor(self.x_o, dtype=torch.float32)
                    d = self.posterior.set_default_x(x_o)
                else:
                    d = self.posterior
            else:
                raise ValueError('Density is not available.')
        if limits:
            for param_name, lim_vals in limits.items():
                if param_name not in self.param_names:
                    raise AttributeError(f'Invalid parameter: {param_name}')
                if len(lim_vals) != 2:
                    raise ValueError('Invalid limits for parameter: '
                                     f'{param_name}')
            limits = [[limits[param_name][0].item(),
                       limits[param_name][1].item()]
                      for param_name in self.param_names]
        else:
            limits = [[self.params[param_name][0].item(),
                       self.params[param_name][1].item()]
                      for param_name in self.param_names]
        limits = torch.tensor(limits)
        if subset:
            for param_name in subset:
                if param_name not in self.param_names:
                    raise AttributeError(f'Invalid parameter: {param_name}')
            subset = [self.param_names.index(param_name)
                      for param_name in subset]
        cond_coeff = sbi.analysis.conditional_corrcoeff(density=d,
                                                        limits=limits,
                                                        condition=condition,
                                                        subset=subset,
                                                        **kwargs)
        return cond_coeff.numpy()

    def generate_traces(self, posterior=None, output_var=None, param_init=None,
                        level=0):
        """Generates traces for a single drawn sample from the trained
        posterior and all inputs.

        Parameters
        ----------
        posterior : sbi.inference.NeuralPosterior, optional
            Posterior distribution.
        output_var: str or sequence of str
            Name of the output variable to be monitored, it can also be
            a sequence of names to record multiple variables.
        param_init : dict
            Dictionary of initial values for the model.
        level : int, optional
            How far to go back to get the locals/globals.

        Returns
        -------
        brian2.units.fundamentalunits.Quantity or dict
            If a single output variable is observed, 2-D array of
            traces generated by using a set of parameters sampled from
            the trained posterior distribution of shape
            (``n.traces``, number of time steps). Otherwise, a
            dictionary with keys set to names of output variables, and
            values to generated traces of respective output variables.
        """
        # sample a single set of parameters from posterior distribution
        if posterior:
            p = posterior
        elif posterior is None and self.posterior:
            p = self.posterior
        else:
            raise ValueError('Need to provide posterior argument if no'
                             ' posterior has been calculated by the `infere`'
                             ' method.')
        x_o = torch.tensor(self.x_o, dtype=torch.float32)
        params = p.sample((1, ), x=x_o)

        # set output variable that is monitored
        if output_var is None:
            output_var = self.output_var

        # set up initial values
        if param_init is None:
            param_init = self.param_init
        else:
            param_init = dict(self.param_init)
            self.param_init.update(param_init)

        # create a dictionary with repeated sampled prior
        d_param = get_param_dict(params, self.param_names, self.n_traces)

        # set up and run the simulator
        network_name = 'generate_traces'
        simulator = self.setup_simulator('generate_traces',
                                         self.n_traces,
                                         output_var=output_var,
                                         param_init=param_init,
                                         level=level+1)
        simulator.run(self.sim_time, d_param, self.param_names, iteration=0,
                      name=network_name)

        # create dictionary of traces for multiple observed output variables
        if len(output_var) > 1:
            for ov in output_var:
                trace = getattr(simulator.statemonitor, ov)[:]
                traces = {ov: trace}
        else:
            traces = getattr(simulator.statemonitor, output_var[0])[:]
        return traces
