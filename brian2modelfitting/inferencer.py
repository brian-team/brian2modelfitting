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
    `.RuntimeSimulator`, object for use with `.RuntimeDevice`, or
    `.CPPStandaloneSimulator`, object for use with
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
        Iterable of size (``n_samples``, ``len(param_names)``)
        containing parameter values.
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
        ``sbi`` compatible object that contains a uniform prior
        distribution over a given set of parameter
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

    It offers an interface similar to that of `.Fitter` class but
    instead of fitting, neural density estimator is trained using a
    generative model. This class serves as a wrapper for ``sbi``
    library for inferencing posterior over unknown parameters of a
    given model.

    Parameters
    ----------
    dt : brian2.units.fundamentalunits.Quantity
        Integration time step.
    model : str or brian2.equations.equations.Equations
        Single cell model equations.
    input : dict
        Input traces in dictionary format, where key corresponds to the
        name of the input variable as defined in ``model`` and value
        corresponds to a single dimensional array of data traces.
    output : dict
        Dictionary of recorded (or simulated) output data traces, where
        key corresponds to the name of the output variable as defined
        in ``model`` and value corresponds to a single dimensional
        array of recorded data traces.
    features : list
        List of callables that take the voltage trace and output
        summary statistics.
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
    def __init__(self, dt, model, input, output, features, method=None,
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

        # placeholder for samples
        self.n_samples = None
        self.samples = None
        # placeholder for the posterior
        self.posterior = None
        # observation the focus is on
        x_o = []
        for o in self.output:
            o = np.array(o)
            obs = []
            for feature in features:
                obs.extend(feature(o.transpose()))
            x_o.append(obs)
        x_o = torch.tensor(x_o, dtype=torch.float32)
        self.x_o = x_o
        self.features = features
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
        return prior

    def generate_training_data(self, n_samples, prior):
        """Return sampled prior and executed simulator containing
        recorded variables to be used for training the neural density
        estimator.

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

        # sample from prior
        theta = prior.sample((n_samples, ))
        theta = np.atleast_2d(theta.numpy())

        return theta

    def extract_summary_statistics(self, theta, level=1):
        """Return summary statistics to be used for training the neural
        density estimator.

        Parameters
        ----------
        theta : numpy.ndarray
            Sampled prior of shape (``n_samples``, -1).
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

        # extract features
        obs = simulator.statemonitor.recorded_variables
        x = []
        for ov in self.output_var:
            x_val = obs[ov].get_value()
            summary_statistics = []
            for feature in self.features:
                summary_statistics.append(feature(x_val))
            x.append(summary_statistics)
        x = np.array(x, dtype=np.float32)
        x = x.reshape((self.n_samples, -1))
        return x

    def save_summary_statistics(self, f, theta=None, x=None):
        """Save sampled prior data, theta, and extracted summary
        statistics, x, into a single file in compressed ``.pz`` format.

        Parameters
        ----------
        f : str or os.PathLike
            Path to file either as string or ``os.PathLike`` object
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
                                 ' `infere_step` method first.')
        if x is not None:
            pass
        elif self.x is not None:
            x = self.x
        else:
            raise AttributeError('Provide summary feautures or call '
                                 ' `infere_step` method first.')
        np.savez_compressed(f, theta=t, x=x)

    def load_summary_statistics(self, f, **kwargs):
        """Load sampled prior data, theta, and extracted summary
        statistics, x, from a compressed ``.npz`` format. Arrays should
        be named either `arr_0` and `arr_1` or `theta` and `x` for
        sampled priors and extracted features, respectively.

        Parameters
        ----------
        f : str or os.PathLike
            Path to file either as string or ``os.PathLike`` object
            that contains file name.
        kwargs : dict, optional
            Additional keyword arguments for ``numpy.load`` method.

        Returns
        -------
        tuple
            Consisting of sampled prior and summary statistics arrays.
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
            ``sbi.utils.get_nn_models.posterior_nn`` method.

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
        """Return inference object with stored training data and
        trained density estimator.

        Parameters
        ----------
        inference : sbi.inference.NeuralInference
            Instantiated inference object with stored paramaters and
            simulation outputs prepared for training.
        theta : torch.tensor
            Sampled prior.
        x : torch.tensor
            Summary statistics.
        args : list, optional
            Contains a uniformly distributed sbi.utils.BoxUniform
            prior/proposal. Used only for SNPE, for SNLE and SNRE,
            ``proposal`` should not be passed to ``append_simulations``
            method, thus ``args`` should not be passed.
        train_kwargs : dict, optional
            Additional keyword arguments for ``train`` method of
            ``sbi.inference.NeuralInference`` object.

        Returns
        -------
        tuple
            ``sbi.inference.NeuralInference`` object with stored
            paramaters and simulation outputs prepared for training and
            trained neural density estimator object.
        """
        inference = inference.append_simulations(theta, x, *args)
        density_estimator = inference.train(**train_kwargs)
        return (inference, density_estimator)

    def build_posterior(self, inference, density_estimator,
                        **posterior_kwargs):
        """Return instantiated inference object.

        Parameters
        ----------
        inference : sbi.inference.NeuralInference
            Instantiated inference object with stored paramaters and
            simulation outputs prepared for training.
        theta : torch.tensor
            Sampled prior.
        x : torch.tensor
            Summary statistics.
        args : list, optional
            Contains a uniformly distributed sbi.utils.BoxUniform
            prior/proposal. Used only for SNPE, for SNLE and SNRE,
            ``proposal`` should not be passed to ``append_simulations``
            method, thus ``args`` should not be passed.
        posterior_kwargs : dict, optional
            Additional keyword arguments for ``build_posterior`` method
            of ``sbi.inference.NeuralInference`` object.

        Returns
        -------
        tuple
            ``sbi.inference.NeuralInference`` object with stored
            paramaters and simulation outputs prepared for training and
            ``sbi.inference.NeuralInference`` object from which.
        """
        posterior = inference.build_posterior(density_estimator,
                                              **posterior_kwargs)
        return (inference, posterior)

    def infere_step(self, proposal, inference,
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
        inference, density_estimator = self.train(inference,
                                                  theta, x,
                                                  *args, **train_kwargs)

        # use the density estimator to build the posterior
        inference, posterior = self.build_posterior(inference,
                                                    density_estimator,
                                                    **posterior_kwargs)
        self.inference = inference
        self.posterior = posterior
        return posterior

    def infere(self, n_samples=None, theta=None, x=None, n_rounds=1,
               inference_method='SNPE', density_estimator_model='maf',
               inference_kwargs={}, train_kwargs={}, posterior_kwargs={},
               **params):
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
        inference_method : str, optional
            Inference method. Either of SNPE, SNLE or SNRE.
        density_estimator_model : str, optional
            The type of density estimator to be created. Either
            ``mdn``, ``made``, ``maf`` or ``nsf``.
        inference_kwargs : dict, optional
            Additional keyword arguments for
            ``sbi.utils.get_nn_models.posterior_nn`` method.
        train_kwargs : dict, optional
            Additional keyword arguments for training the posterior
            estimator.
        posterior_kwargs : dict, optional
            Dictionary of arguments for `.build_posterior` method.
        params : dict
            Bounds for each parameter.

        Returns
        -------
        sbi.inference.NeuralPosterior
            Trained posterior.
        """
        if self.posterior is None:  # `.infere_step` has not been called
            # handle the number of rounds
            if not isinstance(n_rounds, int):
                raise ValueError('`n_rounds` has to be a positive integer.')

            # handle inference methods
            try:
                inference_method = str.upper(inference_method)
            except ValueError as e:
                print(e, '\nInvalid inference method.')
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
                    x = self.extract_summary_statistics(theta)
            self.x = x

            # initialize inference object
            self.inference = self.init_inference(inference_method,
                                                 density_estimator_model,
                                                 prior,
                                                 **inference_kwargs)

            # additional args for `.train` method are needed only for SNPE
            if inference_method == 'SNPE':
                args = [prior]
            else:
                args = []
        else:  # `.infere_step` has been called manually
            prior = self.posterior.set_default_x(self.x_o)
            if self.posterior._method_family == 'snpe':
                args = [prior]
            else:
                args = []

        # allocate empty list of posterior
        posteriors = []

        # set a proposal
        proposal = prior

        # main inference loop
        for round in range(n_rounds):
            print(f'Round {round + 1}/{n_rounds}.')

            # inference step
            posterior = self.infere_step(proposal, self.inference,
                                         n_samples, self.theta, self.x,
                                         train_kwargs, posterior_kwargs, *args)

            # append the current posterior to the list of posteriors
            posteriors.append(posterior)

            # update the proposal given the observation
            if n_rounds > 1:
                proposal = posterior.set_default_x(self.x_o)
        self.posterior = posterior
        return posterior

    def save_posterior(self, f):
        """Saves the density estimator state dictionary to a disk file.

        Parameters
        ----------
        posterior : sbi.inference.posteriors.DirectPosterior, optional
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
        posterior : sbi.inference.posteriors.DirectPosterior, optional
            Posterior distribution.
        **kwargs : dict, optional
            Additional keyword arguments for ``sample`` method in
            ``sbi.inference.posteriors.DirectPosterior`` class
        Returns
        -------
        torch.tensor
            Samples from posterior of the shape as given in ``shape``.
        """
        if posterior:
            p = posterior
        elif posterior is None and self.posterior:
            p = self.posterior
        else:
            raise ValueError('Need to provide posterior argument if no'
                             ' posterior has been calculated by the `infere`'
                             ' method.')
        samples = p.sample(shape, x=self.x_o, **kwargs)
        self.samples = samples
        return samples

    def pairplot(self, samples=None, **kwargs):
        """Plot samples in a 2-D grid with marginals and pairwise
        marginals.

        Check ``sbi.analysis.plot.pairplot`` for more details.

        Parameters
        ----------
        samples : iterable, optional
            Samples used to build the pairplot.
        **kwargs : dict, optional
            Additional keyword arguments for the
            ``sbi.analysis.plot.pairplot`` function.

        Returns
        -------
        tuple
            Figure and axis of posterior distribution plot.
        """
        if samples is not None:
            s = samples
        else:
            try:
                s = self.samples
            except AttributeError as e:
                print(e, '\nProvide samples or call `sample` method first.')
                raise
        fig, axes = sbi.analysis.pairplot(s, **kwargs)
        return fig, axes

    def conditional_pairplot(self, condition, limits, density=None, **kwargs):
        """Plot conditional distribution given all other parameters.

        Check ``sbi.analysis.plot.conditional_pairplot`` for more
        details.

        Parameters
        ----------
        condition : torch.tensor
            Condition that all but the one/two regarded parameters are
            fixed to.
        limits : list or torch.tensor
            Limits in between which each parameter will be evaulated.
        density : sbi.inference.NeuralPosterior, optional
            Posterior probability density.
        **kwargs : dict, optional
            Additional keyword arguments for the
            ``sbi.analysis.plot.pairplot`` function.

        Returns
        -------
        tuple
            Figure and axis of posterior distribution plot.
        """
        if density is not None:
            d = density
        else:
            try:
                d = self.posterior
            except AttributeError as e:
                print(e, '\nDensity is not available.')
                raise
        fig, axes = sbi.analysis.conditional_pairplot(density=d,
                                                      condition=condition,
                                                      limits=limits,
                                                      *kwargs)
        return fig, axes

    def conditional_corrcoeff(self, condition, limits, density=None, **kwargs):
        """Plot conditional distribution given all other parameters.

        Check ``sbi.analysis.conditional_density.conditional_corrcoeff``
        for more details.

        Parameters
        ----------
        condition : torch.tensor
            Condition that all but the one/two regarded parameters are
            fixed to.
        limits : list or torch.tensor
            Limits in between which each parameter will be evaulated.
        density : sbi.inference.NeuralPosterior, optional
            Posterior probability density.
        **kwargs : dict, optional
            Additional keyword arguments for the
            ``sbi.analysis.plot.pairplot`` function.

        Returns
        -------
        tuple
            Figure and axis of posterior distribution plot.
        """
        if density is not None:
            d = density
        else:
            try:
                d = self.posterior
            except AttributeError as e:
                print(e, '\nDensity is not available.')
                raise
        fig, axes = sbi.analysis.conditional_corrcoeff(density=d,
                                                       condition=condition,
                                                       limits=limits,
                                                       *kwargs)
        return fig, axes

    def generate_traces(self, posterior=None, output_var=None, param_init=None,
                        level=0):
        """Generates traces for a single drawn sample from the trained
        posterior and all inputs.

        Parameters
        ----------
        posterior : sbi.inference.posteriors.DirectPosterior, optional
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
        params = p.sample((1, ), x=self.x_o)

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
