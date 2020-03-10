import abc
import numbers

import sympy
from numpy import ones, array, arange, concatenate, mean, argmin, nanmin, reshape, zeros

from brian2.parsing.sympytools import sympy_to_str, str_to_sympy
from brian2.units.fundamentalunits import DIMENSIONLESS, get_dimensions
from brian2.utils.stringtools import get_identifiers

from brian2 import (NeuronGroup, defaultclock, get_device, Network,
                    StateMonitor, SpikeMonitor, second, get_local_namespace,
                    Quantity, get_logger)
from brian2.input import TimedArray
from brian2.equations.equations import Equations, SUBEXPRESSION
from brian2.devices import set_device, reset_device, device
from brian2.devices.cpp_standalone.device import CPPStandaloneDevice
from brian2.core.functions import Function
from .simulator import RuntimeSimulator, CPPStandaloneSimulator
from .metric import Metric, SpikeMetric, TraceMetric
from .optimizer import Optimizer
from .utils import callback_setup, make_dic


logger = get_logger(__name__)


def get_param_dic(params, param_names, n_traces, n_samples):
    """Transform parameters into a dictionary of appropiate size"""
    params = array(params)

    d = dict()

    for name, value in zip(param_names, params.T):
        d[name] = (ones((n_traces, n_samples)) * value).T.flatten()
    return d


def get_spikes(monitor, n_samples, n_traces):
    """
    Get spikes from spike monitor change format from dict to a list,
    remove units.
    """
    spike_trains = monitor.spike_trains()
    assert len(spike_trains) == n_samples*n_traces
    spikes = []
    i = -1
    for sample in range(n_samples):
        sample_spikes = []
        for trace in range(n_traces):
            i += 1
            sample_spikes.append(array(spike_trains[i], copy=False))
        spikes.append(sample_spikes)
    return spikes


def get_full_namespace(additional_namespace, level=0):
    # Get the local namespace with all the values that could be relevant
    # in principle -- by filtering things out, we avoid circular loops
    namespace = {key: value
                 for key, value in get_local_namespace(level=level + 1).items()
                 if isinstance(value, (Quantity, numbers.Number, Function))}
    namespace.update(additional_namespace)

    return namespace


def setup_fit():
    """
    Function sets up simulator in one of the two availabel modes: runtime
    or standalone.

    Returns
    -------
    simulator : .Simulator
    """
    simulators = {
        'CPPStandaloneDevice': CPPStandaloneSimulator(),
        'RuntimeDevice': RuntimeSimulator()
    }
    if isinstance(get_device(), CPPStandaloneDevice):
        if device.has_been_run is True:
            get_device().reinit()
            get_device().activate()
    return simulators[get_device().__class__.__name__]


def get_sensitivity_equations(group, parameters, namespace=None, level=1,
                              optimize=True):
    """
    Get equations for sensitivity variables.

    Parameters
    ----------
    group : `NeuronGroup`
        The group of neurons that will be simulated.
    parameters : list of str
        Names of the parameters that are fit.
    namespace : dict, optional
        The namespace to use.
    level : `int`, optional
        How much farther to go down in the stack to find the namespace.
    optimize : bool, optional
        Whether to remove sensitivity variables from the equations that do
        not evolve if initialized to zero (e.g. ``dS_x_y/dt = -S_x_y/tau``
        would be removed). This avoids unnecessary computation but will fail
        in the rare case that such a sensitivity variable needs to be
        initialized to a non-zero value. Defaults to ``True``.

    Returns
    -------
    sensitivity_eqs : `Equations`
        The equations for the sensitivity variables.
    """
    if namespace is None:
        namespace = get_local_namespace(level)
        namespace.update(group.namespace)

    eqs = group.equations
    diff_eqs = eqs.get_substituted_expressions(group.variables)
    diff_eq_names = [name for name, _ in diff_eqs]

    system = sympy.Matrix([str_to_sympy(diff_eq[1].code)
                           for diff_eq in diff_eqs])
    J = system.jacobian([str_to_sympy(d) for d in diff_eq_names])

    sensitivity = []
    sensitivity_names = []
    for parameter in parameters:
        F = system.jacobian([str_to_sympy(parameter)])
        names = [str_to_sympy(f'S_{diff_eq_name}_{parameter}')
                 for diff_eq_name in diff_eq_names]
        sensitivity.append(J * sympy.Matrix(names) + F)
        sensitivity_names.append(names)

    new_eqs = []
    for names, sensitivity_eqs, param in zip(sensitivity_names, sensitivity, parameters):
        for name, eq, orig_var in zip(names, sensitivity_eqs, diff_eq_names):
            if param in namespace:
                unit = eqs[orig_var].dim / namespace[param].dim
            elif param in group.variables:
                unit = eqs[orig_var].dim / group.variables[param].dim
            else:
                raise AssertionError(f'Parameter {param} neither in namespace nor variables')
            unit = repr(unit) if not unit.is_dimensionless else '1'
            if optimize:
                # Check if the equation stays at zero if initialized at zero
                zeroed = eq.subs(name, sympy.S.Zero)
                if zeroed == sympy.S.Zero:
                    # No need to include equation as differential equation
                    if unit == '1':
                        new_eqs.append(f'{sympy_to_str(name)} = 0 : {unit}')
                    else:
                        new_eqs.append(f'{sympy_to_str(name)} = 0*{unit} : {unit}')
                    continue
            rhs = sympy_to_str(eq)
            if rhs == '0':  # avoid unit mismatch
                rhs = f'0*{unit}/second'
            new_eqs.append('d{lhs}/dt = {rhs} : {unit}'.format(lhs=sympy_to_str(name),
                                                               rhs=rhs,
                                                               unit=unit))
    new_eqs = Equations('\n'.join(new_eqs))
    return new_eqs


def get_sensitivity_init(group, parameters, param_init):
    """
    Calculate the initial values for the sensitivity parameters (necessary if
    initial values are functions of parameters).

    Parameters
    ----------
    group : `NeuronGroup`
        The group of neurons that will be simulated.
    parameters : list of str
        Names of the parameters that are fit.
    param_init : dict
        The dictionary with expressions to initialize the model variables.

    Returns
    -------
    sensitivity_init : dict
        Dictionary of expressions to initialize the sensitivity
        parameters.
    """
    sensitivity_dict = {}
    for var_name, expr in param_init.items():
        if not isinstance(expr, str):
            continue
        identifiers = get_identifiers(expr)
        for identifier in identifiers:
            if (identifier in group.variables
                    and getattr(group.variables[identifier],
                                'type', None) == SUBEXPRESSION):
                raise NotImplementedError('Initializations that refer to a '
                                          'subexpression are currently not '
                                          'supported')
            sympy_expr = str_to_sympy(expr)
            for parameter in parameters:
                diffed = sympy_expr.diff(str_to_sympy(parameter))
                if diffed != sympy.S.Zero:
                    if getattr(group.variables[parameter],
                               'type', None) == SUBEXPRESSION:
                        raise NotImplementedError('Sensitivity '
                                                  f'S_{var_name}_{parameter} '
                                                  'is initialized to a non-zero '
                                                  'value, but it has been '
                                                  'removed from the equations. '
                                                  'Set optimize=False to avoid '
                                                  'this.')
                    init_expr = sympy_to_str(diffed)
                    sensitivity_dict[f'S_{var_name}_{parameter}'] = init_expr
    return sensitivity_dict


class Fitter(metaclass=abc.ABCMeta):
    """
    Base Fitter class for model fitting applications.

    Creates an interface for model fitting of traces with parameters draw by
    gradient-free algorithms (through ask/tell interfaces).

    Initiates n_neurons = num input traces * num samples, to which drawn
    parameters get assigned and evaluates them in parallel.

    Parameters
    ----------
    dt : `~brian2.units.fundamentalunits.Quantity`
        The size of the time step.
    model : `~brian2.equations.equations.Equations` or str
        The equations describing the model.
    input : `~numpy.ndarray` or `~brian2.units.fundamentalunits.Quantity`
        A 2D array of shape ``(n_traces, time steps)`` given the input that will
        be fed into the model.
    output : `~numpy.ndarray` or `~brian2.units.fundamentalunits.Quantity` or list
        Recorded output of the model that the model should reproduce. Should
        be a 2D array of the same shape as the input when fitting traces with
        `TraceFitter`, a list of spike times when fitting spike trains with
        `SpikeFitter`.
    input_var : str
        The name of the input variable in the model. Note that this variable
        should be *used* in the model (e.g. a variable ``I`` that is added as
        a current in the membrane potential equation), but not *defined*.
    output_var : str
        The name of the output variable in the model. Only needed when fitting
        traces with `.TraceFitter`.
    n_samples: int
        Number of parameter samples to be optimized over in a single iteration.
    threshold: `str`, optional
        The condition which produces spikes. Should be a boolean expression as
        a string.
    reset: `str`, optional
        The (possibly multi-line) string with the code to execute on reset.
    refractory: `str` or `~brian2.units.fundamentalunits.Quantity`, optional
        Either the length of the refractory period (e.g. 2*ms), a string
        expression that evaluates to the length of the refractory period after
        each spike (e.g. '(1 + rand())*ms'), or a string expression evaluating
        to a boolean value, given the condition under which the neuron stays
        refractory after a spike (e.g. 'v > -20*mV')
    method: `str`, optional
        Integration method
    param_init: `dict`, optional
        Dictionary of variables to be initialized with respective values
    """
    def __init__(self, dt, model, input, output, input_var, output_var,
                 n_samples, threshold, reset, refractory, method, param_init):
        """Initialize the fitter."""

        if dt is None:
            raise ValueError("dt-sampling frequency of the input must be set")

        if isinstance(model, str):
            model = Equations(model)
        if input_var not in model.identifiers:
            raise NameError("%s is not an identifier in the model" % input_var)

        self.dt = dt

        self.simulator = None

        self.parameter_names = model.parameter_names
        self.n_traces, n_steps = input.shape
        self.duration = n_steps * dt
        self.n_neurons = self.n_traces * n_samples

        self.n_samples = n_samples
        self.method = method
        self.threshold = threshold
        self.reset = reset
        self.refractory = refractory

        self.input = input
        self.output = array(output)
        self.output_var = output_var
        self.model = model

        input_dim = get_dimensions(input)
        input_dim = '1' if input_dim is DIMENSIONLESS else repr(input_dim)
        input_eqs = "{} = input_var(t, i % n_traces) : {}".format(input_var,
                                                                  input_dim)
        self.model += input_eqs

        input_traces = TimedArray(input.transpose(), dt=dt)
        self.input_traces = input_traces

        # initialization of attributes used later
        self.best_params = None
        self.optimizer = None
        self.metric = None
        if not param_init:
            param_init = {}
        for param, val in param_init.items():
            if not (param in self.model.diff_eq_names or
                    param in self.model.parameter_names):
                raise ValueError("%s is not a model variable or a "
                                 "parameter in the model" % param)
        self.param_init = param_init

    def setup_simulator(self, network_name, n_neurons, output_var, param_init,
                        calc_gradient=False, optimize=True, level=1):
        simulator = setup_fit()

        namespace = get_full_namespace({'input_var': self.input_traces,
                                        'n_traces': self.n_traces},
                                       level=level+1)
        if network_name != 'generate':
            namespace['output_var'] = TimedArray(self.output.transpose(),
                                                 dt=self.dt)
        neurons = self.setup_neuron_group(n_neurons, namespace,
                                          calc_gradient=calc_gradient,
                                          optimize=optimize)

        if output_var == 'spikes':
            monitor = SpikeMonitor(neurons, name='monitor')
        else:
            record_vars = [output_var]
            if calc_gradient:
                record_vars.extend([f'S_{output_var}_{p}'
                                    for p in self.parameter_names])
            monitor = StateMonitor(neurons, record_vars, record=True,
                                   name='monitor', dt=self.dt)

        network = Network(neurons, monitor)

        simulator.initialize(network, param_init, name=network_name)
        return simulator

    def setup_neuron_group(self, n_neurons, namespace, calc_gradient=False,
                           optimize=True, name='neurons'):
        """
        Setup neuron group, initialize required number of neurons, create
        namespace and initialize the parameters.

        Parameters
        ----------
        n_neurons: int
            number of required neurons
        **namespace
            arguments to be added to NeuronGroup namespace

        Returns
        -------
        neurons : ~brian2.groups.neurongroup.NeuronGroup
            group of neurons

        """
        # We only want to specify the method argument if it is not None –
        # otherwise it should use NeuronGroup's default value
        kwds = {}
        if self.method is not None:
            kwds['method'] = self.method
        neurons = NeuronGroup(n_neurons, self.model,
                              threshold=self.threshold, reset=self.reset,
                              refractory=self.refractory, name=name,
                              namespace=namespace, dt=self.dt, **kwds)
        if calc_gradient:
            sensitivity_eqs = get_sensitivity_equations(neurons,
                                                        parameters=self.parameter_names,
                                                        optimize=optimize,
                                                        namespace=namespace)
            neurons = NeuronGroup(n_neurons, self.model + sensitivity_eqs,
                                  threshold=self.threshold, reset=self.reset,
                                  refractory=self.refractory, name=name,
                                  namespace=namespace, dt=self.dt, **kwds)
        return neurons

    @abc.abstractmethod
    def calc_errors(self, metric):
        """
        Abstract method required in all Fitter classes, used for
        calculating errors

        Parameters
        ----------
        metric: `~.Metric` children
            Child of Metric class, specifies optimization metric
        """
        pass

    def optimization_iter(self, optimizer, metric):
        """
        Function performs all operations required for one iteration of
        optimization. Drawing parameters, setting them to simulator and
        calulating the error.

        Returns
        -------
        results : list
            recommended parameters
        parameters: list of list
            drawn parameters
        errors: list
            calculated errors
        """
        parameters = optimizer.ask(n_samples=self.n_samples)

        d_param = get_param_dic(parameters, self.parameter_names,
                                self.n_traces, self.n_samples)
        self.simulator.run(self.duration, d_param, self.parameter_names)

        errors = self.calc_errors(metric)

        optimizer.tell(parameters, errors)

        results = optimizer.recommend()

        return results, parameters, errors

    def fit(self, optimizer, metric=None, n_rounds=1, callback='text',
            restart=False, level=0, **params):
        """
        Run the optimization algorithm for given amount of rounds with given
        number of samples drawn. Return best set of parameters and
        corresponding error.

        Parameters
        ----------
        optimizer: `~.Optimizer` children
            Child of Optimizer class, specific for each library.
        metric: `~.Metric` children
            Child of Metric class, specifies optimization metric
        n_rounds: int
            Number of rounds to optimize over (feedback provided over each
            round).
        callback: `str` or `~typing.Callable`
            Either the name of a provided callback function (``text`` or
            ``progressbar``), or a custom feedback function
            ``func(parameters, errors, best_parameters, best_error, index)``.
            If this function returns ``True`` the fitting execution is
            interrupted.
        restart: bool
            Flag that reinitializes the Fitter to reset the optimization.
            With restart True user is allowed to change optimizer/metric.
        **params
            bounds for each parameter
        level : `int`, optional
            How much farther to go down in the stack to find the namespace.

        Returns
        -------
        best_results : dict
            dictionary with best parameter set
        error: float
            error value for best parameter set
        """
        if not (isinstance(metric, Metric) or metric is None):
            raise TypeError("metric has to be a child of class Metric or None "
                            "for OnlineTraceFitter")

        if not (isinstance(optimizer, Optimizer)) or optimizer is None:
            raise TypeError("metric has to be a child of class Optimizer")

        if self.metric is not None and restart is False:
            if metric is not self.metric:
                raise Exception("You can not change the metric between fits")

        if self.optimizer is not None and restart is False:
            if optimizer is not self.optimizer:
                raise Exception("You can not change the optimizer between fits")

        if self.optimizer is None or restart is True:
            optimizer.initialize(self.parameter_names, popsize=self.n_samples,
                                 **params)

        self.optimizer = optimizer
        self.metric = metric

        callback = callback_setup(callback, n_rounds)

        # Check whether we can reuse the current simulator or whether we have
        # to create a new one (only relevant for standalone, but does not hurt
        # for runtime)
        if self.simulator is None or self.simulator.current_net != 'fit':
            self.simulator = self.setup_simulator('fit', self.n_neurons,
                                                  output_var=self.output_var,
                                                  param_init=self.param_init,
                                                  level=level+1)

        # Run Optimization Loop
        error = None
        for index in range(n_rounds):
            best_params, parameters, errors = self.optimization_iter(optimizer,
                                                                     metric)

            # create output variables
            self.best_params = make_dic(self.parameter_names, best_params)
            error = nanmin(self.optimizer.errors)
            param_dicts = [{p: v for p, v in zip(self.parameter_names,
                                                one_param_set)}
                          for one_param_set in parameters]

            if callback(param_dicts, errors, self.best_params, error, index) is True:
                break

        return self.best_params, error

    def results(self, format='list'):
        """
        Returns all of the gathered results (parameters and errors).
        In one of the 3 formats: 'dataframe', 'list', 'dict'.

        Parameters
        ----------
        format: str
            The desired output format. Currently supported: ``dataframe``,
            ``list``, or ``dict``.

        Returns
        -------
        object
            'dataframe': returns pandas `~pandas.DataFrame` without units
            'list': list of dictionaries
            'dict': dictionary of lists
        """
        names = list(self.parameter_names)
        names.append('errors')

        params = array(self.optimizer.tested_parameters)
        params = params.reshape(-1, params.shape[-1])

        errors = array([array(self.optimizer.errors).flatten()])
        data = concatenate((params, errors.transpose()), axis=1)
        dim = self.model.dimensions

        if format == 'list':
            res_list = []
            for j in arange(0, len(params)):
                temp_data = data[j]
                res_dict = dict()

                for i, n in enumerate(names[:-1]):
                    res_dict[n] = Quantity(temp_data[i], dim=dim[n])
                res_dict[names[-1]] = temp_data[-1]
                res_list.append(res_dict)

            return res_list

        elif format == 'dict':
            res_dict = dict()
            for i, n in enumerate(names[:-1]):
                res_dict[n] = Quantity(data[:, i], dim=dim[n])

            res_dict[names[-1]] = data[:, -1]
            return res_dict

        elif format == 'dataframe':
            from pandas import DataFrame
            return DataFrame(data=data, columns=names)

    def generate(self, params=None, output_var=None, param_init=None, level=0):
        """
        Generates traces for best fit of parameters and all inputs.
        If provided with other parameters provides those.

        Parameters
        ----------
        params: dict
            Dictionary of parameters to generate fits for.
        output_var: str
            Name of the output variable to be monitored.
        param_init: dict
            Dictionary of initial values for the model.
        level : `int`, optional
            How much farther to go down in the stack to find the namespace.
        """
        if params is None:
            params = self.best_params
        if param_init is None:
            param_init = self.param_init
        else:
            param_init = dict(self.param_init)
            self.param_init.update(param_init)
        if output_var is None:
            output_var = self.output_var

        self.simulator = self.setup_simulator('generate', self.n_traces,
                                              output_var=output_var,
                                              param_init=param_init,
                                              level=level+1)
        param_dic = get_param_dic([params[p] for p in self.parameter_names],
                                  self.parameter_names, self.n_traces, 1)
        self.simulator.run(self.duration, param_dic, self.parameter_names,
                           name='generate')

        if output_var == 'spikes':
            fits = get_spikes(self.simulator.monitor,
                              1, self.n_traces)[0]  # a single "sample"
        else:
            fits = getattr(self.simulator.monitor, output_var)

        return fits


class TraceFitter(Fitter):
    """Input nad output have to have the same dimensions."""
    def __init__(self, model, input_var, input, output_var, output, dt,
                 n_samples=30, method=None, reset=None, refractory=False,
                 threshold=None, param_init=None):
        """Initialize the fitter."""
        super().__init__(dt, model, input, output, input_var, output_var,
                         n_samples, threshold, reset, refractory, method,
                         param_init)
        # We store the bounds set in TraceFitter.fit, so that Tracefitter.refine
        # can reuse them
        self.bounds = None

        if output_var not in self.model.names:
            raise NameError("%s is not a model variable" % output_var)
        if output.shape != input.shape:
            raise ValueError("Input and output must have the same size")

    def calc_errors(self, metric):
        """
        Returns errors after simulation with StateMonitor.
        To be used inside optim_iter.
        """
        traces = getattr(self.simulator.networks['fit']['monitor'],
                         self.output_var+'_')
        # Reshape traces for easier calculation of error
        traces = reshape(traces, (traces.shape[0]//self.n_traces,
                                  self.n_traces,
                                  -1))
        errors = metric.calc(traces, self.output, self.dt)
        return errors

    def fit(self, optimizer, metric=None, n_rounds=1, callback='text',
            restart=False, level=0, **params):
        if not isinstance(metric, TraceMetric):
            raise TypeError("You can only use TraceMetric child metric with "
                            "TraceFitter")
        self.bounds = dict(params)
        self.best_params, error = super().fit(optimizer, metric, n_rounds,
                                              callback, restart, level=level+1,
                                              **params)
        return self.best_params, error

    def generate_traces(self, params=None, param_init=None, level=0):
        """Generates traces for best fit of parameters and all inputs"""
        fits = self.generate(params=params, output_var=self.output_var,
                             param_init=param_init, level=level+1)
        return fits

    def refine(self, params=None, t_start=None, normalization=None,
               callback='text', calc_gradient=False, optimize=True,
               level=0, **kwds):
        """
        Refine the fitting results with a sequentially operating minimization
        algorithm. Uses the `lmfit <https://lmfit.github.io/lmfit-py/>`_
        package which itself makes use of
        `scipy.optimize <https://docs.scipy.org/doc/scipy/reference/optimize.html>`_.
        Has to be called after `~.TraceFitter.fit`, but a call with
        ``n_rounds=0`` is enough.

        Parameters
        ----------
        params : dict, optional
            A dictionary with the parameters to use as a starting point for the
            refinement. If not given, the best parameters found so far by
            `~.TraceFitter.fit` will be used.
        t_start : `~brian2.units.fundamentalunits.Quantity`, optional
            Initial simulation/model time that should be ignored for the error
            calculation. If not set, will reuse the `t_start` value from the
            previously used metric.
        normalization : float, optional
            A normalization term that will be used rescale results before
            handing them to the optimization algorithm. Can be useful if the
            algorithm makes assumptions about the scale of errors, e.g. if the
            size of steps in the parameter space depends on the absolute value
            of the error. The difference between simulated and target traces
            will be divided by this value. If not set, will reuse the
            `normalization` value from the previously used metric.
        callback: `str` or `~typing.Callable`
            Either the name of a provided callback function (``text`` or
            ``progressbar``), or a custom feedback function
            ``func(parameters, errors, best_parameters, best_error, index)``.
            If this function returns ``True`` the fitting execution is
            interrupted.
        calc_gradient: bool, optional
            Whether to add "sensitivity variables" to the equation that track
            the sensitivity of the equation variables to the parameters. This
            information will be used to pass the local gradient of the error
            with respect to the parameters to the optimization function. This
            can lead to much faster convergence than with an estimated gradient
            but comes at the expense of additional computation. Defaults to
            ``False``.
        optimize : bool, optional
            Whether to remove sensitivity variables from the equations that do
            not evolve if initialized to zero (e.g. ``dS_x_y/dt = -S_x_y/tau``
            would be removed). This avoids unnecessary computation but will fail
            in the rare case that such a sensitivity variable needs to be
            initialized to a non-zero value. Only taken into account if
            ``calc_gradient`` is ``True``. Defaults to ``True``.
        level : int, optional
            How much farther to go down in the stack to find the namespace.
        kwds
            Additional arguments can overwrite the bounds for individual
            parameters (if not given, the bounds previously specified in the
            call to `~.TraceFitter.fit` will be used). All other arguments will
            be passed on to `.lmfit.minimize` and can be used to e.g. change the
            method, or to specify method-specific arguments.

        Returns
        -------
        parameters : dict
            The parameters at the end of the optimization process as a
            dictionary.
        result : `.lmfit.MinimizerResult`
            The result of the optimization process.

        Notes
        -----
        The default method used by `lmfit` is least-squares minimization using
        a Levenberg-Marquardt method. Note that there is no support for
        specifying a `Metric`, the given output trace(s) will be subtracted
        from the simulated trace(s) and passed on to the minimization algorithm.

        This method always uses the runtime mode, independent of the selection
        of the current device.
        """
        try:
            import lmfit
        except ImportError:
            raise ImportError('Refinement needs the "lmfit" package.')
        if params is None:
            if self.best_params is None:
                raise TypeError('You need to either specify parameters or run '
                                'the fit function first.')
            params = self.best_params

        if t_start is None:
            t_start = getattr(self.metric, 't_start', 0*second)
        if normalization is None:
            normalization = getattr(self.metric, 'normalization', 1.)
        else:
            normalization = 1/normalization

        callback_func = callback_setup(callback, None)

        # Set up Parameter objects
        parameters = lmfit.Parameters()
        for param_name in self.parameter_names:
            if param_name not in kwds:
                if self.bounds is None:
                    raise TypeError('You need to either specify bounds for all '
                                    'parameters or run the fit function first.')
                min_bound, max_bound = self.bounds[param_name]
            else:
                min_bound, max_bound = kwds.pop(param_name)
            parameters.add(param_name, value=array(params[param_name]),
                           min=array(min_bound), max=array(max_bound))

        self.simulator = self.setup_simulator('refine', self.n_traces,
                                              output_var=self.output_var,
                                              param_init=self.param_init,
                                              calc_gradient=calc_gradient,
                                              optimize=optimize,
                                              level=level+1)

        t_start_steps = int(round(t_start / self.dt))

        def _calc_error(params):
            param_dic = get_param_dic([params[p] for p in self.parameter_names],
                                      self.parameter_names, self.n_traces, 1)
            self.simulator.run(self.duration, param_dic,
                               self.parameter_names, name='refine')
            trace = getattr(self.simulator.monitor, self.output_var+'_')
            residual = trace[:, t_start_steps:] - self.output[:, t_start_steps:]
            return residual.flatten() * normalization

        def _calc_gradient(params):
            residuals = []
            for name in self.parameter_names:
                trace = getattr(self.simulator.monitor,
                                f'S_{self.output_var}_{name}_')
                residual = trace[:, t_start_steps:]
                residuals.append(residual.flatten() * normalization)
            gradient = array(residuals)
            return gradient.T

        tested_parameters = []
        errors = []
        def _callback_wrapper(params, iter, resid, *args, **kwds):
            error = mean(resid**2)
            params =  {p: float(val) for p, val in params.items()}
            tested_parameters.append(params)
            errors.append(error)
            best_idx = argmin(errors)
            best_error = errors[best_idx]
            best_params = tested_parameters[best_idx]
            return callback_func(params, errors, best_params, best_error, iter)

        assert 'Dfun' not in kwds
        if calc_gradient:
            kwds.update({'Dfun': _calc_gradient})
        if 'iter_cb' in kwds:
            # Use the given callback but raise a warning if callback is not
            # set to None
            if callback is not None:
                logger.warn('The iter_cb keyword has been specified together '
                            f'with callback={callback!r}. Only the iter_cb '
                            'callback will be used. Use the standard '
                            'callback mechanism or set callback=None to '
                            'remove this warning.',
                            name_suffix='iter_cb_callback')
            iter_cb = kwds.pop('iter_cb')
        else:
            iter_cb = _callback_wrapper
        result = lmfit.minimize(_calc_error, parameters,
                                iter_cb=iter_cb,
                                **kwds)

        return {p: float(val) for p, val in result.params.items()}, result


class SpikeFitter(Fitter):
    def __init__(self, model, input, output, dt, reset, threshold,
                 input_var='I', refractory=False, n_samples=30,
                 method=None, level=0, param_init=None):
        """Initialize the fitter."""
        if method is None:
            method = 'exponential_euler'
        super().__init__(dt, model, input, output, input_var, 'v',
                         n_samples, threshold, reset, refractory, method,
                         param_init)
        self.output_var = 'spikes'

        if param_init:
            for param, val in param_init.items():
                if not (param in self.model.identifiers or param in self.model.names):
                    raise ValueError("%s is not a model variable or an "
                                     "identifier in the model" % param)
            self.param_init = param_init

        self.simulator = None

    def calc_errors(self, metric):
        """
        Returns errors after simulation with SpikeMonitor.
        To be used inside optim_iter.
        """
        spikes = get_spikes(self.simulator.networks['fit']['monitor'],
                            self.n_samples, self.n_traces)
        errors = metric.calc(spikes, self.output, self.dt)
        return errors

    def fit(self, optimizer, metric=None, n_rounds=1, callback='text',
            restart=False, level=0, **params):
        if not isinstance(metric, SpikeMetric):
            raise TypeError("You can only use SpikeMetric child metric with "
                            "SpikeFitter")
        self.best_params, error = super().fit(optimizer, metric, n_rounds,
                                              callback, restart, level=level+1,
                                              **params)
        return self.best_params, error

    def generate_spikes(self, params=None, param_init=None, level=0):
        """Generates traces for best fit of parameters and all inputs"""
        fits = self.generate(params=params, output_var='spikes',
                             param_init=param_init, level=level+1)
        return fits


class OnlineTraceFitter(Fitter):
    """Input nad output have to have the same dimensions."""
    def __init__(self, model, input_var, input, output_var, output, dt,
                 n_samples=30,  method=None, reset=None, refractory=False,
                 threshold=None, level=0, param_init=None, t_start=0*second):
        """Initialize the fitter."""
        super().__init__(dt, model, input, output, input_var, output_var,
                         n_samples, threshold, reset, refractory, method,
                         param_init)

        if output_var not in self.model.names:
            raise NameError("%s is not a model variable" % output_var)
        if output.shape != input.shape:
            raise ValueError("Input and output must have the same size")

        # Replace input variable by TimedArray
        output_traces = TimedArray(output.transpose(), dt=dt)
        output_dim = get_dimensions(output)
        squared_output_dim = ('1' if output_dim is DIMENSIONLESS
                              else repr(output_dim**2))
        error_eqs = Equations('total_error : {}'.format(squared_output_dim))
        self.model = self.model + error_eqs

        self.t_start = t_start

        if param_init:
            for param, val in param_init.items():
                if not (param in self.model.identifiers or param in self.model.names):
                    raise ValueError("%s is not a model variable or an "
                                     "identifier in the model" % param)
            self.param_init = param_init

        self.simulator = None

    def calc_errors(self, metric=None):
        """Calculates error in online fashion.To be used inside optim_iter."""
        errors = self.simulator.neurons.total_error/int((self.duration-self.t_start)/defaultclock.dt)
        errors = mean(errors.reshape((self.n_samples, self.n_traces)), axis=1)
        return array(errors)

    def generate_traces(self, params=None, param_init=None, level=0):
        """Generates traces for best fit of parameters and all inputs"""
        fits = self.generate(params=params, output_var=self.output_var,
                             param_init=param_init, level=level+1)
        return fits
