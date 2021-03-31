import abc
import numbers

import sympy
from numpy import ones, array, arange, concatenate, mean, argmin, nanmin, reshape, zeros, sqrt, ndarray, broadcast_to

from brian2.parsing.sympytools import sympy_to_str, str_to_sympy
from brian2.units.fundamentalunits import DIMENSIONLESS, get_dimensions, fail_for_dimension_mismatch
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
from .metric import Metric, SpikeMetric, TraceMetric, MSEMetric, normalize_weights
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
    Function sets up simulator in one of the two available modes: runtime
    or standalone. The `.Simulator` that will be used depends on the currently
    set `.Device`. In the case of `.CPPStandaloneDevice`, the device will also
    be reset if it has already run a simulation.

    Returns
    -------
    simulator : `.Simulator`
    """
    simulators = {
        'CPPStandaloneDevice': CPPStandaloneSimulator(),
        'RuntimeDevice': RuntimeSimulator()
    }
    if isinstance(get_device(), CPPStandaloneDevice):
        if device.has_been_run is True:
            build_options = dict(device.build_options)
            get_device().reinit()
            get_device().activate(**build_options)
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
    output : `~brian2.units.fundamentalunits.Quantity` or list
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
    penalty : str, optional
        The name of a variable or subexpression in the model that will be
        added to the error at the end of each iteration. Note that only
        the term at the end of the simulation is taken into account, so
        this term should not be varying in time.
    param_init: `dict`, optional
        Dictionary of variables to be initialized with respective values
    """
    def __init__(self, dt, model, input, output, input_var, output_var,
                 n_samples, threshold, reset, refractory, method, param_init,
                 penalty, use_units=True):
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
        # Sample size requested by user
        self._n_samples = n_samples
        # Actual sample size used (set in fit())
        self.n_samples = n_samples
        self.method = method
        self.threshold = threshold
        self.reset = reset
        self.refractory = refractory
        self.penalty = penalty

        self.input = input
        self.output_var = output_var
        if output_var == 'spikes':
            self.output_dim = DIMENSIONLESS
        else:
            self.output_dim = model[output_var].dim
            fail_for_dimension_mismatch(output, self.output_dim,
                                        'The provided target values '
                                        '("output") need to have the same '
                                        'units as the variable '
                                        '{}'.format(output_var))
        self.model = model

        self.use_units = use_units
        self.iteration = 0
        input_dim = get_dimensions(input)
        input_dim = '1' if input_dim is DIMENSIONLESS else repr(input_dim)
        input_eqs = "{} = input_var(t, i % n_traces) : {}".format(input_var,
                                                                  input_dim)
        self.model += input_eqs

        if output_var != 'spikes':
            # For approaches that couple the system to the target values,
            # provide a convenient variable
            output_expr = 'output_var(t, i % n_traces)'
            output_dim = ('1' if self.output_dim is DIMENSIONLESS
                          else repr(self.output_dim))
            output_eqs = "{}_target = {} : {}".format(output_var,
                                                      output_expr,
                                                      output_dim)
            self.model += output_eqs

        input_traces = TimedArray(input.transpose(), dt=dt)
        self.input_traces = input_traces

        # initialization of attributes used later
        self._best_params = None
        self._best_error = None
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

    @property
    def n_neurons(self):
        return self.n_traces * self.n_samples

    def setup_simulator(self, network_name, n_neurons, output_var, param_init,
                        calc_gradient=False, optimize=True, online_error=False,
                        level=1):
        simulator = setup_fit()

        namespace = get_full_namespace({'input_var': self.input_traces,
                                        'n_traces': self.n_traces},
                                       level=level+1)
        if hasattr(self, 't_start'):  # OnlineTraceFitter
            namespace['t_start'] = self.t_start

        if self.output_var != 'spikes':
            namespace['output_var'] = TimedArray(self.output.transpose(),
                                                 dt=self.dt)
        neurons = self.setup_neuron_group(n_neurons, namespace,
                                          calc_gradient=calc_gradient,
                                          optimize=optimize,
                                          online_error=online_error)
        network = Network(neurons)
        if isinstance(output_var, str):
            output_var = [output_var]
        if 'spikes' in output_var:
            network.add(SpikeMonitor(neurons, name='spikemonitor'))

        record_vars = [v for v in output_var if v != 'spikes']
        if calc_gradient:
            if not len(output_var) == 1:
                raise AssertionError('Cannot calculate gradient with multiple '
                                     'output variables.')
            record_vars.extend([f'S_{output_var[0]}_{p}'
                                for p in self.parameter_names])
        if len(record_vars):
            network.add(StateMonitor(neurons, record_vars, record=True,
                                     name='statemonitor', dt=self.dt))

        if calc_gradient:
            param_init = dict(param_init)
            param_init.update(get_sensitivity_init(neurons, self.parameter_names,
                                                   param_init))
        simulator.initialize(network, param_init, name=network_name)
        return simulator

    def setup_neuron_group(self, n_neurons, namespace, calc_gradient=False,
                           optimize=True, online_error=False, name='neurons'):
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
        model = self.model + Equations('iteration : integer (constant, shared)')
        neurons = NeuronGroup(n_neurons, model,
                              threshold=self.threshold, reset=self.reset,
                              refractory=self.refractory, name=name,
                              namespace=namespace, dt=self.dt, **kwds)
        if calc_gradient:
            sensitivity_eqs = get_sensitivity_equations(neurons,
                                                        parameters=self.parameter_names,
                                                        optimize=optimize,
                                                        namespace=namespace)
            neurons = NeuronGroup(n_neurons, model + sensitivity_eqs,
                                  threshold=self.threshold, reset=self.reset,
                                  refractory=self.refractory, name=name,
                                  namespace=namespace, dt=self.dt, **kwds)
        if online_error:
            neurons.run_regularly('total_error += ({} - {}_target)**2 * '
                                  'int(t>=t_start)'.format(self.output_var,
                                                           self.output_var),
                                  when='end')

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

    def optimization_iter(self, optimizer, metric, penalty):
        """
        Function performs all operations required for one iteration of
        optimization. Drawing parameters, setting them to simulator and
        calulating the error.

        Parameters
        ----------
        optimizer : `Optimizer`
        metric : `Metric`
        penalty : str, optional
            The name of a variable or subexpression in the model that will be
            added to the error at the end of each iteration. Note that only
            the term at the end of the simulation is taken into account, so
            this term should not be varying in time.

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
        self.simulator.run(self.duration, d_param, self.parameter_names,
                           iteration=self.iteration)

        errors = self.calc_errors(metric)

        if penalty is not None:
            error_penalty = getattr(self.simulator.neurons, penalty + '_')
            if self.use_units:
                error_dim = metric.get_dimensions(self.output_dim)
                penalty_dim = self.simulator.neurons.variables[penalty].dim
                fail_for_dimension_mismatch(error_dim, penalty_dim,
                                            error_message='The term used as penalty has to have '
                                                          'the same units as the error.')

            errors += error_penalty

        optimizer.tell(parameters, errors)

        results = optimizer.recommend()

        return results, parameters, errors

    def fit(self, optimizer, metric=None, n_rounds=1, callback='text',
            restart=False, online_error=False, start_iteration=None,
            penalty=None, level=0, **params):
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
        online_error: bool, optional
            Whether to calculate the squared error between target trace and
            simulated trace online. Defaults to ``False``.
        start_iteration: int, optional
            A value for the ``iteration`` variable at the first iteration.
            If not given, will use 0 for the first call of ``fit`` (and for
            later calls when ``restart`` is specified). Later calls will
            continue to increase the value from the previous calls.
        penalty : str, optional
            The name of a variable or subexpression in the model that will be
            added to the error at the end of each iteration. Note that only
            the term at the end of the simulation is taken into account, so
            this term should not be varying in time. If not given, will reuse
            the value specified during ``Fitter`` initialization.
        level : `int`, optional
            How much farther to go down in the stack to find the namespace.
        **params
            bounds for each parameter
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
            raise TypeError("optimizer has to be a child of class Optimizer")

        if self.metric is not None and restart is False:
            if metric is not self.metric:
                raise Exception("You can not change the metric between fits")

        if self.optimizer is not None and restart is False:
            if optimizer is not self.optimizer:
                raise Exception("You can not change the optimizer between fits")

        if start_iteration is not None:
            self.iteration = start_iteration

        if penalty is None:
            penalty = self.penalty

        if self.optimizer is None or restart:
            if start_iteration is None:
                self.iteration = 0
            self.n_samples = optimizer.initialize(self.parameter_names,
                                                  popsize=self._n_samples,
                                                  rounds=n_rounds,
                                                  **params)

        self.optimizer = optimizer
        self.metric = metric

        callback = callback_setup(callback, n_rounds)

        # Check whether we can reuse the current simulator or whether we have
        # to create a new one (only relevant for standalone, but does not hurt
        # for runtime)
        if (restart or
                self.simulator is None or
                self.simulator.current_net != 'fit'):
            self.simulator = self.setup_simulator('fit', self.n_neurons,
                                                  output_var=self.output_var,
                                                  online_error=online_error,
                                                  param_init=self.param_init,
                                                  level=level+1)

        # Run Optimization Loop
        for index in range(n_rounds):
            best_params, parameters, errors = self.optimization_iter(optimizer,
                                                                     metric,
                                                                     penalty)
            self.iteration += 1
            self._best_error = nanmin(self.optimizer.errors)
            # create output variables
            self._best_params = make_dic(self.parameter_names, best_params)
            if self.use_units:
                if self.output_var == 'spikes':
                    output_dim = DIMENSIONLESS
                else:
                    output_dim = self.output_dim
                # Correct the units for the normalization factor
                error_dim = self.metric.get_normalized_dimensions(output_dim)
                best_error = Quantity(float(self.best_error), dim=error_dim)
                errors = Quantity(errors, dim=error_dim)
                param_dicts = [{p: Quantity(v, dim=self.model[p].dim)
                                for p, v in zip(self.parameter_names,
                                                one_param_set)}
                               for one_param_set in parameters]
            else:
                param_dicts = [{p: v for p, v in zip(self.parameter_names,
                                                     one_param_set)}
                               for one_param_set in parameters]
                best_error = self.best_error

            if callback(param_dicts,
                        errors,
                        self.best_params,
                        best_error,
                        index) is True:
                break

        return self.best_params, self.best_error

    @property
    def best_params(self):
        if self._best_params is None:
            return None
        if self.use_units:
            params_with_units = {p: Quantity(v, dim=self.model[p].dim)
                                 for p, v in self._best_params.items()}
            return params_with_units
        else:
            return self._best_params

    @property
    def best_error(self):
        if self._best_error is None:
            return None
        if self.use_units:
            error_dim = self.metric.get_dimensions(self.output_dim)
            return Quantity(self._best_error, dim=error_dim)
        else:
            return self._best_error

    def results(self, format='list', use_units=None):
        """
        Returns all of the gathered results (parameters and errors).
        In one of the 3 formats: 'dataframe', 'list', 'dict'.

        Parameters
        ----------
        format: str
            The desired output format. Currently supported: ``dataframe``,
            ``list``, or ``dict``.
        use_units: bool, optional
            Whether to use units in the results. If not specified, defaults to
            `.Tracefitter.use_units`, i.e. the value that was specified when
            the `.Tracefitter` object was created (``True`` by default).

        Returns
        -------
        object
            'dataframe': returns pandas `~pandas.DataFrame` without units
            'list': list of dictionaries
            'dict': dictionary of lists
        """
        if use_units is None:
            use_units = self.use_units
        names = list(self.parameter_names)

        params = array(self.optimizer.tested_parameters)
        params = params.reshape(-1, params.shape[-1])

        if use_units:
            error_dim = self.metric.get_dimensions(self.output_dim)
            errors = Quantity(array(self.optimizer.errors).flatten(),
                              dim=error_dim)
        else:
            errors = array(array(self.optimizer.errors).flatten())

        dim = self.model.dimensions

        if format == 'list':
            res_list = []
            for j in arange(0, len(params)):
                temp_data = params[j]
                res_dict = dict()

                for i, n in enumerate(names):
                    if use_units:
                        res_dict[n] = Quantity(temp_data[i], dim=dim[n])
                    else:
                        res_dict[n] = float(temp_data[i])
                res_dict['error'] = errors[j]
                res_list.append(res_dict)

            return res_list

        elif format == 'dict':
            res_dict = dict()
            for i, n in enumerate(names):
                if use_units:
                    res_dict[n] = Quantity(params[:, i], dim=dim[n])
                else:
                    res_dict[n] = array(params[:, i])

            res_dict['error'] = errors
            return res_dict

        elif format == 'dataframe':
            from pandas import DataFrame
            if use_units:
                logger.warn('Results in dataframes do not support units. '
                            'Specify "use_units=False" to avoid this warning.',
                            name_suffix='dataframe_units')
            data = concatenate((params, array(errors)[None, :].transpose()), axis=1)
            return DataFrame(data=data, columns=names + ['error'])

    def generate(self, output_var=None, params=None, param_init=None,
                 iteration=1e9, level=0):
        """
        Generates traces for best fit of parameters and all inputs.
        If provided with other parameters provides those.

        Parameters
        ----------
        output_var: str or sequence of str
            Name of the output variable to be monitored, or the special name
            ``spikes`` to record spikes. Can also be a sequence of names to
            record multiple variables.
        params: dict
            Dictionary of parameters to generate fits for.
        param_init: dict
            Dictionary of initial values for the model.
        iteration: int, optional
            Value for the ``iteration`` variable provided to the simulation.
            Defaults to a high value (1e9). This is based on the assumption
            that the model implements some coupling of the fitted variable to
            the target variable, and that this coupling inversely depends on
            the iteration number. In this case, one would usually want to
            switch off the coupling when generating traces/spikes for given
            parameters.
        level : `int`, optional
            How much farther to go down in the stack to find the namespace.

        Returns
        -------
        fit
            Either a 2D `.Quantity` with the recorded output variable over time,
            with shape <number of input traces> × <number of time steps>, or
            a list of spike times for each input trace. If several names were
            given as ``output_var``, then the result is a dictionary with the
            names of the variable as the key.
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
                           iteration=iteration, name='generate')

        if not isinstance(output_var, str):
            fits = {name: self._simulation_result(name) for name in output_var}
        else:
            fits = self._simulation_result(output_var)

        return fits

    def _simulation_result(self, output_var):
        if output_var == 'spikes':
            fits = get_spikes(self.simulator.spikemonitor,
                              1, self.n_traces)[0]  # a single "sample"
        else:
            fits = getattr(self.simulator.statemonitor, output_var)[:]
        return fits


class TraceFitter(Fitter):
    """
    A `Fitter` for fitting recorded traces (e.g. of the membrane potential).

    Parameters
    ----------
    model
    input_var
    input
    output_var
    output
    dt
    n_samples
    method
    reset
    refractory
    threshold
    param_init
    use_units: bool, optional
        Whether to use units in all user-facing interfaces, e.g. in the callback
        arguments or in the returned parameter dictionary and errors. Defaults
        to ``True``.
    """
    def __init__(self, model, input_var, input, output_var, output, dt,
                 n_samples=30, method=None, reset=None, refractory=False,
                 threshold=None, param_init=None, penalty=None, use_units=True):
        super().__init__(dt, model, input, output, input_var, output_var,
                         n_samples, threshold, reset, refractory, method,
                         param_init, penalty=penalty, use_units=use_units)
        self.output = Quantity(output)
        self.output_ = array(output)
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
        To be used inside `optim_iter`.
        """
        traces = getattr(self.simulator.networks['fit']['statemonitor'],
                         self.output_var+'_')
        # Reshape traces for easier calculation of error
        traces = reshape(traces, (traces.shape[0]//self.n_traces,
                                  self.n_traces,
                                  -1))
        errors = metric.calc(traces, self.output_, self.dt)
        return errors

    def fit(self, optimizer, metric=None, n_rounds=1, callback='text',
            restart=False, start_iteration=None, penalty=None,
            level=0, **params):
        if not isinstance(metric, TraceMetric):
            raise TypeError("You can only use TraceMetric child metric with "
                            "TraceFitter")
        if metric.t_weights is not None:
            if not metric.t_weights.shape == (self.output.shape[1], ):
                raise ValueError("The 't_weights' argument of the metric has "
                                 "to be a one-dimensional array of length "
                                 f"{self.output.shape[1]} but has shape "
                                 f"{metric.t_weights.shape}")
        self.bounds = dict(params)
        best_params, error = super().fit(optimizer, metric, n_rounds,
                                         callback, restart,
                                         start_iteration=start_iteration,
                                         penalty=penalty,
                                         level=level+1,
                                         **params)
        return best_params, error

    def generate_traces(self, params=None, param_init=None, iteration=1e9,
                        level=0):
        """Generates traces for best fit of parameters and all inputs"""
        fits = self.generate(params=params, output_var=self.output_var,
                             param_init=param_init, iteration=iteration,
                             level=level+1)
        return fits

    def refine(self, params=None, t_start=None, t_weights=None, normalization=None,
               callback='text', calc_gradient=False, optimize=True,
               iteration=1e9, level=0, **kwds):
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
            Start of time window considered for calculating the fit error.
            If not set, will reuse the `t_start` value from the
            previously used metric.
        t_weights : `~.ndarray`, optional
            A 1-dimensional array of weights for each time point. This array
            has to have the same size as the input/output traces that are used
            for fitting. A value of 0 means that data points are ignored. The
            weight values will be normalized so only the relative values matter.
            For example, an array containing 1s, and 2s, will weigh the
            regions with 2s twice as high (with respect to the squared error)
            as the regions with 1s. Using instead values of 0.5 and 1 would have
            the same effect. Cannot be combined with ``t_start``. If not set, will
            reuse the `t_weights` value from the previously used metric.
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
        iteration: int, optional
            Value for the ``iteration`` variable provided to the simulation.
            Defaults to a high value (1e9). This is based on the assumption
            that the model implements some coupling of the fitted variable to
            the target variable, and that this coupling inversely depends on
            the iteration number. In this case, one would usually want to
            switch off the coupling when refining the solution.
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

        if t_weights is not None:
            t_weights = normalize_weights(t_weights)
        elif t_start is None:
            t_weights = getattr(self.metric, 't_weights', None)
            if t_weights is None:
                t_start = getattr(self.metric, 't_start', 0*second)
            else:
                t_start = None

        if t_start is not None and t_weights is not None:
            raise ValueError("Cannot use both 't_weights' and 't_start'.")

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

        if t_weights is None:
            t_start_steps = int(round(t_start / self.dt))

        def _calc_error(params):
            param_dic = get_param_dic([params[p] for p in self.parameter_names],
                                      self.parameter_names, self.n_traces, 1)
            self.simulator.run(self.duration, param_dic,
                               self.parameter_names, iteration=iteration,
                               name='refine')
            trace = getattr(self.simulator.statemonitor, self.output_var+'_')
            if t_weights is None:
                residual = trace[:, t_start_steps:] - self.output_[:, t_start_steps:]
            else:
                residual = (trace - self.output_) * sqrt(t_weights)
            return residual.flatten() * normalization

        def _calc_gradient(params):
            residuals = []
            for name in self.parameter_names:
                trace = getattr(self.simulator.statemonitor,
                                f'S_{self.output_var}_{name}_')
                if t_weights is None:
                    residual = trace[:, t_start_steps:]
                else:
                    residual = trace * sqrt(t_weights)
                residuals.append(residual.flatten() * normalization)
            gradient = array(residuals)
            return gradient.T

        tested_parameters = []
        errors = []
        def _callback_wrapper(params, iter, resid, *args, **kwds):
            error = mean(resid**2)
            errors.append(error)
            if self.use_units:
                error_dim = self.output_dim**2 * get_dimensions(normalization)**2
                all_errors = Quantity(errors, dim=error_dim)
                params = {p: Quantity(val, dim=self.model[p].dim)
                          for p, val in params.items()}
            else:
                all_errors = array(errors)
                params = {p: float(val) for p, val in params.items()}
            tested_parameters.append(params)

            best_idx = argmin(errors)
            best_error = all_errors[best_idx]
            best_params = tested_parameters[best_idx]

            return callback_func(params, all_errors,
                                 best_params, best_error, iter)

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

        if self.use_units:
            param_dict = {p: Quantity(float(val), dim=self.model[p].dim)
                          for p, val in result.params.items()}
        else:
            param_dict = {p: float(val)
                          for p, val in result.params.items()}

        return param_dict, result


class SpikeFitter(Fitter):
    def __init__(self, model, input, output, dt, reset, threshold,
                 input_var='I', refractory=False, n_samples=30,
                 method=None, param_init=None, penalty=None,
                 use_units=True):
        """Initialize the fitter."""
        if method is None:
            method = 'exponential_euler'
        super().__init__(dt, model, input, output, input_var, 'spikes',
                         n_samples, threshold, reset, refractory, method,
                         param_init, penalty=penalty,
                         use_units=use_units)
        self.output = [Quantity(o) for o in output]
        self.output_ = [array(o) for o in output]

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
        spikes = get_spikes(self.simulator.networks['fit']['spikemonitor'],
                            self.n_samples, self.n_traces)
        errors = metric.calc(spikes, self.output, self.dt)
        return errors

    def fit(self, optimizer, metric=None, n_rounds=1, callback='text',
            restart=False, start_iteration=None, penalty=None,
            level=0, **params):
        if not isinstance(metric, SpikeMetric):
            raise TypeError("You can only use SpikeMetric child metric with "
                            "SpikeFitter")
        best_params, error = super().fit(optimizer, metric, n_rounds,
                                         callback, restart,
                                         start_iteration=start_iteration,
                                         penalty=penalty,
                                         level=level+1,
                                         **params)
        return best_params, error

    def generate_spikes(self, params=None, param_init=None, iteration=1e9, level=0):
        """Generates traces for best fit of parameters and all inputs"""
        fits = self.generate(params=params, output_var='spikes',
                             param_init=param_init, iteration=iteration,
                             level=level+1)
        return fits


class OnlineTraceFitter(Fitter):
    def __init__(self, model, input_var, input, output_var, output, dt,
                 n_samples=30, method=None, reset=None, refractory=False,
                 threshold=None, param_init=None,
                 t_start=0*second, penalty=None):
        """Initialize the fitter."""
        super().__init__(dt, model, input, output, input_var, output_var,
                         n_samples, threshold, reset, refractory, method,
                         param_init, penalty=penalty)

        self.output = Quantity(output)
        self.output_ = array(output)

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

    def fit(self, optimizer, n_rounds=1, callback='text',
            restart=False, start_iteration=None, penalty=None,
            level=0, **params):
        metric = MSEMetric()  # not used, but makes error dimensions correct
        return super(OnlineTraceFitter, self).fit(optimizer, metric=metric,
                                                  n_rounds=n_rounds,
                                                  callback=callback,
                                                  restart=restart,
                                                  online_error=True,
                                                  penalty=penalty,
                                                  start_iteration=start_iteration,
                                                  level=level+1,
                                                  **params)

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
