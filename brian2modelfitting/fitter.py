import abc
import numbers
from distutils.version import LooseVersion
from typing import Sequence, Mapping

import sympy
from numpy import (ones, array, arange, concatenate, mean, argmin, nanargmin,
                   reshape, zeros, sqrt, ndarray, broadcast_to, sum, cumsum,
                   hstack)
from scipy.optimize import least_squares

from brian2.parsing.sympytools import sympy_to_str, str_to_sympy
from brian2.units.fundamentalunits import DIMENSIONLESS, get_dimensions, fail_for_dimension_mismatch
from brian2.utils.stringtools import get_identifiers

from brian2 import (NeuronGroup, defaultclock, get_device, Network,
                    StateMonitor, SpikeMonitor, second, get_local_namespace,
                    Quantity, get_logger, Expression, ms)
from brian2.input import TimedArray
from brian2.equations.equations import Equations, SUBEXPRESSION, SingleEquation
from brian2.devices import device
from brian2.devices.cpp_standalone.device import CPPStandaloneDevice
from brian2.core.functions import Function
from .simulator import RuntimeSimulator, CPPStandaloneSimulator
from .metric import Metric, SpikeMetric, TraceMetric, MSEMetric, GammaFactor, normalize_weights
from .optimizer import Optimizer, NevergradOptimizer
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
            unit = eqs[orig_var].dim / group.variables[param].dim
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
    input : dic, `~numpy.ndarray` or `~brian2.units.fundamentalunits.Quantity`
        A dictionary given the input variable as the key and a 2D array of
        shape ``(n_traces, time steps)`` as the value, defining the input that
        will be fed into the model. Note that this variable should be *used*
        in the model (e.g. a variable ``I`` that is added as a current in the
        membrane potential equation), but not *defined*.
    output : dict, `~brian2.units.fundamentalunits.Quantity` or list
        Recorded output of the model that the model should reproduce. Should be
        given as a dictionary with the name of the variable as the key and the
        desired output as the value. The desired output has to be a 2D array of
        the same shape as the input when fitting traces with
        `TraceFitter`, or a list of spike times when fitting spike trains with
        `SpikeFitter`. Can also be a list of several output 2D arrays or a
        single output array if combined with ``output_var`` (deprecated use).
    input_var : str
        The name of the input variable in the model. Note that this variable
        should be *used* in the model (e.g. a variable ``I`` that is added as
        a current in the membrane potential equation), but not *defined*.
        .. deprecated:: 0.5
            Use a dictionary for ``input`` instead.
    output_var : str or list of str
        The name of the output variable in the model or a list of output
        variables. Only needed when fitting traces with `.TraceFitter`.
        .. deprecated:: 0.5
            Use a dictionary for ``output`` instead.
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
    def __init__(self, dt, model, input, output, n_samples, input_var=None,
                 output_var=None, threshold=None, reset=None, refractory=None,
                 method=None, param_init=None, penalty=None, use_units=True):
        """Initialize the fitter."""

        if isinstance(model, str):
            model = Equations(model)

        # Support deprecated legacy syntax of input_var + input or the new
        # syntax with a dictionary as input
        if input_var is not None:
            logger.warn('Use the \'input\' argument with a dictionary instead '
                        'of giving the name as \'input_var\'',
                        name_suffix='deprecated_input_var')
            if isinstance(input, Mapping) and input_var not in input:
                raise ValueError('Name given as \'input_var\' and key in '
                                 '\'input\' dictionary do not match.')
        else:
            if not isinstance(input, Mapping):
                raise TypeError('\'input\' argument has to be a dictionary '
                                'mapping the name of the input variable to the '
                                'input.')
            if len(input) > 1:
                raise NotImplementedError('Only a single input is currently '
                                          'supported.')
            input_var = list(input.keys())[0]

        if isinstance(input, Mapping):
            input = input[input_var]

        if input_var != 'spikes' and input_var not in model.identifiers:
            raise NameError(f"{input_var} is not an identifier in the model")

        # Support deprecated legacy syntax of output_var + input or the new
        # syntax with a dictionary as output
        if output_var is not None:
            logger.warn('Use the \'output\' argument with a dictionary instead '
                        'of giving the name as \'output_var\'',
                        name_suffix='deprecated_output_var')
            if isinstance(output_var, str):
                output_var = [output_var]

            if isinstance(output, Mapping):
                if set(output_var) != set(output.keys()):
                    raise ValueError('Names given as \'output_var\' and keys '
                                     'in \'output\' dictionary do not match.')
            elif not isinstance(output, list):
                output = [output]
        else:
            if not isinstance(output, Mapping):
                raise TypeError('\'output\' argument has to be a dictionary '
                                'mapping the name of the input variable to the '
                                'input.')
            output_var = list(output.keys())
            output = list(output.values())

        for o_var in output_var:
            if o_var != 'spikes' and o_var not in model.names:
                raise NameError(f"{o_var} is not a model variable")

        self.output_var = output_var
        self.output = output
        self.output_ = [array(o, copy=False) for o in output]

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

        self.output_dim = []
        for o_var, out in zip(self.output_var, output):
            if o_var == 'spikes':
                self.output_dim.append(DIMENSIONLESS)
            else:
                self.output_dim.append(model[o_var].dim)
                fail_for_dimension_mismatch(out, self.output_dim[-1],
                                            'The provided target values '
                                            '("output") need to have the same '
                                            'units as the variable '
                                            '{}'.format(o_var))
        self.model = model

        self.use_units = use_units
        self.iteration = 0
        input_dim = get_dimensions(input)
        input_dim = '1' if input_dim is DIMENSIONLESS else repr(input_dim)
        input_eqs = "{} = input_var(t, i % n_traces) : {}".format(input_var,
                                                                  input_dim)
        self.model += input_eqs

        counter = 0
        for o_var, o_dim in zip(self.output_var, self.output_dim):
            if o_var != 'spikes':
                counter += 1
                # For approaches that couple the system to the target values,
                # provide a convenient variable
                output_expr = f'output_var_{counter}(t, i % n_traces)'
                output_dim = ('1' if o_dim is DIMENSIONLESS
                              else repr(o_dim))
                output_eqs = "{}_target = {} : {}".format(o_var,
                                                          output_expr,
                                                          output_dim)
                self.model += output_eqs

        input_traces = TimedArray(input.transpose(), dt=dt)
        self.input_traces = input_traces

        # initialization of attributes used later
        self._best_params = None
        self._best_error = None
        self._best_objective_errors = None
        self._best_objective_errors_normed = None
        self._objective_errors = []
        self._objective_errors_normed = []
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
        counter = 0
        for o_var, out in zip(self.output_var, self.output):
            if self.output_var != 'spikes':
                counter += 1
                namespace[f'output_var_{counter}'] = TimedArray(out.transpose(),
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
            record_vars.extend([f'S_{out_var}_{p}'
                                for p in self.parameter_names
                                for out_var in self.output_var])
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
            # The sensitivity equations only add variables for variables
            # defined by differential equations. For output variables that
            # are given by subexpressions, we also add subexpressions to
            # calculate their sensitivity.
            sensititivity_subexpressions = Equations('')
            for output_var in self.output_var:
                if output_var in neurons.equations.subexpr_names:
                    subexpr = neurons.equations[output_var]
                    sympy_expr = str_to_sympy(subexpr.expr.code, variables=neurons.variables)
                    for parameter in self.parameter_names:
                        # FIXME: Deal with subexpressions that depend on other
                        #        subexpressions
                        parameter_symbol = sympy.Symbol(parameter, real=True)
                        referred_vars = {var for var in subexpr.identifiers
                                         if var in model.diff_eq_names}
                        var_func_derivatives = {}
                        # Express all referenced variables as functions of the parameters
                        new_sympy_expr = sympy_expr
                        for referred_var in referred_vars:
                            var_symbol = sympy.Symbol(referred_var, real=True)
                            var_func = sympy.Function(var_symbol)(parameter_symbol)
                            var_derivative = sympy.Derivative(var_func, parameter_symbol)
                            var_func_derivatives[var_symbol] = (var_func, var_derivative)
                            new_sympy_expr = new_sympy_expr.subs(var_symbol, var_func)
                        # Differentiate the function with respect to the parameter
                        diffed = sympy.diff(new_sympy_expr, parameter_symbol)
                        # Replace the functions/derivatives by the correct symbols
                        for var, (func, derivative) in var_func_derivatives.items():
                            diffed = diffed.subs({func: var,
                                                  derivative: sympy.Symbol(f'S_{var}_{parameter}',
                                                                           real=True)})

                        new_eqs = Equations([SingleEquation(subexpr.type,
                                                            varname=f'S_{output_var}_{parameter}',
                                                            dimensions=subexpr.dim/neurons.equations[parameter].dim,
                                                            var_type=subexpr.var_type,
                                                            expr=Expression(sympy_to_str(diffed)))])
                        sensititivity_subexpressions += new_eqs
            new_model = model + sensitivity_eqs + sensititivity_subexpressions
            neurons = NeuronGroup(n_neurons, new_model,
                                  threshold=self.threshold, reset=self.reset,
                                  refractory=self.refractory, name=name,
                                  namespace=namespace, dt=self.dt, **kwds)
        if online_error:
            updater = []
            for o_var in self.output_var:
                updater.append(f'total_error_{o_var} += ({o_var} - {o_var}_target)**2 '
                               f'* int(t >= t_start_{o_var})')
            neurons.run_regularly('\n'.join(updater), when='end')

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

        raw_errors = array(self.calc_errors(metric))
        errors = sum(raw_errors, axis=0)

        if penalty is not None:
            error_penalty = getattr(self.simulator.neurons, penalty + '_')
            if self.use_units:
                error_dim = metric[0].get_normalized_dimensions(self.output_dim[0])
                for one_metric, one_dim in zip(metric[1:], self.output_dim[1:]):
                    other_dim = one_metric.get_normalized_dimensions(one_dim)
                    fail_for_dimension_mismatch(error_dim, other_dim,
                                                error_message='The error terms have mismatching '
                                                              'units.')
                penalty_dim = self.simulator.neurons.variables[penalty].dim
                fail_for_dimension_mismatch(error_dim, penalty_dim,
                                            error_message='The term used as penalty has to have '
                                                          'the same units as the error.')

            errors += error_penalty
        self._objective_errors_normed.extend(raw_errors.T.tolist())
        unnormalized_raw_errors = array([metric.revert_normalization(err)
                                         for metric, err in zip(metric, raw_errors)]).T.tolist()
        self._objective_errors.extend(unnormalized_raw_errors)
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
        metric: `~.Metric`, or dict
            Child of Metric class, specifies optimization metric. In the case
            of multiple fitted output variables, can either be a single
            `~.Metric` that is applied to all variables, or a dictionary with a
            `~.Metric` for each variable.
        n_rounds: int
            Number of rounds to optimize over (feedback provided over each
            round).
        callback: `str` or `~typing.Callable`
            Either the name of a provided callback function (``text`` or
            ``progressbar``), or a custom feedback function
            ``func(parameters, errors, best_parameters, best_error, index, additional_info)``.
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
        metric = self._verify_metric_argument(metric)

        # Convert metric dictionary to parallel list with output variables
        metric = [metric[o] if metric is not None else None
                  for o in self.output_var]
        for single_metric, output in zip(metric, self.output):
            if getattr(single_metric, 't_weights', None) is not None:
                if not single_metric.t_weights.shape == (
                        output.shape[1],):
                    raise ValueError(
                        "The 't_weights' argument of the metric has "
                        "to be a one-dimensional array of length "
                        f"{output.shape[1]} but has shape "
                        f"{single_metric.t_weights.shape}")

        if not (isinstance(optimizer, Optimizer) or optimizer is None):
            raise TypeError("optimizer has to be a child of class Optimizer or None")

        if self.metric is not None and restart is False:
            if (len(metric) != len(self.metric) or
                    any(m1 is not m2 for m1, m2 in zip(metric, self.metric))):
                raise Exception("You can not change the metric between fits")

        if self.optimizer is not None and restart is False:
            if optimizer is not self.optimizer:
                raise Exception("You can not change the optimizer between fits")

        if start_iteration is not None:
            self.iteration = start_iteration

        if penalty is None:
            penalty = self.penalty

        if optimizer is None:
            optimizer = NevergradOptimizer()

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
            param_init = dict(self.param_init)
            if online_error:
                for m, o_var in zip(self.metric, self.output_var):
                    param_init[f't_start_{o_var}'] = getattr(m, 't_start', 0*second)

            self.simulator = self.setup_simulator('fit', self.n_neurons,
                                                  output_var=self.output_var,
                                                  online_error=online_error,
                                                  param_init=param_init,
                                                  level=level+1)

        # Run Optimization Loop
        for index in range(n_rounds):
            best_params, parameters, errors = self.optimization_iter(optimizer,
                                                                     self.metric,
                                                                     penalty)
            self.iteration += 1
            best_idx = nanargmin(self.optimizer.errors)
            self._best_error = self.optimizer.errors[best_idx]
            self._best_objective_errors_normed = tuple(self._objective_errors_normed[best_idx])
            self._best_objective_errors = tuple(self._objective_errors[best_idx])
            # create output variables
            self._best_params = make_dic(self.parameter_names, best_params)
            if self.use_units:
                error_dim = self.metric[0].get_normalized_dimensions(self.output_dim[0])
                for metric, output_dim in zip(self.metric[1:], self.output_dim[1:]):
                    # Correct the units for the normalization factor
                    other_dim = metric.get_normalized_dimensions(output_dim)
                    fail_for_dimension_mismatch(error_dim, other_dim,
                                                error_message='The error terms have mismatching '
                                                              'units.')
                best_error = Quantity(float(self.best_error), dim=error_dim)
                errors = Quantity(errors, dim=error_dim)
                param_dicts = [{p: Quantity(v, dim=self.model[p].dim)
                                for p, v in zip(self.parameter_names,
                                                one_param_set)}
                               for one_param_set in parameters]
                best_raw_error_normed = tuple([Quantity(raw_error_normed,
                                                        dim=metric.get_normalized_dimensions(output_dim))
                                               for raw_error_normed, metric, output_dim
                                               in zip(self._best_objective_errors_normed,
                                                      self.metric,
                                                      self.output_dim)])
                best_raw_error = tuple([Quantity(raw_error,
                                                 dim=metric.get_dimensions(output_dim))
                                        for raw_error, metric, output_dim
                                        in zip(self._best_objective_errors,
                                               self.metric,
                                               self.output_dim)])
            else:
                param_dicts = [{p: v for p, v in zip(self.parameter_names,
                                                     one_param_set)}
                               for one_param_set in parameters]
                best_error = self.best_error
                best_raw_error = self._best_objective_errors
                best_raw_error_normed = self._best_objective_errors_normed

            additional_info = {'objective_errors': best_raw_error,
                               'objective_errors_normalized': best_raw_error_normed,
                               'output_var': self.output_var}

            if callback(param_dicts,
                        errors,
                        self.best_params,
                        best_error,
                        index,
                        additional_info) is True:
                break

        return self.best_params, self.best_error

    def _verify_metric_argument(self, metric, metric_class=Metric):
        if isinstance(metric, Metric):
            metric = {varname: metric
                      for varname in self.output_var}
        elif isinstance(metric, Sequence):
            logger.warn('Using a list of metrics is deprecated, use a '
                        'dictionary instead.',
                        name_suffix='deprecated_metric_list')
            if not len(metric) == len(self.output_var):
                raise TypeError('List of metrics needs to have as many '
                                'elements as output variables.')
            metric = {o : m for m, o in zip(metric, self.output_var)}

        if isinstance(metric, Mapping):
            for o, m in metric.items():
                if o not in self.output_var:
                    raise ValueError(f"Metric key '{o}' does not correspond to "
                                     f"an output variable name.")
                    if not isinstance(m, Metric):
                        raise TypeError("metric has to be a child of class "
                                        "Metric.")
                if not isinstance(m, metric_class):
                    raise TypeError(f"metric has to be a child of class "
                                    f"'{metric_class.__name__}' but is a "
                                    f"'{type(m).__name__}'.")
            for o in self.output_var:
                if not o in metric:
                    raise ValueError(f"Output variable '{o}' does not have a "
                                     f"corresponding entry in the metric "
                                     f"dictionary.")
        elif metric is not None:
            if not isinstance(metric, Mapping):
                raise TypeError('Metric has to be a Metric instance, or a'
                                'dictionary mapping output variable names to '
                                'Metric instances. Use None for'
                                'OnlineTraceFitter.')

        return metric

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
            error_dim = self.metric[0].get_normalized_dimensions(self.output_dim[0])
            # We assume that the error units have already been checked to
            # be consistent at this point
            return Quantity(self._best_error, dim=error_dim)
        else:
            return self._best_error

    @property
    def best_objective_errors(self):
        if self._best_objective_errors is None:
            return None
        if self.use_units:
            return {output_var: Quantity(raw_error, dim=metric.get_dimensions(output_dim))
                    for output_var, raw_error, metric, output_dim in
                    zip(self.output_var, self._best_objective_errors, self.metric, self.output_dim)}
        else:
            return {output_var: raw_error
                    for output_var, raw_error
                    in zip(self.output_var, self._best_objective_errors)}

    @property
    def best_objective_errors_normalized(self):
        if self._best_objective_errors_normed is None:
            return None
        if self.use_units:
            return {output_var: Quantity(raw_error_normed,
                                         dim=metric.get_normalized_dimensions(output_dim))
                    for output_var, raw_error_normed, metric, output_dim in
                    zip(self.output_var, self._best_objective_errors_normed, self.metric, self.output_dim)}
        else:
            return {output_var: raw_error
                    for output_var, raw_error
                    in zip(self.output_var, self._best_objective_errors_normed)}

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
            error_dim = self.metric[0].get_dimensions(self.output_dim[0])
            errors = Quantity(array(self.optimizer.errors).flatten(),
                              dim=error_dim)
            if len(self.output_var) > 1:
                # Add additional information for the raw errors
                raw_errors_normed = {output_var: Quantity([raw_error_normed[idx]
                                                           for raw_error_normed in self._objective_errors_normed],
                                                          dim=metric.get_normalized_dimensions(output_dim))
                                     for idx, (output_var, metric, output_dim)
                                     in enumerate(zip(self.output_var, self.metric, self.output_dim))
                                     }
                raw_errors = {output_var: Quantity([raw_error[idx]
                                                    for raw_error in self._objective_errors],
                                                   dim=metric.get_dimensions(output_dim))
                              for idx, (output_var, metric, output_dim)
                              in enumerate(zip(self.output_var, self.metric, self.output_dim))
                              }
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
                if len(self.output_var) > 1:
                    if use_units:
                        res_dict['objective_errors_normalized'] = {output_var: raw_errors_normed[output_var][j]
                                                                   for output_var in self.output_var}
                        res_dict['objective_errors'] = {output_var: raw_errors[output_var][j]
                                                        for output_var in self.output_var}
                    else:
                        res_dict['objective_errors_normalized'] = {output_var: self._objective_errors_normed[j][idx]
                                                               for idx, output_var in enumerate(self.output_var)}
                        res_dict['objective_errors'] = {output_var: self._objective_errors[j][idx]
                                                        for idx, output_var in enumerate(self.output_var)}
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
            if len(self.output_var) > 1:
                if use_units:
                    res_dict['objective_errors_normalized'] = {output_var: raw_errors_normed[output_var]
                                                               for output_var in self.output_var}
                    res_dict['objective_errors'] = {output_var: raw_errors[output_var]
                                                    for output_var in self.output_var}
                else:
                    res_dict['objective_errors_normalized'] = {output_var: array([raw_error_normed[idx]
                                                                                  for raw_error_normed in self._objective_errors_normed])
                                                               for idx, output_var in enumerate(self.output_var)}
                    res_dict['objective_errors'] = {output_var: array([raw_error[idx]
                                                                       for raw_error in self._objective_errors])
                                                    for idx, output_var in enumerate(self.output_var)}
            return res_dict

        elif format == 'dataframe':
            from pandas import DataFrame
            if use_units:
                logger.warn('Results in dataframes do not support units. '
                            'Specify "use_units=False" to avoid this warning.',
                            name_suffix='dataframe_units')
            data = concatenate((params, array(errors)[None, :].transpose()), axis=1)
            columns = names + ['error']
            if len(self.output_var) > 1:
                data = concatenate((data, self._objective_errors_normed), axis=1)
                columns += [f'normalized_error_{output_var}'
                            for output_var in self.output_var]
                data = concatenate((data, self._objective_errors), axis=1)
                columns += [f'error_{output_var}'
                            for output_var in self.output_var]
            return DataFrame(data=data, columns=columns)

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
        elif isinstance(output_var, str):
            output_var = [output_var]
        self.simulator = self.setup_simulator('generate', self.n_traces,
                                              output_var=output_var,
                                              param_init=param_init,
                                              level=level+1)
        param_dic = get_param_dic([params[p] for p in self.parameter_names],
                                  self.parameter_names, self.n_traces, 1)
        self.simulator.run(self.duration, param_dic, self.parameter_names,
                           iteration=iteration, name='generate')

        if len(output_var) > 1:
            fits = {name: self._simulation_result(name) for name in output_var}
        else:
            fits = self._simulation_result(output_var[0])

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
    def __init__(self, model, input, output, dt,
                 n_samples=60, input_var=None, output_var=None,
                 method=None, reset=None, refractory=False,
                 threshold=None, param_init=None, penalty=None, use_units=True):
        super().__init__(dt=dt, model=model, input=input, output=output,
                         input_var=input_var, output_var=output_var,
                         n_samples=n_samples, threshold=threshold, reset=reset,
                         refractory=refractory, method=method,
                         param_init=param_init, penalty=penalty,
                         use_units=use_units)
        for o in self.output:
            if o.shape != self.input.shape:
                raise ValueError("Input and output must have the same size")
        # We store the bounds set in TraceFitter.fit, so that Tracefitter.refine
        # can reuse them
        self.bounds = None

    def calc_errors(self, metric):
        """
        Returns errors after simulation with StateMonitor.
        To be used inside `optim_iter`.
        """
        all_errors = []
        for m, o_var, o in zip(metric, self.output_var, self.output_):
            traces = getattr(self.simulator.networks['fit']['statemonitor'],
                             o_var+'_')
            # Reshape traces for easier calculation of error
            traces = reshape(traces, (traces.shape[0]//self.n_traces,
                                      self.n_traces,
                                      -1))
            errors = m.calc(traces, o, self.dt)
            all_errors.append(errors)
        return all_errors

    def fit(self, optimizer, metric=None, n_rounds=1, callback='text',
            restart=False, start_iteration=None, penalty=None,
            level=0, **params):
        self.bounds = dict(params)
        if metric is None:
            metric = MSEMetric()
        metric = self._verify_metric_argument(metric, metric_class=TraceMetric)
        best_params, error = super().fit(optimizer=optimizer,
                                         metric=metric,
                                         n_rounds=n_rounds,
                                         callback=callback,
                                         restart=restart,
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

    def refine(self, params=None, metric=None,
               callback='text', calc_gradient=False, optimize=True,
               iteration=1e9, level=0, **kwds):
        """
        Refine the fitting results with a sequentially operating minimization
        algorithm. The `~.scipy.optimize.least_squares` algorithm from
        `scipy.optimize <https://docs.scipy.org/doc/scipy/reference/optimize.html>`_.
        Has to be called after `~.TraceFitter.fit`, but a call with
        ``n_rounds=0`` is enough.

        Parameters
        ----------
        params : dict, optional
            A dictionary with the parameters to use as a starting point for the
            refinement. If not given, the best parameters found so far by
            `~.TraceFitter.fit` will be used.
        metric: `~.MSEMetric` or dict, optional
            Optimization metrics to use. Since the refinement only supports
            mean-squared-error metrics, this is only useful to provide the
            ``t_start``/``t_weights``/``normalization`` values. In
            the case of multiple fitted output variables, can either be a single
            `~.MSEMetric` that is applied to all variables, or a dictionary with a
            `~.MSEMetric` for each variable. If not given, will reuse the metrics
            of a previous `~.Fitter.fit` call.
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
            be passed on to `.~scipy.optimize.least_squares`.

        Returns
        -------
        parameters : dict
            The parameters at the end of the optimization process as a
            dictionary.
        result : `.scipy.optimize.OptimizeResult`
            The result of the optimization process.

        Notes
        -----
        There is no support for specifying a `Metric`, the given output trace(s)
        will be subtracted from the simulated trace(s) and passed on to the
        minimization algorithm which will internally calculate the sum of
        squares.
        """
        if params is None:
            if self.best_params is None:
                raise TypeError('You need to either specify parameters or run '
                                'the fit function first.')
            params = self.best_params

        if metric is None:
            if self.metric is None:
                metric = {output_var: MSEMetric()
                          for output_var in self.output_var}
            else:
                metric = {output_var: m
                          for output_var, m in zip(self.output_var, self.metric)}
        elif not isinstance(metric, Mapping):
            metric = {output_var: metric
                      for output_var in self.output_var}

        for var, m in metric.items():
            if not isinstance(m, MSEMetric):
                raise TypeError(f"The refine method only supports MSEMetric, but "
                                f"the metric for variable '{var}' is of type "
                                f"'{type(m)}'")

        # Extract the necessary normalization info in flat arrays
        t_weights = [getattr(metric[v], 't_weights', None)
                     for v in self.output_var]
        t_start = [getattr(metric[v], 't_start', 0*second)
                   for v in self.output_var]
        normalization = [getattr(metric[v], 'normalization', 1.0)
                         for v in self.output_var]

        callback_func = callback_setup(callback, None)

        # Set up Parameter objects
        min_bounds = []
        max_bounds = []
        x0 = []
        for param_name in self.parameter_names:
            if param_name not in kwds:
                if self.bounds is None:
                    raise TypeError('You need to either specify bounds for all '
                                    'parameters or run the fit function first.')
                min_bound, max_bound = self.bounds[param_name]
            else:
                min_bound, max_bound = kwds.pop(param_name)
            x0.append(params[param_name])
            min_bounds.append(min_bound)
            max_bounds.append(max_bound)

        self.simulator = self.setup_simulator('refine', self.n_traces,
                                              output_var=self.output_var,
                                              param_init=self.param_init,
                                              calc_gradient=calc_gradient,
                                              optimize=optimize,
                                              level=level+1)

        t_start_steps = [int(round(t_s / self.dt)) if t_w is None else 0
                         for t_s, t_w in zip(t_start, t_weights)]

        # TODO: Move all this into a class
        tested_parameters = []
        errors = []
        combined_errors = []
        n_evaluations = [-1]

        def _calc_error(x):
            param_dic = get_param_dic(x, self.parameter_names, self.n_traces, 1)
            self.simulator.run(self.duration, param_dic,
                               self.parameter_names, iteration=iteration,
                               name='refine')
            one_residual = []

            for out_var, out, t_s_steps, t_w, norm in zip(self.output_var,
                                                          self.output_,
                                                          t_start_steps,
                                                          t_weights,
                                                          normalization):
                trace = getattr(self.simulator.statemonitor, out_var+'_')
                if t_w is None:
                    residual = trace[:, t_s_steps:] - out[:, t_s_steps:]
                else:
                    residual = (trace - out) * sqrt(t_w)
                one_residual.append((residual*norm).flatten())

            output_len = [output[:, t_s_steps:].size
                          for output, t_s_steps in zip(self.output,
                                                       t_start_steps)]
            end_idx = cumsum(output_len)
            start_idx = hstack([0, end_idx[:-1]])
            error = tuple([mean(r**2) for r in one_residual])
            combined_error = sum(array(error))
            errors.append(error)
            combined_errors.append(combined_error)
            best_idx = argmin(combined_errors)

            if self.use_units:
                norm_dim = get_dimensions(normalization[0]) ** 2
                error_dim = self.output_dim[0] ** 2 * norm_dim
                for output_dim, norm in zip(self.output_dim[1:],
                                            normalization[1:]):
                    norm_dim = get_dimensions(norm) ** 2
                    other_dim = output_dim ** 2 * norm_dim
                    fail_for_dimension_mismatch(error_dim, other_dim,
                                                error_message='The error terms have mismatching '
                                                              'units.')
                all_errors = Quantity(combined_errors, dim=error_dim)
                params = {p: Quantity(val, dim=self.model[p].dim)
                          for p, val in zip(self.parameter_names, x)}
                best_raw_error_normed = tuple([Quantity(raw_error,
                                                        dim=output_dim ** 2 * get_dimensions(
                                                            norm) ** 2)
                                               for raw_error, output_dim, norm
                                               in zip(errors[best_idx],
                                                      self.output_dim,
                                                      normalization)])
                best_raw_error = tuple([Quantity(raw_error / norm ** 2,
                                                 dim=output_dim ** 2)
                                        for raw_error, output_dim, norm
                                        in zip(errors[best_idx],
                                               self.output_dim,
                                               normalization)])
            else:
                all_errors = array(combined_errors)
                params = {p: float(val) for p, val
                          in zip(self.parameter_names, x)}
                best_raw_error_normed = errors[best_idx]
                best_raw_error = [err / norm ** 2
                                  for norm, err in
                                  zip(errors[best_idx], normalization)]
            tested_parameters.append(params)

            best_error = all_errors[best_idx]
            best_params = tested_parameters[best_idx]
            additional_info = {'objective_errors': best_raw_error,
                               'objective_errors_normalized': best_raw_error_normed,
                               'output_var': self.output_var}
            callback_func(params, errors,
                          best_params, best_error, n_evaluations[0], additional_info)
            n_evaluations[0] += 1
            return array(hstack(one_residual))

        def _calc_gradient(params):
            residuals = []
            for name in self.parameter_names:
                one_residual = []
                for out_var, t_s_steps, t_w, norm in zip(self.output_var,
                                                         t_start_steps,
                                                         t_weights,
                                                         normalization):
                    trace = getattr(self.simulator.statemonitor,
                                    f'S_{out_var}_{name}_')
                    if t_w is None:
                        residual = trace[:, t_s_steps:]
                    else:
                        residual = trace * sqrt(t_w)
                    one_residual.append((residual*norm).flatten())
                residuals.append(array(hstack(one_residual)))
            gradient = array(residuals)
            return gradient.T

        assert 'jac' not in kwds
        if calc_gradient:
            kwds.update({'jac': _calc_gradient})

        if 'maxfev' in kwds:
            if 'max_nfev' in kwds:
                raise ValueError("Cannot provide both 'maxfev' and 'max_nfev' "
                                 "as arguments. Please only provide "
                                 "'max_nfev'.")
            logger.warn("The 'maxfev' argument is deprecated, please use "
                        "'max_nfev' instead.", name_suffix='deprecated_maxfev',
                        once=True)
            kwds['max_nfev'] = kwds.pop('maxfev')

        result = least_squares(_calc_error, x0,
                               bounds=(min_bounds, max_bounds),
                               **kwds)

        if self.use_units:
            param_dict = {p: Quantity(float(val), dim=self.model[p].dim)
                          for p, val in zip(self.parameter_names, result.x)}
        else:
            param_dict = {p: float(val)
                          for p, val in zip(self.parameter_names, result.x)}

        return param_dict, result


class SpikeFitter(Fitter):
    def __init__(self, model, input, output, dt, reset, threshold,
                 input_var='I', refractory=False, n_samples=30,
                 method=None, param_init=None, penalty=None,
                 use_units=True):
        """Initialize the fitter."""
        if method is None:
            method = 'exponential_euler'
        super().__init__(dt, model, input=input, output=output,
                         input_var=input_var, output_var=['spikes'],
                         n_samples=n_samples, threshold=threshold,
                         reset=reset, refractory=refractory,
                         method=method, param_init=param_init,
                         penalty=penalty, use_units=use_units)
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
        assert len(metric) == 1
        errors = [metric[0].calc(spikes, self.output, self.dt)]
        return errors

    def fit(self, optimizer, metric=None, n_rounds=1, callback='text',
            restart=False, start_iteration=None, penalty=None,
            level=0, **params):
        if metric is None:
            metric = GammaFactor(delta=2*ms, time=self.duration)
        metric = self._verify_metric_argument(metric, metric_class=SpikeMetric)
        best_params, error = super().fit(optimizer=optimizer,
                                         metric=metric,
                                         n_rounds=n_rounds,
                                         callback=callback,
                                         restart=restart,
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
    def __init__(self, model, input, output, dt,
                 n_samples=30, input_var=None, output_var=None,
                 method=None, reset=None, refractory=False,
                 threshold=None, param_init=None,
                 penalty=None):
        """Initialize the fitter."""
        super().__init__(dt, model, input=input, output=output,
                         input_var=input_var, output_var=output_var,
                         n_samples=n_samples, threshold=threshold,
                         reset=reset, refractory=refractory, method=method,
                         param_init=param_init, penalty=penalty)

        for o_var, o_data, o_dim in zip(self.output_var, self.output, self.output_dim):
            squared_output_dim = ('1' if o_dim is DIMENSIONLESS
                                  else repr(o_dim**2))
            self.model += Equations(f'total_error_{o_var} : {squared_output_dim}')
            self.model += Equations(f't_start_{o_var} : second (constant, shared)')

        if param_init:
            for param, val in param_init.items():
                if not (param in self.model.identifiers or param in self.model.names):
                    raise ValueError("%s is not a model variable or an "
                                     "identifier in the model" % param)
            self.param_init = param_init

        self.simulator = None

    def fit(self, optimizer, metric=None, n_rounds=1, callback='text',
            restart=False, start_iteration=None, penalty=None,
            level=0, **params):
        if metric is None:
            metric = MSEMetric()
        metric = self._verify_metric_argument(metric)
        for m in metric.values():
            if getattr(m, 't_weights', None) is not None:
                raise NotImplementedError("The 't_weights' argument cannot be used "
                                          "with OnlineTraceFitter, only 't_start' is "
                                          "supported.")
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
        all_errors = []
        for o_var in self.output_var:
            t_start = getattr(self.simulator.neurons, f't_start_{o_var}')
            timesteps = int((self.duration - t_start) / defaultclock.dt)
            errors = getattr(self.simulator.neurons, f'total_error_{o_var}')
            errors = mean(errors[:].reshape((self.n_samples, self.n_traces))/timesteps,
                          axis=1)
            all_errors.append(array(errors))
        return all_errors

    def generate_traces(self, params=None, param_init=None, level=0):
        """Generates traces for best fit of parameters and all inputs"""
        fits = self.generate(params=params, output_var=self.output_var,
                             param_init=param_init, level=level+1)
        return fits
