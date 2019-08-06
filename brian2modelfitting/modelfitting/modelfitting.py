from types import FunctionType
from numpy import mean, ones, array, arange
from brian2 import (NeuronGroup,  defaultclock, get_device, Network,
                    StateMonitor, SpikeMonitor, ms, second)
from brian2.input import TimedArray
from brian2.equations.equations import Equations
from .simulation import RuntimeSimulation, CPPStandaloneSimulation
from .metric import Metric

__all__ = ['fit_traces', 'fit_spikes']


def make_dic(names, values):
    """Create dictionary based on list of strings and 2D array"""
    result_dict = dict()
    for name, value in zip(names, values):
        result_dict[name] = value

    return result_dict


def get_param_dic(params, param_names, n_traces, n_samples):
    """Transform parameters into a dictionary of appropiate size"""
    params = array(params)

    d = dict()

    for name, value in zip(param_names, params.T):
        d[name] = (ones((n_traces, n_samples)) * value).T.flatten()
    return d


def get_spikes(monitor):
    """
    Get spikes from spike monitor change format from dict to a list,
    remove units.
    """
    spike_trains = monitor.spike_trains()

    spikes = []
    for i in arange(len(spike_trains)):
        spike_list = spike_trains[i] / ms
        spikes.append(spike_list)

    return spikes


def setup_fit(model=None, dt=None, param_init=None, input_var=None,
              metric=None):
    """
    Function sets up simulator in one of the two availabel modes: runtime or
    standalone (set in the script calling fit_traces/fit spikes) and checks
    the variables.

    Verifyies:
        - if dt is set
        - if input variables belong to the model
        - if initialized parameters exsists in the model
        - metric instance

    Returns
    -------
    simulator : object ~brian2tools.modelfitting.Simulator
    """
    simulators = {
        'CPPStandaloneDevice': CPPStandaloneSimulation(),
        'RuntimeDevice': RuntimeSimulation()
    }

    simulator = simulators[get_device().__class__.__name__]

    if dt is None:
        raise Exception('dt (sampling frequency of the input) must be set')
    defaultclock.dt = dt

    if input_var not in model.identifiers:
        raise Exception("%s is not an identifier in the model" % input_var)

    if not (isinstance(metric, Metric) or metric is None):
        raise Exception("metric has to be a child of class Metric or None")

    if param_init:
        for param, val in param_init.items():
            if not (param in model.identifiers or param in model.names):
                raise Exception("%s is not a model variable or an identifier \
                                in the model")

    return simulator


def setup_neuron_group(model, n_neurons, method, threshold, reset, refractory,
                       param_init, **namespace):
    """
    Setup neuron group, initialize required number of neurons, create namespace
    and initite the parameters.

    Returns
    -------
    neurons : object ~brian2.groups.neurongroup.NeuronGroup
        group of neurons

    """
    neurons = NeuronGroup(n_neurons, model, method=method, threshold=threshold,
                          reset=reset, refractory=refractory, name='neurons')
    for name in namespace:
        neurons.namespace[name] = namespace[name]

    if param_init:
        neurons.set_states(param_init)

    return neurons


def calc_errors_spikes(metric, simulator, n_traces, output):
    """
    Returns errors after simulation with SpikeMonitor.
    To be used inside optim_iter.
    """
    spikes = get_spikes(simulator.network['monitor'])
    errors = metric.calc(spikes, output, n_traces)

    return errors


def calc_errors_traces(metric, simulator, n_traces, output, output_var):
    """
    Returns errors after simulation with StateMonitor.
    To be used inside optim_iter.
    """
    traces = getattr(simulator.network['monitor'], output_var)
    errors = metric.calc(traces, output, n_traces)
    return errors


def optim_iter(simulator, optimizer, metric, parameter_names, n_samples,
               n_traces, duration, output, calc_errors, *args):
    """
    Function performs all operations required for one iteration of optimization.
    Drawing parameters, setting them to simulator and calulating the error.

    Returns
    -------
    results : list
        recommended parameters
    parameters: 2D list
        drawn parameters
    errors: list
        calculated errors
    """
    parameters = optimizer.ask(n_samples=n_samples)

    d_param = get_param_dic(parameters, parameter_names, n_traces,
                            n_samples)
    simulator.run(duration, d_param, parameter_names)
    errors = calc_errors(metric, simulator, n_traces, output, *args)

    optimizer.tell(parameters, errors)
    results = optimizer.recommend()

    return results, parameters, errors


def fit_traces(model=None,
               input_var=None,
               input=None,
               output_var=None,
               output=None,
               dt=None,
               method=('linear', 'exponential_euler', 'euler'),
               optimizer=None,
               metric=None,
               n_samples=10,
               n_rounds=1,
               callback=None,
               param_init=None,
               reset=None, refractory=False, threshold=None,
               **params):
    """
    Creates an interface for model fitting of traces with parameters draw by
    gradient-free algorithms (through ask/tell interfaces).
    Input nad output have to have the same dimensions.

    Initiates n_neurons = num input traces * num samples, to which drawn
    parameters get assigned and evaluates them in parallel.

    Parameters
    ----------
    model : `~brian2.equations.Equations` or string
        The equations describing the model.
    input_var : string
        Input variable name.
    input : input data as a 2D array
    output_var : string
        Output variable name.
    output : output data as a 2D array
    dt : time step
    method: string, optional
        Integration method
    optimizer: ~brian2tools.modelfitting.Optimizer children
        Child of Optimizer class, specific for each library.
    metric: ~brian2tools.modelfitting.Metric children
        Child of Metric class, specifies optimization metric
    n_samples: int
        Number of parameter samples to be optimized over.
    n_rounds: int
        Number of rounds to optimize over. (feedback provided over each round)
    callback: callable
        Provide custom feedback function func(results, errors, parameters, index)
        If callback returns True the fitting execution is interrupted.
    param_init: dict
        Dictionary of variables to be initialized with respective value
    **params:
        bounds for each parameter

    Returns
    -------
    result_dict : dict
        dictionary with best parameter set
    error: float
        error value for best parameter set
    """
    if output_var not in model.names:
        raise Exception("%s is not a model variable" % output_var)
        if output.shape != input.shape:
            raise Exception("Input and output must have the same size")

    simulator = setup_fit(model, dt, param_init, input_var, metric)

    parameter_names = model.parameter_names
    Ntraces, Nsteps = input.shape
    duration = Nsteps * dt
    n_neurons = Ntraces * n_samples

    if metric is None:
        model = model + Equations('total_error : %s' % repr(output.dim**2))

    # Replace input variable by TimedArray
    output_traces = TimedArray(output.transpose(), dt=dt)
    input_traces = TimedArray(input.transpose(), dt=dt)
    model = model + Equations(input_var + '= input_var(t, i % Ntraces) :\
                              ' + "% s" % repr(input.dim))

    # Setup NeuronGroup
    neurons = setup_neuron_group(model, n_neurons, method, threshold, reset,
                                 refractory, param_init,
                                 input_var=input_traces,
                                 output_var=output_traces,
                                 Ntraces=Ntraces)

    # Pick Metric, prep env for it
    if metric is None:
        t_start = 0*second
        neurons.namespace['t_start'] = t_start
        neurons.run_regularly('total_error +=  (' + output_var + '-output_var\
                              (t,i % Ntraces))**2 * int(t>=t_start)', when='end')

    # Set up Simulator and Optimizer
    monitor = StateMonitor(neurons, output_var, record=True, name='monitor')
    network = Network(neurons, monitor)
    simulator.initialize(network)
    optimizer.initialize(parameter_names, **params)

    if metric is None:
        def calc_online_error(metric, simulator, output, n_traces, output_var):
            """Calculates error in online fashion"""
            err = simulator.network['neurons'].total_error/int((duration-t_start)/defaultclock.dt)
            err = neurons.total_error/int((duration-t_start)/defaultclock.dt)
            err = mean(err.reshape((n_samples, Ntraces)), axis=1)

            return array(err)

        calc_errors = calc_online_error

    elif isinstance(metric, Metric):
        calc_errors = calc_errors_traces

    # Run Optimization Loop
    for k in range(n_rounds):
        res, parameters, errors = optim_iter(simulator, optimizer, metric,
                                             parameter_names, n_samples,
                                             Ntraces, duration, output,
                                             calc_errors, output_var)

        # create output variables
        result_dict = make_dic(parameter_names, res)
        error = min(errors)

        if callback is not None:
            if callback(res, errors, parameters, k) is True:
                break

    return result_dict, error


def fit_spikes(model=None,
               input_var=None,
               input=None,
               output=None,
               dt=None,
               method=('linear', 'exponential_euler', 'euler'),
               optimizer=None,
               metric=None,
               n_rounds=1,
               n_samples=10,
               callback=None,
               param_init=None,
               reset=None, refractory=False, threshold=None,
               **params):
    """
    Creates an interface for model fitting of spike trains with parameters
    drawn by gradient-free algorithms (through ask/tell interfaces).
    Input nad output dimensions don't have to agree. Output has to contain
    times of spikes. Initiates n_neurons = num input traces * num samples, to
    which drawn parameters get assigned and evaluates them in parallel.

    Parameters
    ----------
    model : `~brian2.equations.Equations` or string
        The equations describing the model.
    input_var : string
        Input variable name.
    input : input data as a 2D array
    output : output data as a 2D list of arrays
    dt : time step
    method: string, optional
        Integration method
    optimizer: ~brian2tools.modelfitting.Optimizer children
        Child of Optimizer class, specific for each library.
    metric: ~brian2tools.modelfitting.Metric children
        Child of Metric class, specifies optimization metric
    n_samples: int
        Number of parameter samples to be optimized over.
    n_rounds: int
        Number of rounds to optimize over. (feedback provided over each round)
    callback: callable
        Provide custom feedback function func(results, errors, parameters, index)
        If callback returns True the fitting execution is interrupted.
    param_init: dict
        Dictionary of variables to be initialized with respective value
    **params:
        bounds for each parameter

    Returns
    -------
    result_dict : dict
        dictionary with best parameter set
    error: float
        error value for best parameter set==
    """
    simulator = setup_fit(model=model, dt=dt, param_init=param_init,
                          input_var=input_var, metric=metric)

    parameter_names = model.parameter_names
    Ntraces, Nsteps = input.shape
    duration = Nsteps * dt

    n_neurons = Ntraces * n_samples

    # Replace input variable by TimedArray
    input_traces = TimedArray(input.transpose(), dt=dt)
    model = model + Equations(input_var + '= input_var(t, i % Ntraces) :\
                               ' + "% s" % repr(input.dim))

    # Population size for differential evolution
    neurons = setup_neuron_group(model, n_neurons, method, threshold, reset,
                                 refractory, param_init,
                                 input_var=input_traces,
                                 Ntraces=Ntraces)

    # Set up Simulator and Optimizer
    monitor = SpikeMonitor(neurons, record=True, name='monitor')
    network = Network(neurons, monitor)
    simulator.initialize(network)
    optimizer.initialize(parameter_names, **params)

    # Run Optimization Loop
    for k in range(n_rounds):
        res, parameters, errors = optim_iter(simulator, optimizer, metric,
                                             parameter_names, n_samples,
                                             Ntraces, duration, output,
                                             calc_errors_spikes)

        # create output variables
        result_dict = make_dic(parameter_names, res)
        error = min(errors)

        if callback is not None:
            if callback(res, errors, parameters, k) is True:
                break

    return result_dict, error
