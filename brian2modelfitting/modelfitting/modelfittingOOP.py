import abc
from types import FunctionType
from numpy import mean, ones, array, arange
from brian2 import (NeuronGroup,  defaultclock, get_device, Network,
                    StateMonitor, SpikeMonitor, ms, second)
from brian2.input import TimedArray
from brian2.equations.equations import Equations
from .simulation import RuntimeSimulation, CPPStandaloneSimulation
from .metric import Metric
from tqdm import tqdm


def callback_text(res, errors, parameters, k):
    print("Round {}: fit {} with error: {}".format(k, res, min(errors)))


class ProgressBar(object):
    def __init__(self, toolbar_width=10):
        self.toolbar_width = toolbar_width
        self.t = tqdm(total=toolbar_width)

    def __call__(self, res, errors, parameters, k):
        self.t.update(1)


def callback_setup(set_type, n_rounds):
    if set_type == 'text':
        callback = callback_text
    elif set_type == 'progressbar':
        callback = ProgressBar(n_rounds)
    elif set_type is not None:
        callback = set_type

    return callback


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


### temp functions


# def setup_fit(model=None, dt=None, param_init=None, input_var=None, metric=None):
def setup_fit(model=None, dt=None, input_var=None):
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

    # if not (isinstance(metric, Metric) or metric is None):
    #     raise Exception("metric has to be a child of class Metric or None")

    # if param_init:
    #     for param, val in param_init.items():
    #         if not (param in model.identifiers or param in model.names):
    #             raise Exception("%s is not a model variable or an identifier \
    #                             in the model")

    return simulator


def setup_neuron_group(model, n_neurons, method, threshold, reset, refractory,
                       **namespace):
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
               n_traces, duration, output, calc_errors, network, param_init,
               *args):
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
    if param_init:
        for k, v in param_init.items():
            network['neurons'].__setattr__(k, v)

    parameters = optimizer.ask(n_samples=n_samples)

    d_param = get_param_dic(parameters, parameter_names, n_traces,
                            n_samples)
    simulator.run(duration, d_param, parameter_names)
    errors = calc_errors(metric, simulator, n_traces, output, *args)

    optimizer.tell(parameters, errors)
    results = optimizer.recommend()

    return results, parameters, errors



class Fitter(object):
    """
    Abstract Fitter class for model fitting applications.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self,
                 model=None,
                 input_var=None, input=None,
                 output_var=None, output=None,
                 dt=None, method=None,
                 reset=None, refractory=False, threshold=None,
                 **params):
        """Initialize the fitter."""
        pass

    def setup(self):
        pass

    # @abc.abstractmethod
    # def calc_errors(self):
    #     pass

    def optimization_iter(self):
        pass

    # @abc.abstractmethod
    def run(self,
            optimizer=None,
            metric=None,
            n_samples=10,
            n_rounds=1,
            callback=None,
            param_init=None):
        """
        Run the optimization algorithm for given amount of rounds with given
        number of samples drawn.
        Return best result and it's error.
        """
        pass

    def results(self):
        """Returns all of the so far gathered results"""
        pass

    def generate_traces(self):
        """Generates traces for best fit of parameters and all inputs"""
        pass

class TraceFitter(Fitter):
    def __init__(self, model=None, input_var=None, input=None,
                 output_var=None, output=None, dt=None, method=None,
                 reset=None, refractory=False, threshold=None,
                 callback=None, n_samples=None):
        """Initialize the fitter."""
        if output_var not in model.names:
            raise Exception("%s is not a model variable" % output_var)
            if output.shape != input.shape:
                raise Exception("Input and output must have the same size")

        simulator = setup_fit(model, dt, input_var)

        parameter_names = model.parameter_names
        Ntraces, Nsteps = input.shape
        duration = Nsteps * dt
        n_neurons = Ntraces * n_samples


        # Replace input variable by TimedArray
        output_traces = TimedArray(output.transpose(), dt=dt)
        input_traces = TimedArray(input.transpose(), dt=dt)
        model = model + Equations(input_var + '= input_var(t, i % Ntraces) :\
                                  ' + "% s" % repr(input.dim))

        # Setup NeuronGroup
        neurons = setup_neuron_group(model, n_neurons, method, threshold, reset,
                                     refractory,
                                     input_var=input_traces,
                                     output_var=output_traces,
                                     Ntraces=Ntraces)

        # Set up Simulator and Optimizer
        monitor = StateMonitor(neurons, output_var, record=True, name='monitor')
        network = Network(neurons, monitor)
        simulator.initialize(network)

        self.simulator = simulator
        self.parameter_names = parameter_names
        self.n_samples = n_samples
        self.Ntraces = Ntraces
        self.duration = duration
        self.output = output
        self.network = network
        self.output_var = output_var


    def run(self, optimizer=None, metric=None,
            n_samples=10,
            n_rounds=1,
            callback='progressbar',
            param_init=None,
            **params):

        callback = callback_setup(callback, n_rounds)

        simulator = self.simulator
        parameter_names = self.parameter_names
        n_samples = self.n_samples
        Ntraces = self.Ntraces
        duration = self.duration
        output = self.output
        network = self.network
        output_var = self.output_var

        optimizer.initialize(parameter_names, **params)

        # Run Optimization Loop
        for k in range(n_rounds):
            res, parameters, errors = optim_iter(simulator, optimizer, metric,
                                                 parameter_names, n_samples,
                                                 Ntraces, duration, output,
                                                 calc_errors_traces, network,
                                                 param_init, output_var)

            # create output variables
            result_dict = make_dic(parameter_names, res)
            error = min(errors)

            if callback(res, errors, parameters, k) is True:
                break

        return result_dict, error


class SpikeFitter(Fitter):
    def __init__(self, **kwds):
        """Initialize the fitter."""
        pass
