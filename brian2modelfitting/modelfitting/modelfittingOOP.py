import abc
from types import FunctionType
from numpy import mean, ones, array, arange
from brian2 import (NeuronGroup,  defaultclock, get_device, Network,
                    StateMonitor, SpikeMonitor, ms, second, device)
from brian2.input import TimedArray
from brian2.equations.equations import Equations
from .simulation import RuntimeSimulation, CPPStandaloneSimulation
from .metric import Metric
from tqdm.autonotebook import tqdm


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
def calc_errors_spikes(metric, simulator, n_traces, output):
    """
    Returns errors after simulation with SpikeMonitor.
    To be used inside optim_iter.
    """
    spikes = get_spikes(simulator.network['monitor'])
    errors = metric.calc(spikes, output, n_traces)

    return errors


class Fitter(object):
    """
    Abstract Fitter class for model fitting applications.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, dt, model, input, output, input_var, output_var, n_samples,
                 threshold, reset, refractory, method):
        """Initialize the fitter."""
        if dt is None:
            raise Exception('dt (sampling frequency of the input) must be set')
        defaultclock.dt = dt
        self.simulator = self.setup_fit()

        parameter_names = model.parameter_names
        n_traces, n_steps = input.shape
        duration = n_steps * dt
        n_neurons = n_traces * n_samples

        # Replace input variable by TimedArray
        output_traces = TimedArray(output.transpose(), dt=dt)
        input_traces = TimedArray(input.transpose(), dt=dt)
        model = model + Equations(input_var + '= input_var(t, i % n_traces) :\
                                  ' + "% s" % repr(input.dim))
        self.model = model
        self.method = method
        self.threshold = threshold
        self.reset = reset
        self.refractory = refractory
        self.parameter_names = parameter_names
        self.duration = duration
        self.n_samples = n_samples
        self.n_traces = n_traces
        self.output = output
        self.output_var = output_var
        self.input_traces = input_traces
        self.input = input

        # Setup NeuronGroup
        self.neurons = self.setup_neuron_group(n_neurons,
                                               input_var=input_traces,
                                               output_var=output_traces,
                                               n_traces=n_traces)

    def setup_fit(self):
        simulators = {
            'CPPStandaloneDevice': CPPStandaloneSimulation(),
            'RuntimeDevice': RuntimeSimulation()
        }

        simulator = simulators[get_device().__class__.__name__]
        return simulator

    def setup_neuron_group(self, n_neurons, **namespace):
        neurons = NeuronGroup(n_neurons, self.model, method=self.method,
                              threshold=self.threshold, reset=self.reset,
                              refractory=self.refractory, name='neurons')
        for name in namespace:
            neurons.namespace[name] = namespace[name]

        return neurons

    @abc.abstractmethod
    def calc_errors(self):
        pass

    def optimization_iter(self, optimizer, metric, param_init, *args):
        """
        Function performs all operations required for one iteration of
        optimization. Drawing parameters, setting them to simulator and
        calulating the error.

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
                self.network['neurons'].__setattr__(k, v)

        parameters = optimizer.ask(n_samples=self.n_samples)

        d_param = get_param_dic(parameters, self.parameter_names, self.n_traces,
                                self.n_samples)
        self.simulator.run(self.duration, d_param, self.parameter_names)
        errors = self.calc_errors(metric)

        optimizer.tell(parameters, errors)
        results = optimizer.recommend()

        return results, parameters, errors


    def fit(self, optimizer=None, metric=None,
            n_samples=10,
            n_rounds=1,
            callback='progressbar',
            param_init=None,
            **params):
        """
        Run the optimization algorithm for given amount of rounds with given
        number of samples drawn.
        Return best result and it's error.
        """

        if not (isinstance(metric, Metric) or metric is None):
            raise Exception("metric has to be a child of class Metric or None")

        if param_init:
            for param, val in param_init.items():
                if not (param in self.model.identifiers or param in self.model.names):
                    raise Exception("%s is not a model variable or an identifier \
                                    in the model")

        callback = callback_setup(callback, n_rounds)
        optimizer.initialize(self.parameter_names, **params)

        # Run Optimization Loop
        for k in range(n_rounds):
            res, parameters, errors = self.optimization_iter(optimizer, metric,
                                                             param_init)

            # create output variables
            self.results = make_dic(self.parameter_names, res)
            error = min(errors)

            if callback(res, errors, parameters, k) is True:
                break

        return self.results, error

    def results(self):
        """
        Returns all of the so far gathered results
        In one of the 3 formats: dataframe, list, dict.
        """
        pass

    def generate(self, params=None, output_var=None, param_init=None):
        """
        Generates traces for best fit of parameters and all inputs.
        If provided with other parameters provides those.
        """
        if params is None:
            params = self.results

        if get_device().__class__.__name__ == 'CPPStandaloneDevice':
            device.has_been_run = False
            # reinit_devices()
            device.reinint()
            device.activate()

        Ntraces, Nsteps = self.input.shape
        self.neurons = self.setup_neuron_group(Ntraces,
                                               input_var=self.input_traces,
                                               output_var=output_var,
                                               n_traces=Ntraces)

        if output_var == 'spikes':
            monitor = SpikeMonitor(self.neurons, record=True, name='monitor')
        else:
            monitor = StateMonitor(self.neurons, output_var, record=True,
                                   name='monitor')
        network = Network(self.neurons, monitor)
        self.simulator.initialize(self.network)

        if param_init:
            for k, v in param_init.items():
                network['neurons'].__setattr__(k, v)

        self.simulator.initialize(network)
        self.simulator.run(self.duration, params, self.parameter_names)

        if output_var == 'spikes':
            fits = get_spikes(self.simulator.network['monitor'])
        else:
            fits = getattr(self.simulator.network['monitor'], output_var)

        return fits


class TraceFitter(Fitter):
    def __init__(self, model=None, input_var=None, input=None,
                 output_var=None, output=None, dt=None, method=None,
                 reset=None, refractory=False, threshold=None,
                 callback=None, n_samples=None):
        """Initialize the fitter."""
        super().__init__(dt, model, input, output, input_var, output_var,
                         n_samples, threshold, reset, refractory, method)

        if input_var not in model.identifiers:
            raise Exception("%s is not an identifier in the model" % input_var)

        if output_var not in model.names:
            raise Exception("%s is not a model variable" % output_var)
            if output.shape != input.shape:
                raise Exception("Input and output must have the same size")

        # Set up Simulator and Optimizer
        monitor = StateMonitor(self.neurons, output_var, record=True,
                               name='monitor')
        self.network = Network(self.neurons, monitor)
        self.simulator.initialize(self.network)

    def calc_errors(self, metric):
        """
        Returns errors after simulation with StateMonitor.
        To be used inside optim_iter.
        """
        traces = getattr(self.simulator.network['monitor'], self.output_var)
        errors = metric.calc(traces, self.output, self.n_traces)
        return errors

    def generate_traces(self):
        """Generates traces for best fit of parameters and all inputs"""
        pass


class SpikeFitter(Fitter):
    def __init__(self, **kwds):
        """Initialize the fitter."""
        pass

    def generate_spikes(self):
        """Generates traces for best fit of parameters and all inputs"""
        pass
