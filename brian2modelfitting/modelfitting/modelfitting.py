import abc
from numpy import ones, array, arange, concatenate, mean
from brian2 import (NeuronGroup,  defaultclock, get_device, Network,
                    StateMonitor, SpikeMonitor, ms, device, second,
                    get_local_namespace, Quantity)
from brian2.input import TimedArray
from brian2.equations.equations import Equations
from .simulation import RuntimeSimulation, CPPStandaloneSimulation
from .metric import Metric
from .optimizer import Optimizer
from .utils import callback_setup, make_dic


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


def setup_fit():
    """
    Function sets up simulator in one of the two availabel modes: runtime
    or standalone.

    Returns
    -------
    simulator : object ~brian2modelfitting.modelfitting.Simulator
    """
    simulators = {
        'CPPStandaloneDevice': CPPStandaloneSimulation(),
        'RuntimeDevice': RuntimeSimulation()
    }

    return simulators[get_device().__class__.__name__]

class Fitter(metaclass=abc.ABCMeta):
    """
    Base Fitter class for model fitting applications.

    Creates an interface for model fitting of traces with parameters draw by
    gradient-free algorithms (through ask/tell interfaces).

    Initiates n_neurons = num input traces * num samples, to which drawn
    parameters get assigned and evaluates them in parallel.

    Parameters
    ----------
    dt : time step
    model : `~brian2.equations.Equations` or string
        The equations describing the model.
    input : input data as a 2D array
    output : output data as a 2D array
    input_var : string
        Input variable name.
    output_var : string
        Output variable name.
    n_samples: int
        Number of parameter samples to be optimized over.
    threshold: str, optional
        The condition which produces spikes. Should be a single line boolean
        expression.
    reset: str, optional
        The (possibly multi-line) string with the code to execute on reset.
    refractory: {str, 'Quantity'}, optional
        Either the length of the refractory period (e.g. 2*ms), a string
        expression that evaluates to the length of the refractory period after
        each spike (e.g. '(1 + rand())*ms'), or a string expression evaluating
        to a boolean value, given the condition under which the neuron stays
        refractory after a spike (e.g. 'v > -20*mV')
    method: string, optional
        Integration method
    level : int, optional
        How much farther to go down in the stack to find the namespace.
    """
    def __init__(self, dt, model, input, output, input_var, output_var,
                 n_samples, threshold, reset, refractory, method, level=0):
        """Initialize the fitter."""

        if get_device().__class__.__name__ == 'CPPStandaloneDevice':
            if device.has_been_run is True:
                raise Exception("To run another fitter in standalone mode you need \
                                 to create new script")
        if method is None: method = 'exponential_euler'
        if dt is None:
            raise ValueError('dt (sampling frequency of the input) must be set')
        defaultclock.dt = dt
        self.dt = dt

        self.results_, self.errors = [], []
        self.simulator = setup_fit()

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
        self.output = output
        self.output_var = output_var

        # initialization of attributes used later
        self.best_res = None
        self.input_traces = None
        self.model = None
        self.network = None

        self.optimizer = None
        self.metric = None



    def setup_neuron_group(self, n_neurons, namespace, name='neurons'):
        """
        Setup neuron group, initialize required number of neurons, create
        namespace and initite the parameters.

        Parameters
        ----------
        n_neurons: int
            number of required neurons
        **namespace :
            arguments to be added to NeuronGroup namespace

        Returns
        -------
        neurons : object ~brian2.groups.neurongroup.NeuronGroup
            group of neurons

        """
        neurons = NeuronGroup(n_neurons, self.model, method=self.method,
                              threshold=self.threshold, reset=self.reset,
                              refractory=self.refractory, name=name, namespace=namespace)

        return neurons

    @abc.abstractmethod
    def calc_errors(self, metric):
        """
        Abstract method required in all Fitter classes, used for
        calculating errors

        Parameters
        ----------
        metric:~brian2modelfitting.modelfitting.Metric children
            Child of Metric class, specifies optimization metric
        """
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

        d_param = get_param_dic(parameters, self.parameter_names,
                                self.n_traces, self.n_samples)
        self.simulator.run(self.duration, d_param, self.parameter_names)
        errors = self.calc_errors(metric)

        optimizer.tell(parameters, errors)
        self.results_.append(parameters)
        self.errors.append(errors)

        results = optimizer.recommend()

        return results, parameters, errors

    def fit(self, optimizer=None, metric=None,
            n_rounds=1,
            callback='text',
            param_init=None,
            restart=False,
            **params):
        """
        Run the optimization algorithm for given amount of rounds with given
        number of samples drawn. Return best set of parameters and
        corresponding error.

        Parameters
        ----------
        optimizer: ~brian2modelfitting.modelfitting.Optimizer children
            Child of Optimizer class, specific for each library.
        metric: ~brian2modelfitting.modelfitting.Metric children
            Child of Metric class, specifies optimization metric
        n_rounds: int
            Number of rounds to optimize over (feedback provided over each
            round).
        callback: str('text' or 'progressbar') or callable
            For strings outputs default feedback or a progressbar. Provide
            custom feedback function func(results, errors, parameters, index)
            If callback returns True the fitting execution is interrupted.
        param_init: dict
            Dictionary of variables to be initialized with respective value
        restart bool
            Flag that reinitializes the Fitter to reset the optimization.
            With restart True user is allowed to change optimizer/metric.
        **params:
            bounds for each parameter

        Returns
        -------
        best_results : dict
            dictionary with best parameter set
        error: float
            error value for best parameter set
        """
        if param_init:
            for param, val in param_init.items():
                if not (param in self.model.identifiers or param in self.model.names):
                    raise ValueError("%s is not a model variable or an \
                                      identifier in the model")

        if not (isinstance(metric, Metric)) or metric is None:
            raise TypeError("metric has to be a child of class Metric")

        if not (isinstance(optimizer, Optimizer)) or optimizer is None:
            raise TypeError("metric has to be a child of class Optimizer")

        if not self.metric is None and restart is False:
            if not metric is self.metric:
                raise Exception("You can not change the metric between fits")

        if not self.optimizer is None and restart is False:
            if not optimizer is self.optimizer:
                raise Exception("You can not change the optimizer between fits")

        if self.optimizer is None or restart is True:
            self.results_, self.errors = [], []
            optimizer.initialize(self.parameter_names, **params)

        self.optimizer = optimizer
        self.metric = metric

        callback = callback_setup(callback, n_rounds)

        # Run Optimization Loop
        for k in range(n_rounds):
            res, parameters, errors = self.optimization_iter(optimizer, metric,
                                                             param_init)

            # create output variables
            self.best_res = make_dic(self.parameter_names, res)
            error = min(errors)

            if callback(res, errors, parameters, k) is True:
                break

        return self.best_res, error

    def results(self, format='list'):
        """
        Returns all of the gathered results (parameters and errors).
        In one of the 3 formats: 'dataframe', 'list', 'dict'.

        Parameters
        ----------
        format: string ('dataframe', 'list', 'dict')
            string with output format

        Returns
        -------
        results:
            'dataframe': returns pandas `DataFrame` without units
            'list': list of dictionaries
            'dict': dictionary of lists
        """
        names = list(self.parameter_names)
        names.append('errors')

        params = array(self.results_)
        params = params.reshape(-1, params.shape[-1])

        errors = array([array(self.errors).flatten()])
        data = concatenate((params, errors.transpose()), axis=1)
        dim = self.model.dimensions

        if format == 'list':
            res_list = []
            for j in arange(0, len(params)):
                temp_data = data[j]
                res_dict = dict()

                for i,n in enumerate(names[:-1]):
                    res_dict[n] = Quantity(temp_data[i], dim=dim[n])
                res_dict[names[-1]] = temp_data[-1]
                res_list.append(res_dict)

            return res_list

        elif format == 'dict':
            res_dict = dict()
            for i,n in enumerate(names[:-1]):
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
        level : int, optional
            How much farther to go down in the stack to find the namespace.
        """

        if get_device().__class__.__name__ == 'CPPStandaloneDevice':
            if device.has_been_run is True:
                raise Exception("You need to reset the device before generating the traces\
                                in standalone mode, which will make you lose monitor data\
                                add: device.reinit() & device.activate()")
        if params is None:
            params = self.best_res

        defaultclock.dt = self.dt
        Ntraces, Nsteps = self.input.shape

        # Setup NeuronGroup
        namespace = get_local_namespace(level=level+1)
        namespace['input_var'] = self.input_traces
        namespace['n_traces'] = Ntraces
        namespace['output_var'] = output_var
        self.neurons = self.setup_neuron_group(Ntraces, namespace, name='neurons_')

        if output_var == 'spikes':
            monitor = SpikeMonitor(self.neurons, record=True, name='monitor_')
        else:
            monitor = StateMonitor(self.neurons, output_var, record=True,
                                   name='monitor_')
        network = Network(self.neurons, monitor)
        self.simulator.initialize(self.network)

        if param_init:
            for k, v in param_init.items():
                network['neurons_'].__setattr__(k, v)

        self.simulator.initialize(network, name='neurons_')
        self.simulator.run(self.duration, params, self.parameter_names, name='neurons_')

        if output_var == 'spikes':
            fits = get_spikes(self.simulator.network['monitor_'])
        else:
            fits = getattr(self.simulator.network['monitor_'], output_var)

        return fits


class TraceFitter(Fitter):
    """Input nad output have to have the same dimensions."""
    def __init__(self, model=None, input_var=None, input=None,
                 output_var=None, output=None, dt=None, method=None,
                 reset=None, refractory=False, threshold=None,
                 callback=None, n_samples=None, level=0):
        """Initialize the fitter."""
        super().__init__(dt, model, input, output, input_var, output_var,
                         n_samples, threshold, reset, refractory, method)

        if input_var not in model.identifiers:
            raise Exception("%s is not an identifier in the model" % input_var)

        if output_var not in model.names:
            raise Exception("%s is not a model variable" % output_var)
        if output.shape != input.shape:
            raise Exception("Input and output must have the same size")

        # Replace input variable by TimedArray
        output_traces = TimedArray(output.transpose(), dt=dt)
        input_traces = TimedArray(input.transpose(), dt=dt)
        model = model + Equations(input_var + '= input_var(t, i % n_traces) :\
                                  ' + "% s" % repr(input.dim))

        self.input_traces = input_traces
        self.model = model

        # Setup NeuronGroup
        namespace = get_local_namespace(level=level+1)
        namespace['input_var'] = input_traces
        namespace['output_var'] = output_traces
        namespace['n_traces'] = self.n_traces
        self.neurons = self.setup_neuron_group(self.n_neurons, namespace)

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

    def generate_traces(self, params=None, param_init=None, level=0):
        """Generates traces for best fit of parameters and all inputs"""
        fits = self.generate(params=params, output_var=self.output_var,
                             param_init=param_init, level=level+1)
        return fits


class SpikeFitter(Fitter):
    def __init__(self, model=None, input_var='I', input=None,
                 output_var='v', output=None, dt=None, method=None,
                 reset=None, refractory=False, threshold=None,
                 callback=None, n_samples=None, level=0):
        """Initialize the fitter."""
        super().__init__(dt, model, input, output, input_var, output_var,
                         n_samples, threshold, reset, refractory, method)

        if input_var not in model.identifiers:
            raise Exception("%s is not an identifier in the model" % input_var)

        # Replace input variable by TimedArray
        input_traces = TimedArray(input.transpose(), dt=dt)
        model = model + Equations(input_var + '= input_var(t, i % n_traces) :\
                                   ' + "% s" % repr(input.dim))

        self.input_traces = input_traces
        self.model = model

        # Setup NeuronGroup
        namespace = get_local_namespace(level=level+1)
        namespace['input_var'] = input_traces
        namespace['n_traces'] = self.n_traces
        self.neurons = self.setup_neuron_group(self.n_neurons, namespace)

        monitor = SpikeMonitor(self.neurons, record=True, name='monitor')
        self.network = Network(self.neurons, monitor)
        self.simulator.initialize(self.network)

    def calc_errors(self, metric):
        """
        Returns errors after simulation with SpikeMonitor.
        To be used inside optim_iter.
        """
        spikes = get_spikes(self.simulator.network['monitor'])
        errors = metric.calc(spikes, self.output, self.n_traces)
        return errors

    def generate_spikes(self, params=None, param_init=None, level=0):
        """Generates traces for best fit of parameters and all inputs"""
        fits = self.generate(params=params, output_var='spikes',
                             param_init=param_init, level=level+1)
        return fits


class OnlineTraceFitter(Fitter):
    """Input nad output have to have the same dimensions."""
    def __init__(self, model=None, input_var=None, input=None,
                 output_var=None, output=None, dt=None, method=None,
                 reset=None, refractory=False, threshold=None,
                 callback=None, n_samples=None, level=0):
        """Initialize the fitter."""
        super().__init__(dt, model, input, output, input_var, output_var,
                         n_samples, threshold, reset, refractory, method)

        if input_var not in model.identifiers:
            raise Exception("%s is not an identifier in the model" % input_var)

        if output_var not in model.names:
            raise Exception("%s is not a model variable" % output_var)
        if output.shape != input.shape:
            raise Exception("Input and output must have the same size")

        # Replace input variable by TimedArray
        output_traces = TimedArray(output.transpose(), dt=dt)
        input_traces = TimedArray(input.transpose(), dt=dt)
        model = model + Equations(input_var + '= input_var(t, i % n_traces) :\
                                  ' + "% s" % repr(input.dim))
        model = model + Equations('total_error : %s' % repr(output.dim**2))

        self.input_traces = input_traces
        self.model = model

        # Setup NeuronGroup
        namespace = get_local_namespace(level=level+1)
        namespace['input_var'] = input_traces
        namespace['output_var'] = output_traces
        namespace['n_traces'] = self.n_traces
        self.neurons = self.setup_neuron_group(self.n_neurons, namespace)

        self.t_start = 0*second
        self.neurons.namespace['t_start'] = self.t_start
        self.neurons.run_regularly('total_error +=  (' + output_var + '-output_var\
                                   (t,i % n_traces))**2 * int(t>=t_start)', when='end')

        monitor = StateMonitor(self.neurons, output_var, record=True,
                               name='monitor')
        self.network = Network(self.neurons, monitor)
        self.simulator.initialize(self.network)

    def calc_errors(self, metric=None):
        """Calculates error in online fashion.To be used inside optim_iter."""
        errors = self.simulator.network['neurons'].total_error/int((self.duration-self.t_start)/defaultclock.dt)
        errors = self.neurons.total_error/int((self.duration-self.t_start)/defaultclock.dt)
        errors = mean(errors.reshape((self.n_samples, self.n_traces)), axis=1)
        return array(errors)

    def generate_traces(self, params=None, param_init=None, level=0):
        """Generates traces for best fit of parameters and all inputs"""
        fits = self.generate(params=params, output_var=self.output_var,
                             param_init=param_init, level=level+1)
        return fits
