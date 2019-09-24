import abc

from brian2.units.fundamentalunits import DIMENSIONLESS, get_dimensions
from numpy import ones, array, arange, concatenate, mean, nanmin, reshape
from brian2 import (NeuronGroup,  defaultclock, get_device, Network,
                    StateMonitor, SpikeMonitor, ms, device, second,
                    get_local_namespace, Quantity)
from brian2.input import TimedArray
from brian2.equations.equations import Equations
from .simulator import RuntimeSimulator, CPPStandaloneSimulator
from .metric import Metric, SpikeMetric, TraceMetric
from .optimizer import Optimizer
from .utils import callback_setup, make_dic


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

        if get_device().__class__.__name__ == 'CPPStandaloneDevice':
            if device.has_been_run is True:
                raise Exception("To run another fitter in standalone mode you "
                                "need to create new script")
        if dt is None:
            raise ValueError("dt-sampling frequency of the input must be set")

        if isinstance(model, str):
            model = Equations(model)
        if input_var not in model.identifiers:
            raise NameError("%s is not an identifier in the model" % input_var)

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
        self.network = None
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

    def setup_neuron_group(self, n_neurons, namespace, name='neurons'):
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
        neurons = NeuronGroup(n_neurons, self.model, method=self.method,
                              threshold=self.threshold, reset=self.reset,
                              refractory=self.refractory, name=name,
                              namespace=namespace)

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
        self.results_.append(parameters)
        self.errors.append(errors)

        results = optimizer.recommend()

        return results, parameters, errors

    def fit(self, optimizer, metric=None, n_rounds=1, callback='text',
            restart=False, **params):
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
            ``func(results, errors, parameters, index)``. If this function
            returns ``True`` the fitting execution is interrupted.
        restart: bool
            Flag that reinitializes the Fitter to reset the optimization.
            With restart True user is allowed to change optimizer/metric.
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
            raise TypeError("metric has to be a child of class Optimizer")

        if self.metric is not None and restart is False:
            if metric is not self.metric:
                raise Exception("You can not change the metric between fits")

        if self.optimizer is not None and restart is False:
            if optimizer is not self.optimizer:
                raise Exception("You can not change the optimizer between fits")

        if self.optimizer is None or restart is True:
            self.results_, self.errors = [], []
            optimizer.initialize(self.parameter_names, popsize=self.n_samples,
                                 **params)

        self.optimizer = optimizer
        self.metric = metric

        callback = callback_setup(callback, n_rounds)

        # Run Optimization Loop
        error = None
        for index in range(n_rounds):
            best_params, parameters, errors = self.optimization_iter(optimizer,
                                                                     metric)

            # create output variables
            self.best_params = make_dic(self.parameter_names, best_params)
            error = nanmin(self.errors)

            if callback(parameters, errors, best_params, error, index) is True:
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

        if get_device().__class__.__name__ == 'CPPStandaloneDevice':
            if device.has_been_run is True:
                raise Exception("You need to reset the device before generating "
                                "the traces in standalone mode, which will make "
                                "you lose monitor data add: device.reinit() "
                                "& device.activate()")
        if params is None:
            params = self.best_params

        defaultclock.dt = self.dt
        Ntraces, Nsteps = self.input.shape

        # Setup NeuronGroup
        namespace = get_local_namespace(level=level+1)
        namespace['input_var'] = self.input_traces
        namespace['n_traces'] = Ntraces
        namespace['output_var'] = output_var
        self.neurons = self.setup_neuron_group(Ntraces, namespace,
                                               name='neurons_')

        if output_var == 'spikes':
            monitor = SpikeMonitor(self.neurons, record=True, name='monitor_')
        else:
            monitor = StateMonitor(self.neurons, output_var, record=True,
                                   name='monitor_')
        network = Network(self.neurons, monitor)

        if param_init:
            self.simulator.initialize(network, param_init, name='neurons_')
        else:
            self.simulator.initialize(network, self.param_init,
                                      name='neurons_')

        self.simulator.run(self.duration, params, self.parameter_names,
                           name='neurons_')

        if output_var == 'spikes':
            fits = get_spikes(self.simulator.network['monitor_'],
                              1, self.n_traces)[0]  # a single "sample"
        else:
            fits = getattr(self.simulator.network['monitor_'], output_var)

        return fits


class TraceFitter(Fitter):
    """Input nad output have to have the same dimensions."""
    def __init__(self, model, input_var, input, output_var, output, dt,
                 n_samples=30, method=None, reset=None, refractory=False,
                 threshold=None, level=0, param_init=None):
        """Initialize the fitter."""
        super().__init__(dt, model, input, output, input_var, output_var,
                         n_samples, threshold, reset, refractory, method,
                         param_init)

        if output_var not in self.model.names:
            raise NameError("%s is not a model variable" % output_var)
        if output.shape != input.shape:
            raise ValueError("Input and output must have the same size")

        output_traces = TimedArray(output.transpose(), dt=dt)

        # Setup NeuronGroup
        namespace = get_local_namespace(level=level+1)
        namespace['input_var'] = self.input_traces
        namespace['output_var'] = output_traces
        namespace['n_traces'] = self.n_traces
        self.neurons = self.setup_neuron_group(self.n_neurons, namespace)

        monitor = StateMonitor(self.neurons, output_var, record=True,
                               name='monitor')
        self.network = Network(self.neurons, monitor)

        self.simulator.initialize(self.network, self.param_init)

    def calc_errors(self, metric):
        """
        Returns errors after simulation with StateMonitor.
        To be used inside optim_iter.
        """
        traces = getattr(self.simulator.network['monitor'],
                         self.output_var+'_')
        # Reshape traces for easier calculation of error
        traces = reshape(traces, (traces.shape[0]//self.n_traces,
                                  self.n_traces,
                                  -1))
        errors = metric.calc(traces, self.output, self.dt)
        return errors

    def fit(self, optimizer, metric=None, n_rounds=1, callback='text',
            restart=False, **params):
        if not isinstance(metric, TraceMetric):
            raise TypeError("You can only use TraceMetric child metric with "
                            "TraceFitter")
        self.best_params, error = super().fit(optimizer, metric, n_rounds,
                                              callback, restart, **params)
        return self.best_params, error

    def generate_traces(self, params=None, param_init=None, level=0):
        """Generates traces for best fit of parameters and all inputs"""
        fits = self.generate(params=params, output_var=self.output_var,
                             param_init=param_init, level=level+1)
        return fits


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

        # Setup NeuronGroup
        namespace = get_local_namespace(level=level+1)
        namespace['input_var'] = self.input_traces
        namespace['n_traces'] = self.n_traces
        self.neurons = self.setup_neuron_group(self.n_neurons, namespace)

        monitor = SpikeMonitor(self.neurons, record=True, name='monitor')
        self.network = Network(self.neurons, monitor)

        if param_init:
            for param, val in param_init.items():
                if not (param in self.model.identifiers or param in self.model.names):
                    raise ValueError("%s is not a model variable or an "
                                     "identifier in the model" % param)
            self.param_init = param_init

        self.simulator.initialize(self.network, self.param_init)

    def calc_errors(self, metric):
        """
        Returns errors after simulation with SpikeMonitor.
        To be used inside optim_iter.
        """
        spikes = get_spikes(self.simulator.network['monitor'],
                            self.n_samples, self.n_traces)
        errors = metric.calc(spikes, self.output, self.dt)
        return errors

    def fit(self, optimizer, metric=None, n_rounds=1, callback='text',
            restart=False, **params):
        if not isinstance(metric, SpikeMetric):
            raise TypeError("You can only use SpikeMetric child metric with "
                            "SpikeFitter")
        self.best_params, error = super().fit(optimizer, metric, n_rounds,
                                              callback, restart, **params)
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

        # Setup NeuronGroup
        namespace = get_local_namespace(level=level+1)
        namespace['input_var'] = self.input_traces
        namespace['output_var'] = output_traces
        namespace['n_traces'] = self.n_traces
        self.neurons = self.setup_neuron_group(self.n_neurons, namespace)

        self.t_start = 0*second
        self.neurons.namespace['t_start'] = self.t_start
        self.neurons.run_regularly('total_error +=  (' + output_var + '-output_var\
                                   (t,i % n_traces))**2 * int(t>=t_start)',
                                   when='end')

        monitor = StateMonitor(self.neurons, output_var, record=True,
                               name='monitor')
        self.network = Network(self.neurons, monitor)
        if param_init:
            for param, val in param_init.items():
                if not (param in self.model.identifiers or param in self.model.names):
                    raise ValueError("%s is not a model variable or an "
                                     "identifier in the model" % param)
            self.param_init = param_init

        self.simulator.initialize(self.network, self.param_init)

    def calc_errors(self, metric=None):
        """Calculates error in online fashion.To be used inside optim_iter."""
        errors = self.neurons.total_error/int((self.duration-self.t_start)/defaultclock.dt)
        errors = mean(errors.reshape((self.n_samples, self.n_traces)), axis=1)
        return array(errors)

    def generate_traces(self, params=None, param_init=None, level=0):
        """Generates traces for best fit of parameters and all inputs"""
        fits = self.generate(params=params, output_var=self.output_var,
                             param_init=param_init, level=level+1)
        return fits
