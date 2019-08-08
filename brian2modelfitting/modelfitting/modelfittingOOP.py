import abc
from types import FunctionType
from numpy import mean, ones, array, arange
from brian2 import (NeuronGroup,  defaultclock, get_device, Network,
                    StateMonitor, SpikeMonitor, ms, second)
from brian2.input import TimedArray
from brian2.equations.equations import Equations
from .simulation import RuntimeSimulation, CPPStandaloneSimulation
from .metric import Metric


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
        if output_var not in model.names:
            raise Exception("%s is not a model variable" % output_var)
        if output.shape != input.shape:
            raise Exception("Input and output must have the same size")

        # simulator = setup_fit(model, dt, param_init, input_var, metric)

        parameter_names = model.parameter_names
        Ntraces, Nsteps = input.shape
        duration = Nsteps * dt
        # n_neurons = Ntraces * n_samples

    def setup(self):
        pass

    @abc.abstractmethod
    def calc_errors(self):
        pass

    def optimization_iter(self):
        pass

    @abc.abstractmethod
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

    @abc.abstractmethod
    def results(self):
        """Returns all of the so far gathered results"""
        pass

    def generate_traces(self):
        """Generates traces for best fit of parameters and all inputs"""
        pass

class SpikeFitter(Fitter):
    def __init__(self, **kwds):
        """Initialize the fitter."""
        pass

class TraceFitter(Fitter):
    def __init__(self, **kwds):
        """Initialize the fitter."""
        pass
