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

    def __init__(self, **kwds):
        """Initialize the fitter."""
        pass

    def setup(self):
        pass

    @abc.abstractmethod
    def calc_errors(self):
        pass

    def optimization_iter(self):
        pass

    @abc.abstractmethod
    def run(self):
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
