import abc
from brian2 import Hz, second
from brian2.units.fundamentalunits import check_units

from numpy import (array, sum, square, reshape, abs, amin, digitize,
                   rint, arange, atleast_2d, NaN, float64)


def firing_rate(spikes):
    """Raturns rate of the spike train"""
    if len(spikes) < 2:
        return NaN
    return (len(spikes) - 1) / (spikes[-1] - spikes[0])


def get_gamma_factor(source, target, delta, dt):
    """
    Calculate gamma factor between source and tagret spike trains,
    with precision delta.

    Parameters
    ----------
    source: list/array
        source trace, goal performance
    target: list/array
        target trace
    delta: float * ms
        time window
    dt: float * ms
        time step

    Returns
    -------
        gamma factor: float
    """
    source = array(source)
    target = array(target)
    target_rate = firing_rate(target) * Hz

    source = array(rint(source / dt), dtype=int)
    target = array(rint(target / dt), dtype=int)
    delta_diff = int(rint(delta / dt))

    source_length = len(source)
    target_length = len(target)

    if (source_length > 1):
        bins = .5 * (source[1:] + source[:-1])
        indices = digitize(target, bins)
        diff = abs(target - source[indices])
        matched_spikes = (diff <= delta_diff)
        coincidences = sum(matched_spikes)
    else:
        indices = [amin(abs(source - target[i])) <= delta_diff for i in arange(target_length)]
        coincidences = sum(indices)

    # Normalization of the coincidences count
    NCoincAvg = 2 * delta * target_length * target_rate
    norm = .5*(1 - 2 * target_rate * delta)
    gamma = (coincidences - NCoincAvg)/(norm*(source_length + target_length))

    return gamma


class Metric(object):
    """
    Metic abstract class to define functions required for a custom metric
    To be used with modelfitting fit_traces.

    TODO: metric.weights
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, **kwds):
        """Initialize the metric."""
        pass

    @abc.abstractmethod
    def get_features(self, traces, output):
        """
        Function calculates features / errors for each of the traces and stores
        it in an attibute metric.features


        Parameters
        ----------
        traces: 2D array
            traces to be evaluated
        output: array
            goal traces
        """
        pass

    @abc.abstractmethod
    def get_errors(self, features, n_traces):
        """
        Function weights features/multiple errors into one final error per each
        set of parameters and inputs stored metric.errors.

        Parameters
        ----------
        features: 2D array
            features set for each simulated trace
        n_traces: int
            number of input traces
        """
        pass

    @abc.abstractmethod
    def calc(self, traces, output, n_traces):
        """
        Perform the error calculation across all parameters,
        calculate error between each output trace and corresponding
        simulation. You can also access metric.features, metric.errors.

        Parameters
        ----------
        traces: 2D array
            traces to be evaluated
        output: array
            goal traces
        n_traces:
            number of input traces

        Returns
        -------
        errors: array
            weigheted/mean error for each set of parameters

        """
        self.get_features(traces, output, n_traces)
        self.get_errors(self.features, n_traces)

        return self.errors


class MSEMetric(Metric):
    __doc__ = "Mean Square Error between goal and calculated output." + \
              Metric.get_features.__doc__

    def __init__(self):
        super(Metric, self).__init__()

    def get_features(self, traces, output, n_traces):
        mselist = []
        output = atleast_2d(output)

        for i in arange(n_traces):
            temp_out = output[i]
            temp_traces = traces[i::n_traces]

            for trace in temp_traces:
                mse = sum(square(temp_out - trace))
                mselist.append(mse)

        self.features = mselist

    def get_errors(self, features, n_traces):
        feat_arr = reshape(array(features), (n_traces,
                           int(len(features)/n_traces)))
        self.errors = feat_arr.mean(axis=0)


class GammaFactor(Metric):
    __doc__ = """
    Calculate gamma factors between goal and calculated spike trains, with
    precision delta.

    Reference:
    R. Jolivet et al., 'A benchmark test for a quantitative assessment of
    simple neuron models',
    Journal of Neuroscience Methods 169, no. 2 (2008): 417-424.
    """ + Metric.get_features.__doc__

    @check_units(dt=second, delta=second)
    def __init__(self, dt, delta=None):
        """
        Initialize the metric with time window delta and time step dt
output
        Parameters
        ----------
        dt: time step [ms]
        delta: time window [ms]
        """
        super(Metric, self)
        if delta is None:
            raise AssertionError('delta (time window for gamma factor), \
                                 has to be set to ms')
        self.delta = delta
        self.dt = dt

    def get_features(self, traces, output, n_traces):
        gamma_factors = []
        if type(output[0]) == float64:
            output = atleast_2d(output)

        for i in arange(n_traces):
            temp_out = output[i]
            temp_traces = traces[i::n_traces]

            for trace in temp_traces:
                gf = get_gamma_factor(trace, temp_out, self.delta, self.dt)
                gamma_factors.append(abs(1 - gf))

        self.features = gamma_factors

    def get_errors(self, features, n_traces):
        feat_arr = reshape(array(features), (n_traces,
                           int(len(features)/n_traces)))
        self.errors = feat_arr.mean(axis=0)
