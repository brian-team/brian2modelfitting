import abc
import efel
from itertools import repeat
from brian2 import Hz, second, Quantity
from brian2.units.fundamentalunits import check_units
from numpy import (array, sum, square, reshape, abs, amin, digitize,
                   rint, arange, atleast_2d, NaN, float64, split, shape,
                   asarray)


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


def calc_eFEL(traces, inp_times, feat_list):
    out_traces = []
    for i, trace in enumerate(traces):
        time = arange(0, len(trace)/10, 0.1)
        temp_trace = {}
        temp_trace['T'] = time
        temp_trace['V'] = trace
        temp_trace['stim_start'] = [inp_times[i][0]]
        temp_trace['stim_end'] = [inp_times[i][1]]
        out_traces.append(temp_trace)

    results = efel.getFeatureValues(out_traces, feat_list)

    return results


class Metric(metaclass=abc.ABCMeta):
    """
    Metic abstract class to define functions required for a custom metric
    To be used with modelfitting Fitters.
    """
    @abc.abstractmethod
    def get_features(self, traces, output, n_traces):
        """
        Function calculates features / errors for each of the traces and stores
        it in an attribute metric.features

        Parameters
        ----------
        traces: 2D array
            traces to be evaluated
        output: array
            goal traces
        n_traces: int
            number of input traces
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
        features = self.get_features(traces, output, n_traces)
        errors = self.get_errors(features, n_traces)

        return errors


class MSEMetric(Metric):
    __doc__ = "Mean Square Error between goal and calculated output." + \
              Metric.get_features.__doc__

    def __init__(self, t_start=None, dt=None, **kwds):
        """
        Initialize the metric.

        Parameters
        ----------
        t_start: beggining of time window [ms] (optional)
        dt: time step [ms] (necessary with t_start)
        """
        self.t_start = t_start
        self.dt = dt


    def get_features(self, traces, output, n_traces):
        mselist = []
        output = atleast_2d(output)

        if self.t_start is not None:
            if not isinstance(self.dt , Quantity):
                raise TypeError("To specify time window you need to also "
                                "specify dt as Quantity")
            t_start = int(self.t_start/self.dt)
            output = output[:, t_start:-1]
            traces = traces[:, t_start:-1]

        for i in arange(n_traces):
            temp_out = output[i]
            temp_traces = traces[i::n_traces]

            for trace in temp_traces:
                mse = sum(square(temp_out - trace))
                mselist.append(mse)

        return mselist


    def get_errors(self, features, n_traces):
        feat_arr = reshape(array(features), (n_traces,
                           int(len(features)/n_traces)))
        errors = feat_arr.mean(axis=0)
        return errors


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
    def __init__(self, dt, delta):
        """
        Initialize the metric with time window delta and time step dt output

        Parameters
        ----------
        dt: time step [ms]
        delta: time window [ms]
        """
        if delta is None:
            raise AssertionError("delta (time window for gamma factor), "
                                 "has to be set to ms")
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

        return gamma_factors


    def get_errors(self, features, n_traces):
        feat_arr = reshape(array(features), (n_traces,
                           int(len(features)/n_traces)))
        errors = feat_arr.mean(axis=0)
        return errors


class FeatureMetric(Metric):
    def __init__(self, traces_times, feat_list, combine=None):
        self.traces_times = traces_times
        self.feat_list = feat_list

        if combine is None:
            def combine(x, y):
                return x - y
        self.combine = combine

    def check_values(self, feat_list):
        """Removes all the None values and checks for array features"""
        for r in feat_list:
            for k, v in r.items():
                if v is None:
                    r[k] = array([99])
                    raise Warning('None for key:{}'.format(k))
                if (len(r[k])) > 1:
                    raise ValueError("you can only use features that return "
                                     "one value")

    def feat_to_err(self, d1, d2):
        d = {}
        err = 0
        for key in d1.keys():
            x = d1[key]
            y = d2[key]
            d[key] = self.combine(x, y)

        for k, v in d.items():
            err += sum(v)

        return err

    def get_features(self, traces, output, n_traces):
        n_times = shape(self.traces_times)[0]

        if (n_times != (n_traces)):
            if (n_times == 1):
                self.traces_times = list(repeat(self.traces_times[0], n_traces))
                print(self.traces_times)
            else:
                raise ValueError("Specify the traces_times variable of appropiate "
                                 "size (same as number of traces or 1).")

        unit = output.get_best_unit()
        output = output/unit
        traces = traces/unit
        self.out_feat = calc_eFEL(output, self.traces_times, self.feat_list)
        self.check_values(self.out_feat)

        sl = int(shape(traces)[0]/n_traces)
        features = []
        temp_traces = split(traces, sl)

        for ii in arange(sl):
            temp_trace = temp_traces[ii]
            temp_feat = calc_eFEL(temp_trace, self.traces_times,
                                  self.feat_list)
            self.check_values(temp_feat)
            features.append(temp_feat)

        return features


    def get_errors(self, features, n_traces):
        errors = []
        for feat in features:
            temp_errors = []
            for i, F in enumerate(feat):
                temp_err = self.feat_to_err(F, self.out_feat[i])
                temp_errors.append(temp_err)

            error = sum(abs(temp_errors))
            errors.append(error)

        return errors
