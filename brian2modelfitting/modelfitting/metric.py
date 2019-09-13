import abc
import efel
from itertools import repeat
from brian2 import Hz, second, Quantity, ms, us
from brian2.units.fundamentalunits import check_units, in_unit
from numpy import (array, sum, square, reshape, abs, amin, digitize,
                   rint, arange, atleast_2d, NaN, float64, split, shape,)


def firing_rate(spikes):
    """Raturns rate of the spike train"""
    if len(spikes) < 2:
        return NaN
    return (len(spikes) - 1) / (spikes[-1] - spikes[0])


def get_gamma_factor(model, data, delta, time, dt):
    """
    Calculate gamma factor between model and tagret spike trains,
    with precision delta.

    Parameters
    ----------
    model: list/array
        model trace, goal performance
    data: list/array
        data trace
    delta: Quantity
        time window
    dt: Quantity
        time step

    Returns
    -------
        gamma factor: float
    """
    model = array(model)
    data = array(data)

    model = array(rint(model / dt), dtype=int)
    data = array(rint(data / dt), dtype=int)
    delta_diff = int(rint(delta / dt))

    model_length = len(model)
    data_length = len(data)
    # data_rate = firing_rate(data) * Hz
    data_rate = data_length / time
    model_rate = model_length / time


    if (model_length > 1):
        bins = .5 * (model[1:] + model[:-1])
        indices = digitize(data, bins)
        diff = abs(data - model[indices])
        matched_spikes = (diff <= delta_diff)
        coincidences = sum(matched_spikes)
    elif model_length == 0:
        return 0
    else:
        indices = [amin(abs(model - data[i])) <= delta_diff for i in arange(data_length)]
        coincidences = sum(indices)

    # Normalization of the coincidences count
    NCoincAvg = 2 * delta * data_length * data_rate
    norm = .5*(1 - 2 * data_rate * delta)
    gamma = (coincidences - NCoincAvg)/(norm*(model_length + data_length))

    corrected_gamma_factor = 2*abs((data_rate - model_rate)/data_rate - gamma)

    return corrected_gamma_factor


def calc_eFEL(traces, inp_times, feat_list, dt):
    out_traces = []
    for i, trace in enumerate(traces):
        time = arange(0, len(trace))*dt/ms
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
    def get_features(self, traces, output, n_traces, dt):
        """
        Function calculates features / errors for each of the traces and stores
        it in an attribute metric.features.

        The output of the function has to take shape of (n_samples, n_traces)
        or (n_traces, n_samples).

        Parameters
        ----------
        traces: 2D array
            traces to be evaluated
        output: array
            goal traces
        n_traces: int
            number of input traces
        dt: Quantity
            time step

        """
        pass

    @abc.abstractmethod
    def get_errors(self, features, n_traces):
        """
        Function weights features/multiple errors into one final error per each
        set of parameters and inputs stored metric.errors.

        The output of the function has to take shape of (n_samples,).


        Parameters
        ----------
        features: 2D array
            features set for each simulated trace
        n_traces: int
            number of input traces
        """
        pass

    def calc(self, traces, output, n_traces, dt):
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
        n_traces: int
            number of input traces
        dt: Quantity
            time step

        Returns
        -------
        errors: array
            weigheted/mean error for each set of parameters

        """
        features = self.get_features(traces, output, n_traces, dt)
        errors = self.get_errors(features)

        return errors


class TraceMetric(Metric):
    """
    Input traces have to be shaped into 2D array.
    """
    pass


class SpikeMetric(Metric):
    """
    Output spikes contain a list of arrays (possibly of different lengths)
    in order to allow different lengths of spike trains.
    Example: [array([1, 2, 3]), array([1, 2])]
    """
    pass


class MSEMetric(TraceMetric):
    __doc__ = "Mean Square Error between goal and calculated output." + \
              Metric.get_features.__doc__

    @check_units(t_start=second)
    def __init__(self, t_start=None, **kwds):
        """
        Initialize the metric.

        Parameters
        ----------
        t_start: beggining of time window (Quantity) (optional)
        """
        self.t_start = t_start

    def get_features(self, traces, output, n_traces, dt):
        mselist = []
        output = atleast_2d(output)

        if self.t_start is not None:
            if not isinstance(dt, Quantity):
                raise TypeError("To specify time window you need to also "
                                "specify dt as Quantity")
            t_start = int(self.t_start/dt)
            output = output[:, t_start:-1]
            traces = traces[:, t_start:-1]

        for i in arange(n_traces):
            temp_out = output[i]
            temp_traces = traces[i::n_traces]

            for trace in temp_traces:
                mse = sum(square(temp_out - trace))
                mselist.append(mse)

        feat_arr = reshape(array(mselist), (n_traces,
                           int(len(mselist)/n_traces)))

        return feat_arr

    def get_errors(self, features):
        errors = features.mean(axis=0)
        return errors


class FeatureMetric(TraceMetric):
    def __init__(self, traces_times, feat_list, weights=None, combine=None):
        self.traces_times = traces_times
        self.feat_list = feat_list

        if combine is None:
            def combine(x, y):
                return x - y
        self.combine = combine

        if weights is None:
            weights = {}
            for key in feat_list:
                weights[key] = 1
        if type(weights) is not dict:
            raise TypeError("Weights has to be a dictionary!")

        self.weights = weights

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
            d[key] = self.combine(x, y) * self.weights[key]
            
        for k, v in d.items():
            err += sum(v)

        return err

    def get_features(self, traces, output, n_traces, dt):
        if isinstance(self.traces_times[0][0], Quantity):
            for n, trace in enumerate(self.traces_times):
                t_start, t_end = trace[0], trace[1]
                t_start = t_start / ms
                t_end = t_end / ms
                self.traces_times[n] = [t_start, t_end]

        n_times = shape(self.traces_times)[0]

        if (n_times != (n_traces)):
            if (n_times == 1):
                self.traces_times = list(repeat(self.traces_times[0], n_traces))
            else:
                raise ValueError("Specify the traces_times variable of appropiate "
                                 "size (same as number of traces or 1).")

        unit = output.get_best_unit()
        output = output/unit
        traces = traces/unit
        self.out_feat = calc_eFEL(output, self.traces_times, self.feat_list, dt)
        self.check_values(self.out_feat)

        sl = int(shape(traces)[0]/n_traces)
        features = []
        temp_traces = split(traces, sl)

        for ii in arange(sl):
            temp_trace = temp_traces[ii]
            temp_feat = calc_eFEL(temp_trace, self.traces_times,
                                  self.feat_list, dt)
            self.check_values(temp_feat)
            features.append(temp_feat)

        return features

    def get_errors(self, features):
        errors = []
        for feat in features:
            temp_errors = []
            for i, F in enumerate(feat):
                temp_err = self.feat_to_err(F, self.out_feat[i])
                temp_errors.append(temp_err)

            error = sum(abs(temp_errors))
            errors.append(error)

        return errors


class GammaFactor(SpikeMetric):
    __doc__ = """
    Calculate gamma factors between goal and calculated spike trains, with
    precision delta.

    Reference:
    R. Jolivet et al., 'A benchmark test for a quantitative assessment of
    simple neuron models',
    Journal of Neuroscience Methods 169, no. 2 (2008): 417-424.
    """ + Metric.get_features.__doc__

    @check_units(delta=second, time=second)
    def __init__(self, delta, time):
        """
        Initialize the metric with time window delta and time step dt output

        Parameters
        ----------
        delta: time window (Quantity)
        time: total lenght of experiment (Quantity)
        """
        self.delta = delta
        self.time = time

    def get_features(self, traces, output, n_traces, dt):
        gamma_factors = []
        if type(output[0]) == float64:
            output = atleast_2d(output)

        for i in arange(n_traces):
            temp_out = output[i]
            temp_traces = traces[i::n_traces]

            for trace in temp_traces:
                gf = get_gamma_factor(trace, temp_out, self.delta, self.time, dt)
                # gamma_factors.append(abs(1 - gf))
                gamma_factors.append(gf)

        feat_arr = reshape(array(gamma_factors), (n_traces,
                           int(len(gamma_factors)/n_traces)))

        return feat_arr

    def get_errors(self, features):
        errors = features.mean(axis=0)
        return errors
