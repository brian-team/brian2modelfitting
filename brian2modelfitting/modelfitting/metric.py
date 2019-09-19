import warnings
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

    @check_units(t_start=second)
    def __init__(self, t_start=0*second, **kwds):
        """
        Initialize the metric.

        Parameters
        ----------
        t_start: Quantity, optional
            Start of time window considered for calculating the fit error.
        """
        self.t_start = t_start

    @abc.abstractmethod
    def get_features(self, model_results, target_results, dt):
        """
        Function calculates features / errors for each of the input traces.

        The output of the function has to take shape of (n_samples, n_traces).
        """
        pass

    @abc.abstractmethod
    def get_errors(self, features, n_traces):
        """
        Function weights features/multiple errors into one final error per each
        set of parameters.

        The output of the function has to take shape of (n_samples,).


        Parameters
        ----------
        features: 2D array
            features set for each simulated trace
        n_traces: int
            number of input traces
        """
        pass


class TraceMetric(Metric):
    """
    Input traces have to be shaped into 2D array.
    """

    def calc(self, model_traces, data_traces, dt):
        """
        Perform the error calculation across all parameters,
        calculate error between each output trace and corresponding
        simulation.

        Parameters
        ----------
        model_traces: ndarray
            Traces that should be evaluated and compared to the target data.
            Provided as an `.ndarray` of shape (samples, traces, time steps),
            where "samples" are the different parameter values that have been
            evaluated, and "traces" are the responses of the model to the
            different input stimuli.
        data_traces: array
            The target traces to which the model should be compared. An
            `ndarray` of shape (traces, time steps).
        dt: Quantity
            The length of a single time step.

        Returns
        -------
        errors: ndarray
            Total error for each set of parameters.

        """
        start_steps = int(round(self.t_start/dt))
        features = self.get_features(model_traces[:, :, start_steps:],
                                     data_traces[:, start_steps:],
                                     dt)
        errors = self.get_errors(features)

        return errors


class SpikeMetric(Metric):
    """
    Output spikes contain a list of arrays (possibly of different lengths)
    in order to allow different lengths of spike trains.
    Example: [array([1, 2, 3]), array([1, 2])]
    """

    def calc(self, model_spikes, data_spikes, dt):
        """
        Perform the error calculation across all parameters,
        calculate error between each output trace and corresponding
        simulation.

        Parameters
        ----------
        model_spikes: list of list of arrays
            A nested list structure for the spikes generated by the model: a
            list where each element contains the results for a single parameter
            set. Each of these results is a list for each of the input traces,
            where the elements of this list are numpy arrays of spike times
            (without units, i.e. in seconds).
        data_spikes: list of arrays
            The target spikes for the fitting, arepresented in the same way as
            ``model_spikes``, i.e. as a list of spike times for each input
            stimulus.
        dt: Quantity
            The length of a single time step.

        Returns
        -------
        errors: ndarray
            Total error for each set of parameters.

        """
        if self.t_start > 0*second:
            relevant_data_spikes = []
            for one_stim in data_spikes:
                relevant_data_spikes.append(one_stim[one_stim>float(self.t_start)])
            relevant_model_spikes = []
            for one_sample in model_spikes:
                sample_spikes = []
                for one_stim in one_sample:
                    sample_spikes.append(one_stim[one_stim>float(self.t_start)])
                relevant_model_spikes.append(sample_spikes)
            model_spikes = relevant_model_spikes
            data_spikes = relevant_data_spikes
        features = self.get_features(model_spikes, data_spikes, dt)
        errors = self.get_errors(features)

        return errors


class MSEMetric(TraceMetric):
    __doc__ = "Mean Square Error between goal and calculated output." + \
              Metric.get_features.__doc__

    def get_features(self, model_traces, data_traces, dt):
        return sum((model_traces - data_traces)**2, axis=2)

    def get_errors(self, features):
        return features.mean(axis=1)


class FeatureMetric(TraceMetric):
    def __init__(self, traces_times, feat_list, weights=None, combine=None,
                 t_start=0*second):
        super(FeatureMetric, self).__init__(t_start=t_start)
        self.traces_times = traces_times
        if isinstance(self.traces_times[0][0], Quantity):
            for n, trace in enumerate(self.traces_times):
                t_start, t_end = trace[0], trace[1]
                t_start = t_start / ms
                t_end = t_end / ms
                self.traces_times[n] = [t_start, t_end]
        n_times = shape(self.traces_times)[0]

        self.feat_list = feat_list

        if combine is None:
            def combine(x, y):
                return abs(x - y)
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
                    warnings.warn('None for key:{}'.format(k))
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

    def get_features(self, traces, output, dt):
        n_samples, n_traces, _ = traces.shape
        if len(self.traces_times) != n_traces:
            if len(self.traces_times) == 1:
                self.traces_times = list(repeat(self.traces_times[0], n_traces))
            else:
                raise ValueError("Specify the traces_times variable of appropiate "
                                 "size (same as number of traces or 1).")

        self.out_feat = calc_eFEL(output, self.traces_times, self.feat_list, dt)
        self.check_values(self.out_feat)

        features = []
        for one_sample in traces:
            temp_feat = calc_eFEL(one_sample, self.traces_times,
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

    @check_units(delta=second, time=second, t_start=0*second)
    def __init__(self, delta, time, t_start=0*second):
        """
        Initialize the metric with time window delta and time step dt output

        Parameters
        ----------
        delta: time window (Quantity)
        time: total lenght of experiment (Quantity)
        """
        super(GammaFactor, self).__init__(t_start=t_start)
        self.delta = delta
        self.time = time

    def get_features(self, traces, output, dt):
        all_gf = []
        for one_sample in traces:
            gf_for_sample = []
            for model_response, target_response in zip(one_sample, output):
                gf = get_gamma_factor(model_response, target_response,
                                      self.delta, self.time, dt)
                gf_for_sample.append(gf)
            all_gf.append(gf_for_sample)
        return array(all_gf)

    def get_errors(self, features):
        errors = features.mean(axis=1)
        return errors
