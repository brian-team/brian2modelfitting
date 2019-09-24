import warnings
import abc
from collections import defaultdict

import efel
from itertools import repeat
from brian2 import Hz, second, Quantity, ms, us
from brian2.units.fundamentalunits import check_units, in_unit
from numpy import (array, sum, square, reshape, abs, amin, digitize,
                   rint, arange, atleast_2d, NaN, float64, split, shape,)


def firing_rate(spikes):
    """Returns rate of the spike train"""
    if len(spikes) < 2:
        return NaN
    return (len(spikes) - 1) / (spikes[-1] - spikes[0])


def get_gamma_factor(model, data, delta, time, dt, rate_correction=True):
    r"""
    Calculate gamma factor between model and target spike trains,
    with precision delta.

    Parameters
    ----------
    model: `list` or `~numpy.ndarray`
        model trace
    data: `list` or `~numpy.ndarray`
        data trace
    delta: `~brian2.units.fundamentalunits.Quantity`
        time window
    dt: `~brian2.units.fundamentalunits.Quantity`
        time step
    time: `~brian2.units.fundamentalunits.Quantity`
        total time of the simulation
    rate_correction: bool
        Whether to include an error term that penalizes differences in firing
        rate, following `Clopath et al., Neurocomputing (2007)
        <https://doi.org/10.1016/j.neucom.2006.10.047>`_.

    Returns
    -------
    float
        An error based on the Gamma factor. If ``rate_correction`` is used,
        then the returned error is :math:`2\frac{\lvert r_\mathrm{data} - r_\mathrm{model}\rvert}{r_\mathrm{data}} - \Gamma`
        (with :math:`r_\mathrm{data}` and :math:`r_\mathrm{model}` being the
        firing rates in the data/model, and :math:`\Gamma` the coincidence
        factor). Without ``rate_correction``, the error is
        :math:`1 - \Gamma`. Note that the coincidence factor :math:`\Gamma`
        has a maximum value of 1 (when the two spike trains are exactly
        identical) and a value of 0 if there are only as many coincidences
        as expected from two homogeneous Poisson processes of the same rate.
        It can also take negative values if there are fewer coincidences
        than expected by chance.
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

    if model_length > 1:
        bins = .5 * (model[1:] + model[:-1])
        indices = digitize(data, bins)
        diff = abs(data - model[indices])
        matched_spikes = (diff <= delta_diff)
        coincidences = sum(matched_spikes)
    elif model_length == 0:
        coincidences = 0
    else:
        indices = [amin(abs(model - data[i])) <= delta_diff for i in arange(data_length)]
        coincidences = sum(indices)

    # Normalization of the coincidences count
    NCoincAvg = 2 * delta * data_length * data_rate
    norm = .5*(1 - 2 * data_rate * delta)
    gamma = (coincidences - NCoincAvg)/(norm*(model_length + data_length))

    if rate_correction:
        rate_term = 2*abs((data_rate - model_rate)/data_rate)
    else:
        rate_term = 1

    return rate_term - gamma

def calc_eFEL(traces, inp_times, feat_list, dt):
    out_traces = []
    for i, trace in enumerate(traces):
        time = arange(0, len(trace))*dt/ms
        temp_trace = {}
        temp_trace['T'] = time
        temp_trace['V'] = array(trace, copy=False)
        temp_trace['stim_start'] = [inp_times[i][0]]
        temp_trace['stim_end'] = [inp_times[i][1]]
        out_traces.append(temp_trace)

    results = efel.getFeatureValues(out_traces, feat_list)
    return results


class Metric(metaclass=abc.ABCMeta):
    """
    Metric abstract class to define functions required for a custom metric
    To be used with modelfitting Fitters.
    """

    @check_units(t_start=second)
    def __init__(self, t_start=0*second, **kwds):
        """
        Initialize the metric.

        Parameters
        ----------
        t_start: `~brian2.units.fundamentalunits.Quantity`, optional
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
    def get_errors(self, features):
        """
        Function weights features/multiple errors into one final error per each
        set of parameters.

        The output of the function has to take shape of (n_samples,).


        Parameters
        ----------
        features: `~numpy.ndarray`
            2D array of shape ``(n_samples, n_traces)`` with the features/errors
            for each simulated trace

        Returns
        -------
        `~numpy.ndarray`
            Errors for each parameter set, i.e. of shape ``(n_samples, )``
        """
        pass

    @abc.abstractmethod
    def calc(self, model_results, data_results, dt):
        """
        Perform the error calculation across all parameter sets by comparing
        the simulated to the experimental data.

        Parameters
        ----------
        model_results
            Results generated by the model. The type and shape of this data
            depends on the fitting problem. See `.TraceMetric.calc` and
            `.SpikeMetric.calc`.
        data_results
            The experimental data that the model is fit against. See
            `.TraceMetric.calc` and `.SpikeMetric.calc` for the type/shape
            of the data.
        dt: `~brian2.units.fundamentalunits.Quantity`
            The length of a single time step.
        Returns
        -------
        `~numpy.ndarray`
            Total error for each set of parameters, i.e. an array of shape
            ``(n_samples, )``.
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
        model_traces: `~numpy.ndarray`
            Traces that should be evaluated and compared to the target data.
            Provided as an `~numpy.ndarray` of shape
            ``(n_samples, n_traces, time steps)`` where ``n_samples`` is the
            number of parameter sets that have been evaluated, and ``n_traces``
            is the number of stimuli.
        data_traces: `~numpy.ndarray`
            The target traces to which the model should be compared. An
            `~numpy.ndarray` of shape ``(n_traces, time steps)``.
        dt: `~brian2.units.fundamentalunits.Quantity`
            The length of a single time step.

        Returns
        -------
        `~numpy.ndarray`
            Total error for each set of parameters, i.e. an array of shape
            ``(n_samples, )``.
        """
        start_steps = int(round(self.t_start/dt))
        features = self.get_features(model_traces[:, :, start_steps:],
                                     data_traces[:, start_steps:],
                                     dt)
        errors = self.get_errors(features)

        return errors

    @abc.abstractmethod
    def get_features(self, model_traces, data_traces, dt):
        """
        Calculate the features/errors for each simulated trace, by comparing
        it to the corresponding data trace.

        Parameters
        ----------
        model_traces: `~numpy.ndarray`
            Traces that should be evaluated and compared to the target data.
            Provided as an `~numpy.ndarray` of shape
            ``(n_samples, n_traces, time steps)``,
            where ``n_samples`` are the number of different parameter sets that
            have been evaluated, and ``n_traces`` are the number of input
            stimuli.
        data_traces: `~numpy.ndarray`
            The target traces to which the model should be compared. An
            `~numpy.ndarray` of shape ``(n_traces, time steps)``.
        dt: `~brian2.units.fundamentalunits.Quantity`
            The length of a single time step.

        Returns
        -------
        `~numpy.ndarray`
            An `~numpy.ndarray` of shape ``(n_samples, n_traces)``
            returning the error/feature value for each simulated trace.
        """
        pass


class SpikeMetric(Metric):
    """
    A metric for comparing the spike trains.
    """

    def calc(self, model_spikes, data_spikes, dt):
        """
        Perform the error calculation across all parameters,
        calculate error between each output trace and corresponding
        simulation.

        Parameters
        ----------
        model_spikes: list of list of `~numpy.ndarray`
            A nested list structure for the spikes generated by the model: a
            list where each element contains the results for a single parameter
            set. Each of these results is a list for each of the input traces,
            where the elements of this list are numpy arrays of spike times
            (without units, i.e. in seconds).
        data_spikes: list of `~numpy.ndarray`
            The target spikes for the fitting, represented in the same way as
            ``model_spikes``, i.e. as a list of spike times for each input
            stimulus.
        dt: `~brian2.units.fundamentalunits.Quantity`
            The length of a single time step.

        Returns
        -------
        `~numpy.ndarray`
            Total error for each set of parameters, i.e. an array of shape
            ``(n_samples, )``

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

    @abc.abstractmethod
    def get_features(self, model_spikes, data_spikes, dt):
        """
        Calculate the features/errors for each simulated spike train by
        comparing it to the corresponding data spike train.

        Parameters
        ----------
        model_spikes: list of list of `~numpy.ndarray`
            A nested list structure for the spikes generated by the model: a
            list where each element contains the results for a single parameter
            set. Each of these results is a list for each of the input traces,
            where the elements of this list are numpy arrays of spike times
            (without units, i.e. in seconds).
        data_spikes: list of `~numpy.ndarray`
            The target spikes for the fitting, represented in the same way as
            ``model_spikes``, i.e. as a list of spike times for each input
            stimulus.
        dt: `~brian2.units.fundamentalunits.Quantity`
            The length of a single time step.

        Returns
        -------
        `~numpy.ndarray`
            An `~numpy.ndarray` of shape ``(n_samples, n_traces)``
            returning the error/feature value for each simulated trace.
        """


class MSEMetric(TraceMetric):
    """
    Mean Square Error between goal and calculated output.
    """

    def get_features(self, model_traces, data_traces, dt):
        return sum((model_traces - data_traces)**2, axis=2)

    def get_errors(self, features):
        return features.mean(axis=1)


class FeatureMetric(TraceMetric):
    def __init__(self, stim_times, feat_list, weights=None, combine=None,
                 t_start=0*second):
        super(FeatureMetric, self).__init__(t_start=t_start)
        self.stim_times = stim_times
        if isinstance(self.stim_times[0][0], Quantity):
            for n, trace in enumerate(self.stim_times):
                t_start, t_end = trace[0], trace[1]
                t_start = t_start / ms
                t_end = t_end / ms
                self.stim_times[n] = [t_start, t_end]

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
            d[key] = self.combine(x, y)

        return d

    def get_features(self, traces, output, dt):
        n_samples, n_traces, _ = traces.shape
        if len(self.stim_times) != n_traces:
            if len(self.stim_times) == 1:
                self.stim_times = list(repeat(self.stim_times[0], n_traces))
            else:
                raise ValueError("Specify the stim_times variable of appropiate "
                                 "size (same as number of traces or 1).")

        out_feat = calc_eFEL(output, self.stim_times, self.feat_list, dt)
        self.check_values(out_feat)

        features = []
        for one_sample in traces:
            sample_feat = calc_eFEL(one_sample, self.stim_times,
                                    self.feat_list, dt)
            self.check_values(sample_feat)
            sample_features = []
            for one_trace_feat, one_out in zip(sample_feat, out_feat):
                sample_features.append(self.feat_to_err(one_trace_feat,
                                                        one_out))
            # Convert the list of dictionaries to a dictionary of lists
            sample_features_dict = {}
            for feature_dict in sample_features:
                for key, value in feature_dict.items():
                    if key not in sample_features_dict:
                        sample_features_dict[key] = []
                    if len(value) != 1:
                        raise TypeError('Feature "{}" returned more than a '
                                        'single value, such features are not '
                                        'supported yet.'.format(key))
                    sample_features_dict[key].append(value[0])

            # Convert lists into array
            for key, l in sample_features_dict.items():
                sample_features_dict[key] = array(l)
            features.append(sample_features_dict)

        return features

    def get_errors(self, features):
        errors = []
        for one_sample in features:
            sample_error = 0
            for feature, values in one_sample.items():
                # sum over the traces
                total = sum(values) * self.weights[feature]
                sample_error += total
            errors.append(sample_error)

        return errors


class GammaFactor(SpikeMetric):
    """
    Calculate gamma factors between goal and calculated spike trains, with
    precision delta.

    References:

    * `R. Jolivet et al. “A Benchmark Test for a Quantitative Assessment of
      Simple Neuron Models.” Journal of Neuroscience Methods, 169, no. 2 (2008):
      417–24. <https://doi.org/10.1016/j.jneumeth.2007.11.006>`_
    * `C. Clopath et al. “Predicting Neuronal Activity with Simple Models of the
      Threshold Type: Adaptive Exponential Integrate-and-Fire Model with
      Two Compartments.” Neurocomputing, 70, no. 10 (2007): 1668–73.
      <https://doi.org/10.1016/j.neucom.2006.10.047>`_
    """

    @check_units(delta=second, time=second, t_start=0*second)
    def __init__(self, delta, time, t_start=0*second, rate_correction=True):
        """
        Initialize the metric with time window delta and time step dt output

        Parameters
        ----------
        delta: `~brian2.units.fundamentalunits.Quantity`
            time window
        time: `~brian2.units.fundamentalunits.Quantity`
            total length of experiment
        rate_correciton: bool
            Whether to include an error term that penalizes differences in firing
            rate, following `Clopath et al., Neurocomputing (2007)
            <https://doi.org/10.1016/j.neucom.2006.10.047>`_.
        """
        super(GammaFactor, self).__init__(t_start=t_start)
        self.delta = delta
        self.time = time
        self.rate_correction = rate_correction

    def get_features(self, traces, output, dt):
        all_gf = []
        for one_sample in traces:
            gf_for_sample = []
            for model_response, target_response in zip(one_sample, output):
                gf = get_gamma_factor(model_response, target_response,
                                      self.delta, self.time, dt,
                                      rate_correction=self.rate_correction)
                gf_for_sample.append(gf)
            all_gf.append(gf_for_sample)
        return array(all_gf)

    def get_errors(self, features):
        errors = features.mean(axis=1)
        return errors
