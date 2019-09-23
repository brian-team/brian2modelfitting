Metric
======

A *Metric* specifies the fitness function measuring the performance of the
simulation. This function gets applied on each simulated trace. A few metrics
are already implemented and included in the toolbox, but the user can also
provide their own metric.

.. contents::
     :local:
     :depth: 1


Mean Square Error
-----------------

`~brian2modelfitting.metric.MSEMetric` is provided for
use with `~brian2modelfitting.fitter.TraceFitter`.
It calculates the mean squared difference between the data and the simulated
trace according to the well known formula:

.. math:: MSE ={\frac {1}{n}}\sum _{i=1}^{n}(Y_{i}-{\hat {Y_{i}}})^{2}


It can be initialized in the following way:

.. code:: python

  metric = MSEMetric()

Additionally, `~brian2modelfitting.metric.MSEMetric`
accepts an optional input argument start time ``t_start`` (as a
`~brian2.units.fundamentalunits.Quantity`). The start time allows the
user to ignore an initial period that will not be included in the error
calculation.

.. code:: python

  metric = MSEMetric(t_start=5*ms)


In `~brian2modelfitting.fitter.OnlineTraceFitter`,
the mean square error gets calculated in online manner, with no need of
specifying a metric object.


GammaFactor
-----------
`~brian2modelfitting.metric.GammaFactor` is provided for
use with `~brian2modelfitting.fitter.SpikeFitter`
and measures the coincidence between spike times in the simulated and the target
trace. It is calculcated according to:

.. math:: \Gamma = \left (\frac{2}{1-2\Delta r_{exp}}\right) \left(\frac{N_{coinc} - 2\delta N_{exp}r_{exp}}{N_{exp} + N_{model}}\right)

:math:`N_{coinc}` - number of coincidences

:math:`N_{exp}` and :math:`N_{model}`- number of spikes in experimental and model spike trains

:math:`r_{exp}` - average firing rate in experimental train

:math:`2 \Delta N_{exp}r_{exp}` - expected number of coincidences with a Poission process

For more details on the gamma factor, see
`Jolivet et al. 2008, “A benchmark test for a quantitative assessment of simple
neuron models”, J. Neurosci. Methods.
<https://doi.org/10.1016/j.jneumeth.2007.11.006>`_

Upon initialization the user has to specify the :math:`\Delta` value, defining
the maximal tolerance for spikes to be considered coincident:

.. code:: python

  metric = GammaFactor(delta=10*ms)

.. warning::
    The ``delta`` parameter has to be smaller than the smallest inter-spike
    interval in the spike trains.

FeatureMetric
-------------
`~brian2modelfitting.metric.FeatureMetric` is provided
for use with `~brian2modelfitting.fitter.TraceFitter`.
This metric allows the user to optimize the match of certain features between
the simulated and the target trace. The features get calculated by Electrophys
Feature Extract Library (eFEL) library, for which the documentation is
available under following link: https://efel.readthedocs.io

To get a list of all the available eFEL features, you can run the following code:

.. code:: python

  import efel
  efel.api.getFeatureNames()


.. note::

  Currently, only features that are described by a single value are supported
  (e.g. the time of the first spike can be used, but not the times of all
  spikes).


To use the `~brian2modelfitting.metric.FeatureMetric`,
you have to provide the following input parameters:

- ``traces_times`` - a list of times indicating start and end of the stimulus
  for each of input traces. This information is used by several features, e.g.
  the ``voltage_base`` feature will consider the average membrane potential
  during the last 10% of time before the stimulus (see the
  `eFel documentation <https://efel.readthedocs.io/en/latest/eFeatures.html>`_
  for details).
- ``feat_list`` - list of strings with names of features to be used
- ``combine`` - function to be used to compare features between output and
  simulated traces, (for ``combine=None``, subtracts the feature values)

Example code usage:

.. code:: python

  traces_times = [(50*ms, 100*ms), (50*ms, 100*ms), (50*ms, 100*ms), (50, 100*ms)]
  feat_list = ['voltage_base', 'time_to_first_spike', 'Spikecount']
  metric = FeatureMetric(traces_times, feat_list, combine=None)

.. note::

  If times of stimulation are the same for all of the traces, then you  can
  specify a single interval instead: ``traces_times = [(50*ms, 100*ms)]``.

Custom Metric
-------------

Users are not limited to the metrics provided in the toolbox. If needed, they
can provide their own metric based on one of the abstract classes
`~brian2modelfitting.metric.TraceMetric`
and `~brian2modelfitting.metric.SpikeMetric`.

A new metric will need to specify the following functions:

 - `~brian2modelfitting.metric.Metric.get_features()`
    calculates features / errors for each of the simulations. The representation
    of the model results and the target data depend on whether traces or spikes
    are fitted, see below.

 - `~brian2modelfitting.metric.Metric.get_errors()`
    weights features/multiple errors into one final error per each set of
    parameters and inputs. The features are received as a 2-dimensional
    `~numpy.ndarray` of shape ``(n_samples, n_traces)`` The output has
    to be an array of length ``n_samples``, i.e. one value for each parameter
    set.

 - `~brian2modelfitting.metric.Metric.calc()`
    performs the error calculation across simulation for all parameters of each
    round. Already implemented in the abstract class and therefore does not
    need to be reimplemented necessarily.

TraceMetric
~~~~~~~~~~~
To create a new metric for
`~brian2modelfitting.fitter.TraceFitter`, you have
to inherit from `~brian2modelfitting.metric.TraceMetric`
and overwrite the `~.TraceMetric.get_features` and/or
`~.TraceMetric.get_errors` method. The model traces for the
`~.TraceMetric.get_features` function are provided as a 3-dimensional
`~numpy.ndarray` of shape ``(n_samples, n_traces, time steps)``,
where ``n_samples`` are the number of different parameter sets that have been
evaluated, and ``n_traces`` the number of different stimuli that have been
evaluated for each parameter set. The output of the function has to take the
shape of ``(n_samples, n_traces)``. This array is the input to the
`~.TraceMetric.get_errors` method (see above).

.. code:: python

  class NewTraceMetric(TraceMetric):
    def get_features(self, model_traces, data_traces, dt):
      ...

    def get_errors(self, features):
      ...

SpikeMetric
~~~~~~~~~~~
To create a new metric for
`~brian2modelfitting.fitter.SpikeFitter`, you have
to inherit from `~brian2modelfitting.metric.SpikeMetric`.
Inputs of the metric in `~.SpikeMetric.get_features` are a nested list
structure for the spikes generated by the model: a list where each element
contains the results for a single parameter set. Each of these results is a list
for each of the input traces, where the elements of this list are numpy arrays
of spike times (without units, i.e. in seconds). For example, if two parameters
sets and 3 different input stimuli were tested, this structure could look like
this::

    [
        [array([0.01, 0.5]), array([]), array([])],
        [array([0.02]), array([]), array([])]
    ]

This means that the both parameter sets only generate spikes for the first input
stimulus, but the first parameter sets generates two while the second generates
only a single one.

The target spikes are represented in the same way as a list of spike times for
each input stimulus. The results of the function have to be returned as in
`~.TraceMetric`, i.e. as a 2-d array of shape
``(n_samples, n_traces)``.