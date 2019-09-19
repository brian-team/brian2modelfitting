Metric
======

Metric input to specifies the fitness function measuring the performance of the simulation.
This function gets applied on each simulated trace. We have implemented few metrics within
modelfitting.

.. contents::
     :local:
     :depth: 1


Mean Square Error
-----------------

:py:class:`~brian2modelfitting.modelfitting.metric.MSEMetric` is implemented to use with :py:class:`~brian2modelfitting.modelfitting.modelfitting.TraceFitter`. Calculated according to well known formula:

.. math:: MSE ={\frac {1}{n}}\sum _{i=1}^{n}(Y_{i}-{\hat {Y_{i}}})^{2}


To be called in a following way:

.. code:: python

  metric = MSEMetric()

Additionally, :py:class:`~brian2modelfitting.modelfitting.metric.MSEMetric` accepts an optional input argument
start time ``t_start`` (as a :py:class:`~brian2.units.fundamentalunits.Quantity`). The start time allows the user to
ignore an initial period that will not be included in the error calculation.

.. code:: python

  metric = MSEMetric(t_start=5*ms)


In :py:class:`~brian2modelfitting.modelfitting.modelfitting.OnlineTraceFitter` mean square error gets calculated in online manner,
with no need of specifying a metric object.


GammaFactor
-----------

:py:class:`~brian2modelfitting.modelfitting.metric.GammaFactor` is implemented to use with :py:class:`~brian2modelfitting.modelfitting.modelfitting.SpikeFitter`. Calculated according to:


.. math:: \Gamma = \left (\frac{2}{1-2\delta r_{exp}}\right) \left(\frac{N_{coinc} - 2\delta N_{exp}r_{exp}}{N_{exp} + N_{model}}\right)

:math:`N_{coinc}` - number of coincidences

:math:`N_{exp}` and :math:`N_{model}`- number of spikes in experimental and model spike trains

:math:`r_{exp}` - average firing rate in experimental train

:math:`2 \delta N_{exp}r_{exp}` - expected number of coincidences with a Poission process

For more details on the gamma factor, see
Jolivet et al. 2008, “A benchmark test for a quantitative assessment of simple neuron models”, J. Neurosci. Methods.
(https://www.ncbi.nlm.nih.gov/pubmed/18160135)

Upon initialization user has to specify the delta as a :py:class:`~brian2.units.fundamentalunits.Quantity`:

.. code:: python

  metric = GammaFactor(delta=10*ms)


FeatureMetric
-------------
:py:class:`~brian2modelfitting.modelfitting.metric.FeatureMetric` is implemented to use with :py:class:`~brian2modelfitting.modelfitting.modelfitting.TraceFitter`.
Metric demonstrates a use of feature based metric in the toolbox. Features used for optimization get calculated with use of

The Electrophys Feature Extract Library (eFEL) library, for which the documentation is available under following link: https://efel.readthedocs.io/en/latest/

To get all of the eFEL features you can run the following code:

.. code:: python

  import efel
  efel.api.getFeatureNames()


.. note::

  User is only allowed to use features that return array of no more than one value.


To define the :py:class:`~brian2modelfitting.modelfitting.metric.FeatureMetric`, user has to define following input parameters:

- ``traces_times`` - list of times indicating start and end of input current, has to be specified for each of input traces, each value has to be a :py:class:`~brian2.units.fundamentalunits.Quantity`
- ``feat_list`` - list of strings with names of features to be used
- ``combine`` - function to be used to compare features between output and simulated traces, (for `combine=None`, subtracts the features)

Example code usage:

.. code:: python

  traces_times = [[50*ms, 100*ms], [50*ms, 100*ms], [50*ms, 100*ms], [50, 100*ms]]
  feat_list = ['voltage_base', 'time_to_first_spike', 'Spikecount']
  metric = FeatureMetric(traces_times, feat_list, combine=None)

.. note::

  If times of stimulation are same for all of the traces, user can specify a single list that will be replicated for
  ``eFEL`` library: ``traces_times = [[50*ms, 100*ms]]``.




Custom Metric
-------------

User is not limited to the provided in the module metrics. Modularity applies
here as well, with one of the two provided abstract classes :py:class:`~brian2modelfitting.modelfitting.metric.TraceMetric`
and :py:class:`~brian2modelfitting.modelfitting.metric.SpikeMetric` prepared for different custom made metrics.

New metric will need to have specify following functions:

 - :py:func:`~brian2modelfitting.modelfitting.metric.Metric.get_features()`
    calculates features / errors for each of the simulations. The representation of the model results and the target
    data depend on whether traces or spikes are fitted, see below.

 - :py:func:`~brian2modelfitting.modelfitting.metric.Metric.get_errors()`
    weights features/multiple errors into one final error per each set of parameters and inputs. The features are
    received as a 2-dimensional :py:class:`~numpy.ndarray` of shape ``(n_samples, n_traces)`` The output has to be an
    array of length ``n_samples``, i.e. one value for each parameter set.

 - :py:func:`~brian2modelfitting.modelfitting.metric.Metric.calc()`
    performs the error calculation across simulation for all parameters of each round. Specified in the abstract class, can be reused.


TraceMetric
~~~~~~~~~~~
To create a new metric for :py:class:`~brian2modelfitting.modelfitting.modelfitting.TraceFitter`, you have to inherit
from :py:class:`~brian2modelfitting.modelfitting.metric.TraceMetric` and overwrite the
:py:func:`.TraceMetric.get_features` and/or :py:func:`~.TraceMetric.get_errors` method. The model traces for the
:py:func:`~.TraceMetric.get_features` function are provided as a 3-dimensional :py:class:`~numpy.ndarray` of shape
``(n_samples, n_traces, time steps)``, where ``n_samples`` are the number of
different parameter sets that have been evaluated, and ``n_traces`` the number of different stimuli that have been
evaluated for each parameter set. The output of the function has to take the shape of ``(n_samples, n_traces)``. This
array is the input to the :py:func:`~.TraceMetric.get_errors` method (see above).

.. code:: python

  class NewTraceMetric(TraceMetric):
    def get_features(self, model_traces, data_traces, dt):
      ...

    def get_errors(self, features):
      ...

SpikeMetric
~~~~~~~~~~~
To create a new metric for :py:class:`~brian2modelfitting.modelfitting.modelfitting.SpikeFitter`, you have to inherit
from :py:class:`~brian2modelfitting.modelfitting.metric.SpikeMetric`. Inputs of the metric in
:py:func:`~.SpikeMetric.get_features` are a nested list structure for the spikes generated by the model: a list where
each element contains the results for a single parameter set. Each of these results is a list for each of the input
traces, where the elements of this list are numpy arrays of spike times (without units, i.e. in seconds). For example,
if two parameters sets and 3 different input stimuli were tested, this structure could look like this::

    [
        [array([0.01, 0.5]), array([]), array([])],
        [array([0.02]), array([]), array([])]
    ]

This means that the both parameter sets only generate spikes for the first input stimulus, but the first parameter sets
generates two while the second generates only a single one.

The target spikes are represented in the same way as a list of spike times for each input stimulus. The results of the
function have to be returned as in :py:class:`~.TraceMetric`, i.e. as a 2-d array of shape ``(n_samples, n_traces)``.