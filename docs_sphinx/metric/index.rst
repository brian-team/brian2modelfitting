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

Additionally, :py:class:`~brian2modelfitting.modelfitting.metric.MSEMetric` accepts two optional input arguments
start time ``t_start``, and time step `dt``. The following have to always be provided together and have units
(be a :py:class:`~brian2.units.fundamentalunits.Quantity`). The start time allows the user to measure the error starting
from the provided time (i.e. start of stimulation).

.. code:: python

  metric = MSEMetric(t_start=5*ms, dt=0.01*ms)


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

.. code:: python

  metric = GammaFactor(delta=10*ms, dt=0.1*ms)


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

- ``traces_times`` - list of times indicating start and end of input current, has to be specified for each of input traces
- ``feat_list`` - list of strings with names of features to be used
- ``combine`` - function to be used to compare features between output and simulated traces, (for `combine=None`, subtracts the features)

Example code usage:

.. code:: python

  traces_times = [[50, 100], [50, 100], [50, 100], [50, 100]]
  feat_list = ['voltage_base', 'time_to_first_spike', 'Spikecount']
  metric = FeatureMetric(traces_times, feat_list, combine=None)

.. note::

  If times of stimulation are same for all of the traces, user can specify a single list that will be replicated for
  ``eFEL`` library: ``traces_times = [[50, 100]]``.




Custom Metric
-------------

User is not limited to the provided in the module metrics. Modularity applies
here as well, with provided abstract class :py:class:`~brian2modelfitting.modelfitting.metric.Metric` prepared for different
custom made metrics.

New metric will need to be inherited from :py:class:`~brian2modelfitting.modelfitting.metric.Metric` and specify following
functions:

 - :py:func:`~brian2modelfitting.modelfitting.metric.Metric.get_features()`
    calculates features / errors for each of the traces and stores it in a :py:attr:`~brian2modelfitting.modelfitting.metric.Metric.metric.features` attribute

 - :py:func:`~brian2modelfitting.modelfitting.metric.Metric.get_errors()`
    weights features/multiple errors into one final error per each set of parameters and inputs stored in :py:attr:`~brian2modelfitting.modelfitting.metric.Metric.metric.errors`
 - :py:func:`~brian2modelfitting.modelfitting.metric.Metric.calc()`
    performs the error calculation across simulation for all parameters of each round
