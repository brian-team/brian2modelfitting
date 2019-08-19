Metric
======

Metric input to specifies the fitness function measuring the performance of the simulation.
This function gets applied on each simulated trace. We have implemented few metrics within
modelfitting.

 .. contents::
   Provided metrics:
     :local:
     :depth: 1


Mean Square Error
-----------------

To be used with ``TraceFitter``. Calculated according to well known formula:

.. math:: MSE ={\frac {1}{n}}\sum _{i=1}^{n}(Y_{i}-{\hat {Y_{i}}})^{2}


To be called in a following way:

.. code:: python

  metric = MSEMetric()


In ``OnlineTraceFitter`` mean square error gets calculated in online manner,
with no need of specifying a metric object.


GammaFactor
-----------

To be used with ``SpikeFitter``. Calculated according to:


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



Custom Metric
-------------

User is not limited to the provided in the module metrics. Modularity applies
here as well, with provided abstract ``class Metric`` prepared for different
custom made metrics.

New metric will need to be inherited from ``class Metric`` and specify following
functions:
 - ``get_features()``
    calculates features / errors for each of the traces and stores
    it in an attribute metric.features
 - ``get_errors()``
    weights features/multiple errors into one final error per each
    set of parameters and inputs stored metric.errors.
 - ``calc()``
    performs the error calculation across simulation for all parameters
    of each round
