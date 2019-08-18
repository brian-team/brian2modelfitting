GammaFactor
===========

To be used with `SpikeFitter`. Calculated according to:


.. math:: \Gamma = \left (\frac{2}{1-2\delta r_{exp}}\right) \left(\frac{N_{coinc} - 2\delta N_{exp}r_{exp}}{N_{exp} + N_{model}}\right)

:math:`N_{coinc}$` - number of coincidences

:math:`N_{exp}` and :math:`N_{model}`- number of spikes in experimental and model spike trains

:math:`r_{exp}` - average firing rate in experimental train

:math:`2 \delta N_{exp}r_{exp}` - expected number of coincidences with a Poission process

For more details on the gamma factor, see `Jolivet et al. 2008, “A benchmark test for a quantitative assessment of simple neuron models”, J. Neurosci. Methods. <https://www.ncbi.nlm.nih.gov/pubmed/18160135>`


.. code:: python

  metric = GammaFactor(delta=10*ms, dt=0.1*ms)
