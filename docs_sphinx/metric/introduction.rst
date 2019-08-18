Metric
------

Metric input to specifies the fitness function measuring the performance of the simulation.
This function gets applied on each simulated trace. We have implemented few metrics within
modelfitting. Modularity applies here as well, with provided abstract `class Metric`
prepared for different custom made metrics.

Provided metrics:
**1. Mean Square Error**
**2. GammaFactor - for `fit_spikes()`.**
