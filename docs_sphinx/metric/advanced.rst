Custom Metric
=============

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
