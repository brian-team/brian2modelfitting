Mean Square Error
=================

To be used with ``TraceFitter``. Calculated according to well known formula:

.. math:: MSE ={\frac {1}{n}}\sum _{i=1}^{n}(Y_{i}-{\hat {Y_{i}}})^{2}


To be called in a following way:

.. code:: python

  metric = MSEMetric()


In ``OnlineTraceFitter`` mean square error gets calculated in online manner,
with no need of specifying a metric object.
