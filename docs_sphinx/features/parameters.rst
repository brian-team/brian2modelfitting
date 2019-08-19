Parameters initialization
=========================

Whilst running ``fitter.fit()`` user can specify values with which model evaluation
of differential equations start.

The fitting functions accept additional dictionary input to address that. To do so,
dictionary argument has to be added to ``fit()`` call:

.. code:: python

  param_init = {'v': -30*mV}


.. code:: python

    fitter = TraceFitter(...)
    result, error  = fitter.run(..., param_init = {'v': -30*mV})
