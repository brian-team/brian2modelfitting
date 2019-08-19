Generate Traces
===============

With the same Fitter class user can also generate the traces with newly
optimized parameters.

To simulate and visualize the traces or spikes for the parameters of choice.
For a quick access to best fitted set of parameters Fitter classes provided
ready to use functions:
 - ``generate_traces()`` inside ``TracesFitter``
 - ``generate_spikes()`` inside ``SpikeFitter``

Functions can be called after fitting in the following manner, without
any input arguments:

.. code:: python

    fitter = TraceFitter(...)
    results, error = fitter.fit(...)
    traces = fitter.generate_traces()

.. code:: python

    fitter = SpikeFitter(...)
    results, error = fitter.fit(...)
    spikes = fitter.generate_traces()


Custom generate
---------------

To create traces for other parameters, or generate traces after spike
train fitting, user can call the ``generate()`` call, that takes in following
arguments:

.. code:: python

  fitter.generate(params=None, output_var=None, param_init=None, level=0)

Where ``params`` is a dictionary of parameters for which the traces we generate.
``output_var`` provides an option to pick variable for visualization. With
``param_init``, user can define the initial values for differential equations.
``level`` allows for specification of namespace level from which we get
the constant parameters of the model.
