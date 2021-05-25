Advanced Features
=================

This part of documentation list other features provided alongside or inside `~brian2modelfitting.fitter.Fitter`
objects, to help the user with easier and more flexible applications.

.. contents::
    :local:
    :depth: 1

Parameters initialization
-------------------------

Whilst running `~brian2modelfitting.fitter.Fitter` user can specify values with which model evaluation
of differential equations start.

The fitting functions accept additional dictionary input to address that. To do so,
dictionary argument has to be added to `~brian2modelfitting.fitter.Fitter` initialization:

.. code:: python

   param_init = {'v': -30*mV}


.. code:: python

   fitter = TraceFitter(..., param_init = {'v': -30*mV})


Restart
-------
By default any `~brian2modelfitting.fitter.Fitter` works in continuous optimization mode between run, where all of the
parameters drawn are being evaluated.

Through changing the input flag in `~brian2modelfitting.fitter.Fitter.fit()`: ``restart`` to ``True``, user can reset the optimizer and
start the optimization from scratch.

Used by Fitter optimizer and metric can only be changed when the flat is `True`.


Multiobjective normalization
----------------------------
It is possible to fit more than one output variable at the same time, by combining the errors for each variable. To do
so, users can specify several output variables during the initialization::

    fitter = TraceFitter(..., output={'x': target_x,
                                      'y': target_y},
                         ...)

If the fit uses a single metric, it applies to both variables. Note that this requires that the resulting error has the
same units for both variables – for example, it would not be possible to use the same `.MSEMetric` on variables with
different units, since the errors cannot be simply added up. As a more general solution, users can specify a metric for
each variable and use their normalization arguments to make the units compatible (most commonly by turning both errors
into dimensionless quantities). This normalization also defines the relative weights of both errors. For example, if the
variable ``x`` has dimensions of volt and the variable ``y`` is dimensionless, the following metrics can be used to make
an error of 10mV in ``x`` to be weighed as much as an error of 0.1 in ``y``::

    metrics = {'x': MSEMetric(normalization=10*mV)
               'y': MSEMetric(normalization=0.1)}

This dictionary then has to be provided as the ``metric`` argument of the `~.Fitter.fit` function.


Callback function
-----------------

To visualize the progress of the optimization we provided few possibilities of feedback
inside `~brian2modelfitting.fitter.Fitter`.


The 'callback' input provides few default options, updated in each round:
 - ``'text'`` (default) that prints out the parameters of the best fit and corresponding error
 - ``'progressbar'`` that uses ``tqdm.autonotebook`` to provide a progress bar
 - ``None`` for non-verbose option

as well as **customized feedback option**. User can provide
a *callable* (i.e. function), that will provide an output or printout. If callback returns
``True`` the fitting execution is interrupted.

User gets four arguments to customize over:
 - ``params`` - set of parameters from current round
 - ``errors`` - set of errors from current round
 - ``best_params`` - best parameters globally, from all rounds
 - ``best_error`` - best parameters globally, from all rounds
 - ``index`` - index of current round

An example function:

.. code:: python

 def callback(params, errors, best_params, best_error, index):
    print('index {} errors minimum: {}'.format(index, min(errors)))

.. code:: python

   fitter = TraceFitter(...)
   result, error  = fitter.fit(..., callback=...)



Generate Traces
---------------

With the same `~brian2modelfitting.fitter.Fitter` class user can also generate the traces with newly
optimized parameters.

To simulate and visualize the traces or spikes for the parameters of choice.
For a quick access to best fitted set of parameters Fitter classes provided
ready to use functions:

 - `~brian2modelfitting.fitter.TraceFitter.generate_traces` inside `~brian2modelfitting.fitter.TraceFitter`
 - `~brian2modelfitting.fitter.SpikeFitter.generate_spikes` inside `~brian2modelfitting.fitter.SpikeFitter`

Functions can be called after fitting in the following manner, without
any input arguments:

.. code:: python

    fitter = TraceFitter(...)
    results, error = fitter.fit(...)
    traces = fitter.generate_traces()

.. code:: python

    fitter = SpikeFitter(...)
    results, error = fitter.fit(...)
    spikes = fitter.generate_spikes()


Custom generate
~~~~~~~~~~~~~~~

To create traces for other parameters, or generate traces after spike
train fitting, user can call the - `~brian2modelfitting.fitter.Fitter.generate` call, that takes in following
arguments:

.. code:: python

  fitter.generate(params=None, output_var=None, param_init=None, level=0)

Where ``params`` is a dictionary of parameters for which the traces we generate.
``output_var`` provides an option to pick one or more variable for visualization. With
``param_init``, user can define the initial values for differential equations.
``level`` allows for specification of namespace level from which we get
the constant parameters of the model.

If ``output_var`` is the name of a single variable name (or the special name ``'spikes'``), a single `~.Quantity`
(for normal variables) or a list of spikes time arrays (for ``'spikes'``) will be returned. If a list of names is
provided, then the result is a dictionary with all the results.

.. code:: python

    fitter = TraceFitter(...)
    results, error = fitter.fit(...)
    traces = fitter.generate(output_var=['v', 'h', 'n', 'm'])
    v_trace = traces['v']
    h_trace = traces['h']
    ...


Results
-------

Fitter class stores all of the parameters examined by the optimizer as well
as the corresponding error. To retrieve them you can call the - `~brian2modelfitting.fitter.Fitter.results`.


.. code:: python

    fitter = TraceFitter(...)
    ...
    traces = fitter.generate_traces()

.. code:: python

    fitter = SpikeFitter(...)
    ...
    results = fitter.results(format='dataframe')


Results can be returned in one of the following formats:

 - ``'list'`` (default) returns a list of dictionaries with corresponding parameters (including units) and errors
 - ``'dict'`` returns a dictionary of arrays with corresponding parameters (including units) and errors
 - ``'dataframe'`` returns a `~pandas.DataFrame` (without units)

The use of units (only relevant for formats ``'list'`` and ``'dict'``) can be switched
on or off with the ``use_units`` argument. If it is not specified, it will default to
the value used during the initialization of the `Fitter` (which itself defaults to
``True``).

Example output:
~~~~~~~~~~~~~~~
``'list'``:

.. code:: python

  [{'gl': 80.63365773 * nsiemens, 'g_kd': 66.00430921 * usiemens, 'g_na': 145.15634566 * usiemens, 'errors': 0.00019059452295872703},
   {'gl': 83.29319947 * nsiemens, 'g_kd': 168.75187749 * usiemens, 'g_na': 130.64547027 * usiemens, 'errors': 0.00021434415430605653},
   ...]


``'dict'``:

.. code:: python

  {'g_na': array([176.4472297 , 212.57019659, ...]) * usiemens,
   'g_kd': array([ 43.82344525,  54.35309635, ...]) * usiemens,
   'gl': array([ 69.23559876, 134.68463669, ...]) * nsiemens,
   'errors': array([1.16788502, 0.5253008 , ...])}


``'dataframe'``:

.. code:: python

   g_na            gl      g_kd    errors
   0  0.000280  8.870238e-08  0.000047  0.521425
   1  0.000192  1.121861e-07  0.000118  0.387140
   ...



Standalone mode
---------------

Just like with regular Brian script, modelfitting computations can be performed in
``Runtime`` mode (default) or ``Standalone`` mode.
(https://brian2.readthedocs.io/en/stable/user/computation.html)

To enable this mode, add the following line after your Brian import, but before your simulation code:

.. code:: python

  set_device('cpp_standalone')


Important notes:
~~~~~~~~~~~~~~~~

.. warning::
    In standlone mode one script can not be used to contain multiple - `~brian2modelfitting.fitter.Fitter`, use separate scripts!

Note that the generation of traces or spikes via `~brian2modelfitting.fitter.Fitter.generate`
will always use runtime mode, even when the fitting procedure uses standalone mode.


OnlineTraceFitter
-----------------

`~brian2modelfitting.fitter.OnlineTraceFitter` was created to work with long traces or big optimization.
This `~brian2modelfitting.fitter.Fitter` uses online Mean Square Error as a metric.
When `~brian2modelfitting.fitter.Fitter.fit()` is called there is no need of specifying a metric, that is by
default set to None. Instead the errors are calculated with use of brian's `~brian2.groups.group.Group.run_regularly`,
with each simulation.

.. code:: python

  fitter = OnlineTraceFitter(model=model,
                             input={'I': inp_traces},
                             output={'v': out_traces},
                             dt=0.1*ms,
                             n_samples=5)

  result, error = fitter.fit(optimizer=optimizer,
                             n_rounds=1,
                             gl=[1e-8*siemens*cm**-2 * area, 1e-3*siemens*cm**-2 * area],)


Reference the target values in the equations
--------------------------------------------

A model can refer to the target output values within the equations. For example, if you
are fitting a membrane potential trace *v* (i.e. `output_var='v'`), then the equations
can refer to the target trace as `v_target`. This allows you for example to add a coupling
term like `coupling*(v_target - v)` to the equation for `v`, pulling the trajectory towards the
correct solution.