Advanced Features
=================

This part of documentation list other features provided alongside or inside :py:class:`~brian2modelfitting.modelfitting.modelfitting.Fitter`
objects, to help the user with easier and more flexible applications.

.. contents::
    :local:
    :depth: 1

Parameters initialization
-------------------------

Whilst running :py:class:`~brian2modelfitting.modelfitting.modelfitting.Fitter` user can specify values with which model evaluation
of differential equations start.

The fitting functions accept additional dictionary input to address that. To do so,
dictionary argument has to be added to :py:class:`~brian2modelfitting.modelfitting.modelfitting.Fitter` initialization:

.. code:: python

   param_init = {'v': -30*mV}


.. code:: python

   fitter = TraceFitter(..., param_init = {'v': -30*mV})


Restart
-------
By default any :py:class:`~brian2modelfitting.modelfitting.modelfitting.Fitter` works in continuous optimization mode between run, where all of the
parameters drawn are being evaluated.

Through changing the input flag in :py:func:`~brian2modelfitting.modelfitting.modelfitting.Fitter.fit()`: ``restart`` to ``True``, user can reset the optimizer and
start the optimization from scratch.

Used by Fitter optimizer and metric can only be changed when the flat is `True`.




Callback function
-----------------

To visualize the progress of the optimization we provided few possibilities of feedback
inside :py:class:`~brian2modelfitting.modelfitting.modelfitting.Fitter`.


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

With the same :py:class:`~brian2modelfitting.modelfitting.modelfitting.Fitter` class user can also generate the traces with newly
optimized parameters.

To simulate and visualize the traces or spikes for the parameters of choice.
For a quick access to best fitted set of parameters Fitter classes provided
ready to use functions:

 - :py:func:`~brian2modelfitting.modelfitting.modelfitting.TraceFitter.generate_traces` inside :py:class:`~brian2modelfitting.modelfitting.modelfitting.TraceFitter`
 - :py:func:`~brian2modelfitting.modelfitting.modelfitting.SpikeFitter.generate_spikes` inside :py:class:`~brian2modelfitting.modelfitting.modelfitting.SpikeFitter`

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
~~~~~~~~~~~~~~~

To create traces for other parameters, or generate traces after spike
train fitting, user can call the - :py:func:`~brian2modelfitting.modelfitting.modelfitting.Fitter.generate` call, that takes in following
arguments:

.. code:: python

  fitter.generate(params=None, output_var=None, param_init=None, level=0)

Where ``params`` is a dictionary of parameters for which the traces we generate.
``output_var`` provides an option to pick variable for visualization. With
``param_init``, user can define the initial values for differential equations.
``level`` allows for specification of namespace level from which we get
the constant parameters of the model.



Results
-------

Fitter class stores all of the parameters examined by the optimizer as well
as the corresponding error. To retrieve them you can call the - :py:func:`~brian2modelfitting.modelfitting.modelfitting.Fitter.results`.


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
 - ``'dataframe'`` returns `pandas dataframe` (without units)


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
    In standlone mode one script can not be used to contain multiple - :py:class:`~brian2modelfitting.modelfitting.modelfitting.Fitter`, use separate scripts!


Before generation of traces, you have to  reinitialize the device add additional
piece of code before calling :py:func:`~brian2modelfitting.modelfitting.modelfitting.Fitter.generate`:

.. code:: python

    device.reinit()
    device.activate()

    fitter.generate_traces()


.. warning::
     Device reinitialization causes the device to reset, and disables the possibility for further fitting or
     retrieving information from fitter monitors.



OnlineTraceFitter
-----------------

:py:class:`~brian2modelfitting.modelfitting.modelfitting.OnlineTraceFitter` was created to work with long traces or big optimization.
This :py:class:`~brian2modelfitting.modelfitting.modelfitting.Fitter` uses online Mean Square Error as a metric.
When :py:func:`~brian2modelfitting.modelfitting.modelfitting.Fitter.fit()` is called there is no need of specifying a metric, that is by
default set to None. Instead the errors are calculated with use of brian's :py:meth:`~brian2.groups.group.Group.run_regularly`,
with each simulation.

.. code:: python

  fitter = OnlineTraceFitter(model=model,
                             input=inp_traces,
                             output=out_traces,
                             input_var='I',
                             output_var='v',
                             dt=0.1*ms,
                             n_samples=5)

  result, error = fitter.fit(optimizer=optimizer,
                             n_rounds=1,
                             gl=[1e-8*siemens*cm**-2 * area, 1e-3*siemens*cm**-2 * area],)
