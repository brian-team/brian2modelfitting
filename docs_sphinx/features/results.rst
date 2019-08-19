Results
=======

Fitter class stores all of the parameters examined by the optimizer as well
as the corresponding error. To retrieve them you can call the ``fitter.results()``.


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
