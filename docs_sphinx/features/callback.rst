Callback function
=================

To visualize the progress of the optimization we provided few possibilities of feedback
inside `Fitters`.


The 'callback' input provides few default options, updated in each round:
 - ``'text'`` (default)
   that prints out the parameters of the best fit and corresponding error
 - ``'progressbar'``
   that uses tqdm.autonotebook to provide a progress bar
 - ``None``
   for non-verbose option

as well as **customized feedback option**. User can provide
a *callable* (i.e. function), that will provide an output or printout. If callback returns
`True` the fitting execution is interrupted.
User gets four arguments to customize over:

``results, errors, parameters, index``

An example function:

.. code:: python

  def callback(results, errors, parameters, index):
      print('index {} errors minimum: {}'.format(index, min(errors)) )


.. code:: python

    fitter = TraceFitter(...)
    result, error  = fitter.run(..., callback=)
