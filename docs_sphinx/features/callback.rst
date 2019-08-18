Callback function
=================

The 'callback' input provides custom feedback function option. User can provide
a callable (function), that will provide an output or printout. If callback returns
`True` the fitting execution is interrupted.
User gets four arguments to customize over:

``results, errors, parameters, index``

An example function:

.. code:: python

  def callback(results, errors, parameters, index):
      print('index {} errors minimum: {}'.format(index, min(errors)) )

- 'progressbar'
- 'print'
- callback
