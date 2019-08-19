OnlineTraceFitter
=================

``OnlineTraceFitter`` was created to work with long traces or big optimization.
This ``Fitter`` uses online Mean Square Error as a metric.
When ``fit()`` is called there is no need of specifying a metric, that is by
default set to None. Instead the errors are calculated with use of brian's ``run_regularly``,
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
