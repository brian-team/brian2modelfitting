Simple Examples
===============

fit_spikes
----------

.. code:: python

  n_opt = NevergradOptimizer('DE')
  metric = GammaFactor(dt, 60*ms)


  params, error = fit_spikes(model=eqs, input_var='I', dt=0.1*ms,
                             input=inp_traces, output=out_spikes,
                             n_rounds=2, n_samples=30, optimizer=n_opt,
                             metric=metric,
                             threshold='v > -50*mV',
                             reset='v = -70*mV',
                             method='exponential_euler',
                             param_init={'v': -70*mV},
                             gL=[20*nS, 40*nS],
                             C = [0.5*nF, 1.5*nF])



fit_traces
----------

.. code:: python

  n_opt = NevergradOptimizer(method='PSO')
  metric = MSEMetric()

  params, error = fit_traces(model=model,
                             input_var='I',
                             output_var='v',
                             input=inp_trace,
                             output=out_trace,
                             param_init={'v': -65*mV},
                             method='exponential_euler',
                             dt=0.1*ms,
                             optimizer=n_opt,
                             metric=metric,
                             callback=True,
                             n_rounds=1, n_samples=5,
                             gl=[1e-8*siemens*cm**-2 * area, 1e-3*siemens*cm**-2 * area],
                             g_na=[1*msiemens*cm**-2 * area, 2000*msiemens*cm**-2 * area],
                             g_kd=[1*msiemens*cm**-2 * area, 1000*msiemens*cm**-2 * area],)
