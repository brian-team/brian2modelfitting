Simple Examples
===============

Following pieces of code show an example of Fitter class calls with possible inputs.


TraceFitter
------------

.. code:: python

  n_opt = NevergradOptimizer(method='PSO')
  metric = MSEMetric()

  fitter = TraceFitter(model=model,
                        input={'I': inp_trace},
                        output={'v': out_trace},
                        dt=0.1*ms, n_samples=5
                        method='exponential_euler')

  results, error = fitter.fit(optimizer=n_opt,
                              metric=metric,
                              callback='text',
                              n_rounds=1,
                              param_init={'v': -65*mV},
                              gl=[10*nS*cm**-2 * area, 1*mS*cm**-2 * area],
                              g_na=[1*mS*cm**-2 * area, 2000*mS*cm**-2 * area],
                              g_kd=[1*mS*cm**-2 * area, 1000*mS*cm**-2 * area])



SpikeFitter
-----------

.. code:: python

  n_opt = SkoptOptimizer('ET')
  metric = GammaFactor(dt, delta=2*ms)

  fitter = SpikeFitter(model=eqs,
                       input={'I': inp_traces},
                       output=out_spikes,
                       dt=0.1*ms,
                       n_samples=30,
                       threshold='v > -50*mV',
                       reset='v = -70*mV',
                       method='exponential_euler')

  results, error = fitter.fit(n_rounds=2,
                   optimizer=n_opt,
                   metric=metric,
                   gL=[20*nS, 40*nS],
                   C = [0.5*nF, 1.5*nF])
