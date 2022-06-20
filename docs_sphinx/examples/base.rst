Simple Examples
===============

Following pieces of code show an example of two ``Fitter`` class calls and an
``Inferencer`` class call with possible inputs.


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

Inferencer
----------

.. code:: python

  v_features = [
      lambda x: max(x[(t > t_start) & (t < t_end)]),  # AP max
      lambda x: mean(x[(t > t_start) & (t < t_end)]),  # AP mean
      lambda x: std(x[(t > t_start) & (t < t_end)]),  # AP std
      lambda x: mean(x[(t > .25 * t_start) & (t < .75 * t_start)]),  # resting
  ]
  s_features = [lambda x: x.size]  # number of spikes in a spike train

  inferencer = Inferencer(model=eqs, dt=0.1*ms,
                          input={'I': inp_traces},
                          output={'v': out_traces, 'spike': spike_times}, 
                          features={'v': v_features, 'spikes': s_features},
                          method='exponential_euler',
                          threshold='m > 0.5',
                          refractory='m > 0.5',
                          param_init={'v': 'VT'})

  posterior = inferencer.infer(n_samples=1_000,
                               n_rounds=2,
                               inference_method='SNPE',
                               gL=[20*nS, 40*nS],
                               C = [0.5*nF, 1.5*nF])
