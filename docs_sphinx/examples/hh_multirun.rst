Multirun fitting of Hodgkin-Huxley
==================================

Here you can download the data:
:download:`input_traces <../../examples/input_traces_hh.csv>`
:download:`output_traces <../../examples/output_traces_hh.csv>`

.. code:: python

  import numpy as np
  from brian2 import *
  from brian2modelfitting import *


To load the data, use following code:

.. code:: python

  import pandas as pd
  # Load Input and Output Data
  df_inp_traces = pd.read_csv('input_traces_hh.csv')
  df_out_traces = pd.read_csv('output_traces_hh.csv')

  inp_traces = df_inp_traces.to_numpy()
  inp_traces = inp_traces[:, 1:]

  out_traces = df_out_traces.to_numpy()
  out_traces = out_traces[:, 1:]

Then the multiple round optimization can be run with following code:

.. code:: python

  # Model Fitting
  ## Parameters
  area = 20000*umetre**2
  El = -65*mV
  EK = -90*mV
  ENa = 50*mV
  VT = -63*mV
  dt = 0.01*ms
  defaultclock.dt = dt

  ## Modle Definition
  eqs = Equations(
  '''
  dv/dt = (gl*(El-v) - g_na*(m*m*m)*h*(v-ENa) - g_kd*(n*n*n*n)*(v-EK) + I)/Cm : volt
  dm/dt = 0.32*(mV**-1)*(13.*mV-v+VT)/
      (exp((13.*mV-v+VT)/(4.*mV))-1.)/ms*(1-m)-0.28*(mV**-1)*(v-VT-40.*mV)/
      (exp((v-VT-40.*mV)/(5.*mV))-1.)/ms*m : 1
  dn/dt = 0.032*(mV**-1)*(15.*mV-v+VT)/
      (exp((15.*mV-v+VT)/(5.*mV))-1.)/ms*(1.-n)-.5*exp((10.*mV-v+VT)/(40.*mV))/ms*n : 1
  dh/dt = 0.128*exp((17.*mV-v+VT)/(18.*mV))/ms*(1.-h)-4./(1+exp((40.*mV-v+VT)/(5.*mV)))/ms*h : 1
  g_na : siemens (constant)
  g_kd : siemens (constant)
  gl   : siemens (constant)
  Cm   : farad (constant)
  ''')

  ## Optimization and Metric Choice
  n_opt = NevergradOptimizer()
  metric = MSEMetric()

  ## Fitting
  fitter = TraceFitter(model=eqs, input={'I': inp_traces*amp},
                       output={'v': out_traces*mV},
                       dt=dt, n_samples=20, param_init={'v': -65*mV},
                       method='exponential_euler')

  res, error = fitter.fit(n_rounds=2,
                          optimizer=n_opt, metric=metric,
                          callback='progressbar',
                          gl = [1e-09 *siemens, 1e-07 *siemens],
                          g_na = [2e-06*siemens, 2e-04*siemens],
                          g_kd = [6e-07*siemens, 6e-05*siemens],
                          Cm=[0.1*ufarad*cm**-2 * area, 2*ufarad*cm**-2 * area])

  ## Show results
  all_output = fitter.results(format='dataframe')
  print(all_output)

  # Second round
  res, error = fitter.fit(restart=True,
                          n_rounds=20,
                          optimizer=n_opt, metric=metric,
                          callback='progressbar',
                          gl = [1e-09 *siemens, 1e-07 *siemens],
                          g_na = [2e-06*siemens, 2e-04*siemens],
                          g_kd = [6e-07*siemens, 6e-05*siemens],
                          Cm=[0.1*ufarad*cm**-2 * area, 2*ufarad*cm**-2 * area])


To get the results and traces:

.. code:: python

  ## Show results
  all_output = fitter.results(format='dataframe')
  print(all_output)

  ## Visualization of the results
  fits = fitter.generate_traces(params=None, param_init={'v': -65*mV})

  fig, axes = plt.subplots(ncols=5, figsize=(20,5), sharey=True)

  for ax, data, fit in zip(axes, out_traces, fits):
      ax.plot(data.transpose())
      ax.plot(fit.transpose()/mV)

  plt.show()
