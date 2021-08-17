Inference on Hodgin-Huxley model: simple interface
==================================================

You can also download and run a similar example available here:
:download:`hh_sbi_simple_interface.py <../../examples/hh_sbi_simple_interface.py>`

Here you can download the data:
:download:`input traces <../../examples/input_traces_hh.csv>`
:download:`output traces <../../examples/output_traces_hh.csv>`

.. code:: python

  from brian2 import *
  from brian2modelfitting import *
  import pandas as pd

To load the data, use the following:

.. code:: python

  df_inp_traces = pd.read_csv('input_traces_hh.csv')
  df_out_traces = pd.read_csv('output_traces_hh.csv')
  inp_traces = df_inp_traces.to_numpy()
  inp_traces = inp_traces[[0, 1], 1:]
  out_traces = df_out_traces.to_numpy()
  out_traces = out_traces[[0, 1], 1:]

Then we have to define the model and its parameters:

.. code:: python

  area = 20_000*um**2
  El = -65*mV
  EK = -90*mV
  ENa = 50*mV
  VT = -63*mV
  dt = 0.01*ms
  eqs = '''
      dv/dt = (gl*(El-v) - g_na*(m*m*m)*h*(v-ENa) - g_kd*(n*n*n*n)*(v-EK) + I)/Cm : volt
      dm/dt = 0.32*(mV**-1)*(13.*mV-v+VT)/
              (exp((13.*mV-v+VT)/(4.*mV))-1.)/ms*(1-m)-0.28*(mV**-1)*(v-VT-40.*mV)/
              (exp((v-VT-40.*mV)/(5.*mV))-1.)/ms*m : 1
      dn/dt = 0.032*(mV**-1)*(15.*mV-v+VT)/
              (exp((15.*mV-v+VT)/(5.*mV))-1.)/ms*(1.-n)-.5*exp((10.*mV-v+VT)/(40.*mV))/ms*n : 1
      dh/dt = 0.128*exp((17.*mV-v+VT)/(18.*mV))/ms*(1.-h)-4./(1+exp((40.*mV-v+VT)/(5.*mV)))/ms*h : 1

      # free parameters
      g_na : siemens (constant)
      g_kd : siemens (constant)
      gl   : siemens (constant)
      Cm   : farad (constant)
  '''

Let's also specify time domain for more convenient plotting afterwards:

.. code:: python

  t = arange(0, out_traces.shape[1]*dt/ms, dt/ms)
  stim_start, stim_end = t[where(inp_traces[0, :] != 0)[0][[0, -1]]]

Now, we have to define features in order to create a summary statistics representation of the output data traces:

.. code:: python
  
    list_of_features = [
        lambda x: max(x[(t > stim_start) & (t < stim_end)]),  # max active potential
        lambda x: mean(x[(t > stim_start) & (t < stim_end)]),  # mean active potential
        lambda x: std(x[(t > stim_start) & (t < stim_end)]),  # std active potential
        lambda x: mean(x[(t > .25 * stim_start) & (t < .75 * stim_start)]),  # resting
    ]

We have to instantiate the object by using the class ``Inferencer`` in which the data and the list of features should be passed:

.. code:: python

  inferencer = Inferencer(dt=dt, model=eqs,
                          input={'I': inp_traces*amp},
                          output={'v': out_traces*mV},
                          features={'v': list_of_features},
                          method='exponential_euler',
                          threshold='m > 0.5',
                          refractory='m > 0.5',
                          param_init={'v': 'VT'})



Be sure that the names of parameters passed to the ``infer`` method correspond to the names of unknown parameters defined as constatns in the model equations.

.. code:: python

  posterior = inferencer.infer(n_samples=5_000,
                               n_rounds=3,
                               inference_method='SNPE',
                               density_estimator_model='mdn',
                               gl=[1e-09*siemens, 1e-07*siemens],
                               g_na=[2e-06*siemens, 2e-04*siemens],
                               g_kd=[6e-07*siemens, 6e-05*siemens],
                               Cm=[0.1*uF*cm**-2*area, 2*uF*cm**-2*area])

After the training of the neural density estimator stored accessible through ``posterior`` is done, we can draw samples from the approximated posterior distribution as follows:

.. code:: python

  samples = inferencer.sample((5_000, ))

In order to analyze the sampled data further, we can use the embedded ``pairplot`` method which visualizes the pairwise relationship between each two parameters:

.. code:: python

  limits = {'gl': [1e-9*siemens, 1e-07*siemens],
            'g_na': [2e-06*siemens, 2e-04*siemens],
            'g_kd': [6e-07*siemens, 6e-05*siemens],
            'Cm': [0.1*uF*cm**-2*area, 2*uF*cm**-2*area]}
  labels = {'gl': r'$\overline{g}_{l}$',
            'g_na': r'$\overline{g}_{Na}$',
            'g_kd': r'$\overline{g}_{K}$',
            'Cm': r'$C_{m}$'}
  inferencer.pairplot(limits=limits,
                      labels=labels,
                      ticks=limits,
                      figsize=(6, 6))
  condition = inferencer.sample((1, ))
  inferencer.conditional_pairplot(condition=condition,
                                  limits=limits,
                                  labels=labels,
                                  ticks=limits,
                                  figsize=(6, 6))

To obtain a simulated trace from a single sample of parameters drawn from posterior distribution, use the following code:

.. code:: python

  inf_traces = inferencer.generate_traces(output_var='v')

Let us now visualize the recordings and simulated traces:

.. code:: python

  inf_traces = inferencer.generate_traces(output_var='v')

  nrows = 2
  ncols = out_traces.shape[0]
  fig, axs = subplots(nrows, ncols, sharex=True,
                      gridspec_kw={'height_ratios': [3, 1]}, figsize=(9, 3))
  for idx in range(ncols):
      spike_idx = in1d(t, spike_times[idx]).nonzero()[0]
      spike_v = (out_traces[idx, :].min(), out_traces[idx, :].max())
      axs[0, idx].plot(t, out_traces[idx, :].T, 'C3-', lw=3, label='recordings')
      axs[0, idx].plot(t, inf_traces[idx, :].T/mV, 'k--', lw=2,
                       label='sampled traces')
      axs[1, idx].plot(t, inp_traces[idx, :].T/nA, lw=3, c='k', label='stimuli')
      axs[1, idx].set_xlabel('$t$, ms')
      if idx == 0:
          axs[0, idx].set_ylabel('$V$, mV')
          axs[1, idx].set_ylabel('$I$, nA')
  handles, labels = [(h + l) for h, l
                     in zip(axs[0, idx].get_legend_handles_labels(),
                     axs[1, idx].get_legend_handles_labels())]
  fig.legend(handles, labels)
  tight_layout()
  show()
