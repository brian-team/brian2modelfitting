Inference on Hodgin-Huxley model: flexible interface
====================================================

You can also download and run this example by clicking here:
:download:`hh_sbi_simple_interface.py <../../examples/hh_sbi_flexible_interface.py>`

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
  inp_traces = inp_traces[[0, 1, 3], 1:]
  out_traces = df_out_traces.to_numpy()
  out_traces = out_traces[[0, 1, 3], 1:]

The model used for this example is the Hodgkin-Huxley neuron model.
The parameters of the model are defined as follows:

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

      # unknown parameters
      g_na : siemens (constant)
      g_kd : siemens (constant)
      gl   : siemens (constant)
      Cm   : farad (constant)
      '''

Now, let's define the time domain and start with the inferencer procedure
manually:

.. code:: python
    
  t = arange(0, out_traces.shape[1]*dt/ms, dt/ms)
  t_start, t_end = t[where(inp_traces[0, :] != 0)[0][[0, -1]]]

  # Start with the regular instatiation of the class
  inferencer = Inferencer(dt=dt, model=eqs,
                          input={'I': inp_traces*amp},
                          output={'v': out_traces*mV},
                          features={'v': [lambda x: max(x),
                                          lambda x: mean(x[(t > t_start) & (t < t_end)]),
                                          lambda x: std(x[(t > t_start) & (t < t_end)])]},
                          method='exponential_euler',
                          threshold='m > 0.5',
                          refractory='m > 0.5',
                          param_init={'v': 'VT'})

The prior should be initialized by defining the upper and lower bounds for
each unknown parameter:

.. code:: python
  
  prior = inferencer.init_prior(gl=[1e-09*siemens, 1e-07*siemens],
                                g_na=[2e-06*siemens, 2e-04*siemens],
                                g_kd=[6e-07*siemens, 6e-05*siemens],
                                Cm=[0.1*uF*cm**-2*area, 2*uF*cm**-2*area])

If the input and output data for the training of the neural density estimator
already exists, we can load it as follows:

.. code:: python

  path_to_data = ...
  theta, x = inferencer.load_summary_statistics(path_to_data)

Otherwise, we have to generate training data and summary statistics from a
given list of features:

.. code:: python

  theta = inferencer.generate_training_data(n_samples=10_000,
                                            prior=prior)
  x = inferencer.extract_summary_statistics(theta)

And the data can be saved for the later use:

.. code:: python
 
  inferencer.save_summary_statistics(path_to_data, theta, x)

Finally, let's get our hands dirty and let's perform a single step of
inference:

.. code:: python

  # amortized inference
  inference = inferencer.init_inference(inference_method='SNPE',
                                        density_estimator_model='mdn',
                                        prior=prior)
  # first round of inference where no observation data is set to posterior
  posterior_amortized = inferencer.infer_step(proposal=prior,
                                              inference=inference,
                                              theta=theta, x=x)

After the posterior has been built, it can be stored as follows:                                              

.. code:: python
  
  # storing the trained posterior without a default observation
  path_to_posterior = ...
  inferencer.save_posterior(path_to_posterior)

Now, as in the simple interface example, sampling can be performed via
``sample`` method where it is enough to define a number of parameters to
be drawn from the posterior:

.. code:: python

  inferencer.sample((10_000, ))


Creating the pairwise relationship visualizations using the approximated
posterior distribution

.. code:: python

  # define the label for each parameter
  labels = {'gl': r'$\overline{g}_\mathrm{l}$',
            'g_na': r'$\overline{g}_\mathrm{Na}$',
            'g_kd': r'$\overline{g}_\mathrm{K}$',
            'Cm': r'$\overline{C}_{m}$'}
  inferencer.pairplot(labels=labels)


It is possible to continue with the focused inference (to draw parameters
from the posterior and to perform the training of a neural network to
estimate the posterior distribution by focusing on a particular observation)
by using a standard approach through ``infer`` method:

.. code:: python

  posterior_focused = inferencer.infer()

For every future call of ``inferencer``, the last trained posterior will be
used by default, e.g., when generating traces by using a single sample of
parameters from a now non-amortized approximated posterior distribution:

.. code:: python

  inf_traces = inferencer.generate_traces()
  nrows = 2
  ncols = out_traces.shape[0]
  fig, axs = subplots(nrows, ncols, sharex=True,
                      gridspec_kw={'height_ratios': [3, 1]},
                      figsize=(ncols * 3, 3))
  for idx in range(ncols):
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
