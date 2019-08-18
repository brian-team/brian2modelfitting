How it works
============

Model fitting requires two components:
 - A **metric**: to compare results and decide which one is the best
 - An **optimization** algorithm: to decide which parameter combinations to try

That need to be specified before initialization of the fitting function.

Each optimization works with a following scheme:

.. code:: python

  opt = Optimizer()
  metric = Metric()

  params, error = fit_traces(metric=metric, optimizer=opt, ...)
  params, error = fit_spikes(metric=metric, optimizer=opt, ...)


The proposed solution is developed using a modular approach, where both the optimization
method and metric to be optimized can be easily swapped out by a custom implementation.

Both fitting functions require 'model' defined as ``Equation`` object, that has parameters that will be
optimized specified as constants in a following way:

.. code:: python

  model = '''
  ...
  g_na : siemens (constant)
  g_kd : siemens (constant)
  gl   : siemens (constant)
  '''


Additionally, fitting function requires:
 - `reset`, and `threshold` in case of spiking neurons (can take refractory as well)
 - `dt` - time step
 - `input` - set of input traces (list or array)
 - `output` - set of goal output (traces/spike trains) (list or array)
 - `input_var` - name of the input trace variable (string)
 - `output_var` - name of the output trace variable (string)
 - `n_rounds` - number of rounds to optimize over
 - `n_samples` - number of samples to draw in each round (limited by method)

Each free parameter of the model that shall be fitted is defined by two values:

.. code:: python

  param_name = [min, max]


Example of `fit_traces()` with all of the necessary arguments:

.. code:: python

  params, error = fit_traces(model=model,
                             input=inp_traces,
                             output=out_traces,
                             input_var='I',
                             output_var='v',
                             dt=0.1*ms,
                             optimizer=opt,
                             metric=metric,
                             n_rounds=1, n_samples=5,
                             gl=[1e-8*siemens*cm**-2 * area, 1e-3*siemens*cm**-2 * area],)
