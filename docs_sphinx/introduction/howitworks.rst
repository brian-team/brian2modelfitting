How it works
============

Model fitting script requires three components:
 - A **Fitter** of choice: object that will perform the optimization
 - A **metric**: to compare results and decide which one is the best
 - An **optimization** algorithm: to decide which parameter combinations to try

All of which need to be initialized for fitting application.
Each optimization works with a following scheme:

.. code:: python

  opt = Optimizer()
  metric = Metric()

  fitter = Fitter(...)
  result, error = fitter.fit(metric=metric, optimizer=opt, ...)


The proposed solution is developed using a modular approach, where both the optimization
method and metric to be optimized can be easily swapped out by a custom implementation.

``Fitter`` objects require 'model' defined as ``Equation`` object, that has parameters that will be
optimized specified as constants in a following way:

.. code:: python

  model = '''
  ...
  g_na : siemens (constant)
  g_kd : siemens (constant)
  gl   : siemens (constant)
  '''

Initialization of Fitter requires:
  - `dt` - time step
  - `input` - set of input traces (list or array)
  - `output` - set of goal output (traces/spike trains) (list or array)
  - `input_var` - name of the input trace variable (string)
  - `output_var` - name of the output trace variable (string)
  - `n_samples` - number of samples to draw in each round (limited by method)
  - `reset`, and `threshold` in case of spiking neurons (can take refractory as well)



Additionally, upon call of `fit()`, object requires:
 - `n_rounds` - number of rounds to optimize over
 - parameters with ranges to be optimized over

...as well as an `optimizer` and a `metric`

Each free parameter of the model that shall be fitted is defined by two values:

.. code:: python

  param_name = [min, max]

Ready to use elements
---------------------

Alongside three optimization classes:
 - `TraceFitter`
 - `SpikeFitter`
 - `OnlineTraceFitter`

We also provide ready optimizers:
 - `NevergradOptimizer`
 - `SkoptOptimizer`

and metrics:
 - `MSE` (for TraceFitter)
 - `GammaFactor` (for SpikeFitter)


Example of `TraceFitter()` with all of the necessary arguments:

.. code:: python

  fitter = TraceFitter(model=model,
                       input=inp_traces,
                       output=out_traces,
                       input_var='I',
                       output_var='v',
                       dt=0.1*ms,
                       n_samples=5)

  result, error = fitter.fit(optimizer=optimizer,
                             metric=metric,
                             n_rounds=1,
                             gl=[1e-8*siemens*cm**-2 * area, 1e-3*siemens*cm**-2 * area],)

Remarks
-------
 - After performing first fitting, user can continue the optimization
   with another `fitter.fit()` run.

 - Number of samples can not be changed between rounds or `fit()` calls,
   due to parallelization of the simulations.

 - User is not allowed to change the optimizer or metric between `fit()` calls.
