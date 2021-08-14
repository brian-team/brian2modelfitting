============
How it works
============

Fitting
=======

Model fitting script requires three components:
 - a **fitter**: object that will perform the optimization
 - a **metric**: objective function
 - an **optimizer**: optimization algorithm

All of which need to be initialized for fitting application.
Each optimization works with a following scheme:

.. code:: python

  opt = Optimizer()
  metric = Metric()
  fitter = Fitter(...)
  result, error = fitter.fit(metric=metric, optimizer=opt, ...)


The proposed solution is developed using a modular approach, where both the
optimization method and the objective function can be easily swapped out by a
user-defined custom implementation.

`~.Fitter` objects require a model defined as an
`~brian2.equations.equations.Equations` object or as a string, that has
parameters that will be optimized specified as constants in the following way:

.. code:: python

  model = '''
      ...
      g_na : siemens (constant)
      g_kd : siemens (constant)
      gl   : siemens (constant)
      '''

Initialization of Fitter requires:
  - ``dt`` - the time step
  - ``input`` - a dictionary with the name of the input variable and a set of
  input traces (list or array)
  - ``output`` - a dictionary with the name of the output variable(s) and a
  set of goal output (traces/spike trains) (list or array)
  - ``n_samples`` - a number of samples to draw in each round (limited by
  method)
  - ``reset`` and ``threshold`` in case of spiking neurons (can take
  refractory as well)

Additionally, upon call of `~brian2modelfitting.fitter.Fitter.fit()`,
object requires:

 - ``n_rounds`` - a number of rounds to optimize over
 - parameters with ranges to be optimized over

Each free parameter of the model that shall be fitted is defined by two values:

.. code:: python

  param_name = [lower_bound, upper_bound]

Ready to use elements
---------------------

Optimization classes:
 - `~brian2modelfitting.fitter.TraceFitter`
 - `~brian2modelfitting.fitter.SpikeFitter`
 - `~brian2modelfitting.fitter.OnlineTraceFitter`

Optimization algorithms:
 - `~brian2modelfitting.optimizer.NevergradOptimizer`
 - `~brian2modelfitting.optimizer.SkoptOptimizer`

Metrics:
 - `~brian2modelfitting.metric.MSEMetric` (for `~brian2modelfitting.fitter.TraceFitter`)
 - `~brian2modelfitting.metric.GammaFactor` (for `~brian2modelfitting.fitter.SpikeFitter`)


Example of `~brian2modelfitting.fitter.TraceFitter` with all of the necessary arguments:

.. code:: python

  fitter = TraceFitter(model=model,
                       input={'I': inp_traces},
                       output={'v': out_traces},
                       dt=0.1*ms,
                       n_samples=5)

  result, error = fitter.fit(optimizer=optimizer,
                             metric=metric,
                             n_rounds=1,
                             gl=[1e-8*siemens*cm**-2 * area, 1e-3*siemens*cm**-2 * area])

Remarks
-------
- After performing first fitting round, user can continue the optimization
  with another `~brian2modelfitting.fitter.Fitter.fit()` run.

- Number of samples can not be changed between rounds or `~brian2modelfitting.fitter.Fitter.fit()`
  calls, due to parallelization of the simulations.

.. warning::
  User is not allowed to change the optimizer or metric between `~brian2modelfitting.fitter.Fitter.fit()`
  calls.

- When using the `~brian2modelfitting.fitter.TraceFitter`, users can use a standard
  curve fitting algorithm for refinement by calling `~brian2modelfitting.fitter.TraceFitter.refine`.

Simulation-based inference
==========================

tba