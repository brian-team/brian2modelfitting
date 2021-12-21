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

The `~brian2modelfitting.inferencer.Inferencer` class has to be initialized
within the script that will perform a simulation-based inference procedure.

Initialization of `~brian2modelfitting.inferencer.Inferencer` requires:

- ``dt`` - the time step in Brian 2 units.
- ``model`` - single cell model equations, defined as either string or as 
  ``brian2.Equation`` object.
- ``input`` - a dictionary where key corresponds to the name of the input
  variable as defined in ``model`` and value corresponds to an array of
  input traces.
- ``output`` - a dictionary where key corresponds to the name of the output
  variable as defined in ``model`` and value corresponds to an array of
  recorded traces and/or spike trains.

.. code:: python

  inferencer = Inferencer(dt=0.1*ms, model=eqs,
                          input={'I': inp_traces*amp},
                          output={'v': out_traces*mV})

Optionally, arguments to be passed to the constructor are:

- ``features`` - a dictionary of callables that take the voltage
  trace and/or spike trains and output summary statistics. Keys correspond to
  output variable names, while values are lists of callables. If features are
  not provided, automatic feature extraction will be performed either by using
  the default multi-layer perceptron or by using the user-provided embedding
  network.
- ``method`` - a string that defines an integration method.
- ``threshold`` - optional string that defines the condition which produces
  spikes. It should be a single line boolean expression.
- ``reset`` - an optional (multi-line) string that that holds the code to
  execute on reset.
- ``refractory`` - can be either Boolean expression or string. Defines either
  the length of the refractory period (e.g., ``2*ms``), a string expression
  that evaluates to the length of the refractory period after each spike,
  e.g., ``'(1 + rand())*ms'``, or a string expression evaluating to a boolean
  value, given the condition under which the neuron stays refractory after a
  spike, e.g., ``'v > -20*mV'``.
- ``param_init`` - a dictionary of state variables to be initialized with
  respective values, i.e., initial conditions.

.. code:: python

  inferencer = Inferencer(dt=dt, model=eqs_inf,
                          input={'I': inp_trace*amp},
                          output={'v': out_traces*mV},
                          features={'v': voltage_feature_list},
                          method='exponential_euler',
                          threshold='v > -50*mV',
                          reset='v = -70*mV',
                          param_init={'v': -70*mV})

Inference
---------

After the `~brian2modelfitting.inferencer.Inferencer` class is
instantiated, the simplest and the most convenient way to start with the
inferencer procedure is by calling `~brian2modelfitting.inferencer.Inferencer.infer`
method on `~brian2modelfitting.inferencer.Inferencer` object.

In the nutshell, infer method returns the trained neural posterior object,
which may or may not be used by the user, but it has to exist. There are two
possible approaches:

- amortized inference
- multi-round inference

If the number of inference rounds is 1, then amortized inference will be
performed. Otherwise if the number of inference rounds is 2 or above, the
focused multi-round inference will be performed. Multi-round inference,
unlike the amortized one, is focused on a particular observation, where in
each new round of inference, samples are drawn from the posterior distribution
conditioned exactly by this observation. This process can be repeated
aribtrarily many times to get increasingly better approximations of the the
posterior distribution.

The infer method requires:

- ``n_samples`` - the number of samples from which the neural posterior will
  be learnt.
  
or:

- ``theta`` -  sampled prior.
- and ``x`` - summary statistics.

along with the:

- ``params`` - a dictionary of bounds for each free parameter defined in the
  ``model``. Keys should correspond to names of parameters as defined in the
  model equations, values are lists with lower and upper bounds with
  quantities of respective parameter. 

The simplest way to start the inference process is by calling:

.. code:: python

  posterior = inferencer.infer(n_samples=1000,
                               gl=[10*nS, 100*nS],
                               C=[0.1*nF, 10*nF])

Optionally, user can defined the following arguments:

- ``n_rounds`` - if it is set to 1, amortized inference will be performed.
  Otherwise, if ``n_rounds`` is integer larger than 1, multi-round inference
  will be performed. This is only valid if the posterior has not yet been
  defined. Otherwise, if this method is called after the posterior has already
  been built, multi-round inference is performed, e.g. repeated calling of
  ``~brian2modelfitting.inferencer.Inferencer.infer`` method or manually
  building the posterior by approaching the inference with flexible inference.
- ``inference_method`` - either SNPE, SNLE or SNRE.
- ``density_estimator_model`` - string that defines the type of density
  estimator to be created. Either ``mdn``, ``made``, ``maf``, ``nsf`` for SNPE
  and SNLE, or ``linear``, ``mlp``, ``resnet`` for SNRE.
- ``inference_kwargs`` - a dictionary that holds additional keyword arguments
  for the `~brian2modelfitting.inferencer.Inferencer.init_inference`.
- ``train_kwargs`` - a dictionary that holds additional keyword arguments for
  `~brian2modelfitting.inferencer.Inferencer.train`.
- ``posterior_kwargs`` - a dictionary that holds additional keyword arguments
  for `~brian2modelfitting.inferencer.Inferencer.build_posterior`.
- ``restart`` - when the method is called for a second time, set to True if
  amortized inference should be performed. If False, multi-round inference
  with the existing posterior will be performed.
- ``sbi_device`` a string that defines the device on which the ``sbi`` and
  subseqently the ``torch`` will operate. By default this is set to ``cpu``
  and it is advisable to remain so for most cases. In cases where the user
  provides custom embedding network through ``inference_kwargs`` argument,
  which will be trained more efficiently by using GPU, device should be set
  accordingly to ``gpu``.

A bit more comprehensive specification of the infer call is showcased below:

.. code:: python

  posterior = inferencer.infer(n_samples=5_000,
                               n_rounds=3,
                               inference_method='SNPE',
                               density_estimator_model='mdn',
                               restart=True,
                               sbi_device='cpu',
                               gl=[10*nS, 100*nS],
                               C=[0.1*nF, 10*nF])

Remarks
-------

For a better understanding, please go through examples that go step-by-step
through the entire process. Currently, there are two tutorials: the one that
is covering `simple interface <https://brian2modelfitting.readthedocs.io/en/stable/examples/hh_sbi_simple.html>`_,
appropriate for the regular user, and the one that goes a bit more in-depth by
using `flexible interface <https://brian2modelfitting.readthedocs.io/en/stable/examples/hh_sbi_flex.html>`_,
and shows how to manually go through the process of inference, storing/loading
the training data and the trained neural density estimator, parameter space
visualization, conditioning, etc.