Model fitting
=============

The `brian2tools` offers model fitting package, that allows for data driven optimization of custom
models.

The toolbox allows the user to find the best fit of the parameters for recorded traces and
spike trains. Just like Brian the Model Fitting Toolbox is designed to be easily used and
save time through automatic parallelization of the simulations using code generation.

Model provides two functions:
 - `fit_spikes()`
 - `fit_traces()`


The functions accept the a model and data as an input and returns best fit of parameters
and corresponding error. Proposed solution can accept multiple traces to optimize over
at the same time.

.. contents::
    Overview
    :local:


In following documentation we assume that ``brian2tools`` has been imported like this:

.. code:: python

    from brian2tools import *


How it works
------------

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



Optimizer
---------
Optimizer class is responsible for maximizing a fitness function. Our approach
uses gradient free global optimization methods (evolutionary algorithms, genetic algorithms,
 Bayesian optimization). We provided access to two libraries.


Follows `ask()/tell()` interface
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
User can plug in different optimization tool, as long as it follows ```ask() / tell```
interface. Abstract `class Optimizer` prepared for different back-end libraries.
All of the optimizer specific arguments have to be provided upon
optimizers initialization.


```ask() / tell``` interface in optimizer class:

.. code:: python

  parameters = optimizer.ask()

  errors = simulator.run(parameters)

  optimizer.tell(parameters, errors)
  results = optimizer.recommend()


Provided libraries and methods:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**1. Nevergrad**

.. _Nevergrad: https://github.com/facebookresearch/nevergrad

Offers an extensive collection of algorithms that do not require gradient computation.
Nevergrad optimizer can be specified in the following way:

.. code:: python

  opt = NevergradOptimizer(method='PSO')

where method input is a string with specific optimization algorithm.

**Available methods include:**
 - Differential evolution. ['DE']
 - Covariance matrix adaptation.['CMA']
 - Particle swarm optimization.['PSO']
 - Sequential quadratic programming.['SQP']


Nevergrad is not yet documented, to check all available methods use following code:

.. code:: python

  from nevergrad.optimization import registry
  print(sorted(registry.keys()))


Important notes:
 - number of samples per round in Nevergrad optimization methods is limited to 30,
   to increase it user has to specify a popsize upon initialization of NevergradOptimizer

.. code:: python

     opt = NevergradOptimizer(method='DE', popsize=60)



**2. Scikit-Optimize_ (skopt)**

.. _Scikit-Optimize: https://scikit-optimize.github.io/

Skopt implements several methods for sequential model-based ("blackbox") optimization
and focuses on bayesian methods. Algorithms are based on scikit-learn minimize function.

**Available Methods:**
 - Gaussian process-based minimization algorithms ['GP']
 - Sequential optimization using gradient boosted trees ['GBRT']
 - Sequential optimisation using decision trees ['ET']
 - Random forest regressor ['RF']

User can also provide a custom made sklearn regressor. Skopt optimizer can be specified in the following way:


Parameters:

 - method = ["GP", "RF", "ET", "GBRT" or sklearn regressor, default="GP"]
 - n_initial_points [int, default=10]
 - acq_func
 - acq_optimizer
 - random_state

For more detail check Optimizer documentation. https://scikit-optimize.github.io/#skopt.Optimizer

.. code:: python

   opt = SkoptOptimizer(method='GP', acq_func='LCB')



Metric
------

Metric input to specifies the fitness function measuring the performance of the simulation.
This function gets applied on each simulated trace. We have implemented few metrics within
modelfitting. Modularity applies here as well, with provided abstract `class Metric`
prepared for different custom made metrics.

Provided metrics:
**1. Mean Square Error**

.. math:: MSE ={\frac {1}{n}}\sum _{i=1}^{n}(Y_{i}-{\hat {Y_{i}}})^{2} $$

.. code:: python

  metric = MSEMetric()

also calculated offline with ``metric=None`` as input


**2. GammaFactor - for `fit_spikes()`.**

.. math:: \Gamma = \left (\frac{2}{1-2\delta r_{exp}}\right) \left(\frac{N_{coinc} - 2\delta N_{exp}r_{exp}}{N_{exp} + N_{model}}\right)$$

:math:`N_{coinc}$` - number of coincidences

:math:`N_{exp}` and :math:`N_{model}`- number of spikes in experimental and model spike trains

:math:`r_{exp}` - average firing rate in experimental train

:math:`2 \delta N_{exp}r_{exp}` - expected number of coincidences with a Poission process

For more details on the gamma factor, see `Jolivet et al. 2008, “A benchmark test for a quantitative assessment of simple neuron models”, J. Neurosci. Methods. <https://www.ncbi.nlm.nih.gov/pubmed/18160135>`


.. code:: python

  metric = GammaFactor(delta=10*ms, dt=0.1*ms)


Features
--------
Standalone mode
~~~~~~~~~~~~~~~

Just like with regular Brian script, modelfitting computations can be performed in
``Runtime`` mode (default) or ``Standalone`` mode.
<https://brian2.readthedocs.io/en/stable/user/computation.html>

To enable this mode, add the following line after your Brian import, but before your simulation code:

.. code:: python

  set_device('cpp_standalone')



Callback function
~~~~~~~~~~~~~~~~~

The 'callback' input provides custom feedback function option. User can provide
a callable (function), that will provide an output or printout. If callback returns
`True` the fitting execution is interrupted.
 User gets four arguments to customize over:

``results, errors, parameters, index``

An example function:

.. code:: python

  def callback(results, errors, parameters, index):
      print('index {} errors minimum: {}'.format(index, min(errors)) )



Additional inputs
~~~~~~~~~~~~~~~~~
User can specify the initial values of evaluated differential equations. The fitting
functions accept additional dictionary input to address that.

.. code:: python

  param_init = {'v': -30*mV}

Integration method can be manually chosen:

.. code:: python

  method='exponential_euler',

Local Gradient Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Additional local optimization with use of gradient methods can be applied.
Coming soon...


Utils: generate fits
--------------------

In toolboxes utils we provided a helper function that will generate required traces
based on same model and input. To be used after fitting.

.. code:: python

  fits = generate_fits(model=model, params=results, input=input_current,
                       input_var='I', output_var='v', dt=0.1*ms)



Simple Examples
---------------

fit_spikes
~~~~~~~~~~

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
~~~~~~~~~~

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
