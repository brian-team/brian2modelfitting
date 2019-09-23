Optimizer
=========

Optimizer class is responsible for maximizing a fitness function. Our approach
uses gradient free global optimization methods (evolutionary algorithms, genetic algorithms,
Bayesian optimization). We provided access to two libraries.


.. contents::
    :local:
    :depth: 1


Nevergrad
---------

.. _Nevergrad: https://github.com/facebookresearch/nevergrad

Offers an extensive collection of algorithms that do not require gradient computation.
`~brian2modelfitting.metric.NevergradOptimizer` can be specified in the following way:

.. code:: python

  opt = NevergradOptimizer(method='PSO')

where method input is a string with specific optimization algorithm.

**Available methods include:**
 - Differential evolution. [``'DE'``]
 - Covariance matrix adaptation.[``'CMA'``]
 - Particle swarm optimization.[``'PSO'``]
 - Sequential quadratic programming.[``'SQP'``]


Nevergrad is not yet documented, to check all available methods use following code:

.. code:: python

  from nevergrad.optimization import registry
  print(sorted(registry.keys()))


Important notes:
 - number of samples per round in Nevergrad optimization methods is limited to 30,
   to increase it user has to specify a popsize upon initialization of NevergradOptimizer

.. code:: python

     opt = NevergradOptimizer(method='DE', popsize=60)



Scikit-Optimize_ (skopt)
------------------------

.. _Scikit-Optimize: https://scikit-optimize.github.io/

Skopt implements several methods for sequential model-based ("blackbox") optimization
and focuses on bayesian methods. Algorithms are based on scikit-learn minimize function.

**Available Methods:**
 - Gaussian process-based minimization algorithms [``'GP'``]
 - Sequential optimization using gradient boosted trees [``'GBRT'``]
 - Sequential optimisation using decision trees [``'ET'``]
 - Random forest regressor [``'RF'``]

User can also provide a custom made `sklearn regressor`. `~brian2modelfitting.fitter.metric.SkoptOptimizer` can be specified in the following way:


Parameters:

 - ``method = ["GP", "RF", "ET", "GBRT" or sklearn regressor, default="GP"]``
 - ``n_initial_points [int, default=10]``
 - ``acq_func``
 - ``acq_optimizer``
 - ``random_state``

For more detail check Optimizer documentation. https://scikit-optimize.github.io/#skopt.Optimizer

.. code:: python

   opt = SkoptOptimizer(method='GP', acq_func='LCB')



Custom Optimizer
----------------

To use a different back-end optimization library, user can provide a
custom class that inherits from provided abstract class `~brian2modelfitting.metric.Optimizer`

Follows `ask()/tell()` interface
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
User can plug in different optimization tool, as long as it follows ```ask() / tell```
interface. Abstract class `~brian2modelfitting.metric.Optimizer` prepared for different back-end libraries.
All of the optimizer specific arguments have to be provided upon
optimizers initialization.


```ask() / tell``` interface in optimizer class:

.. code:: python

  parameters = optimizer.ask()

  errors = simulator.run(parameters)

  optimizer.tell(parameters, errors)
  results = optimizer.recommend()
