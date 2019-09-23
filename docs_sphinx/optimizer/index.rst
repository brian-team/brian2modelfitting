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
`~brian2modelfitting.optimizer.NevergradOptimizer` can be specified in the following way:

.. code:: python

  opt = NevergradOptimizer(method='PSO')

where method input is a string with specific optimization algorithm.

**Available methods include:**
 - Differential evolution. [``'DE'``]
 - Covariance matrix adaptation.[``'CMA'``]
 - Particle swarm optimization.[``'PSO'``]
 - Sequential quadratic programming.[``'SQP'``]


Nevergrad is still poorly documented, to check all the available methods use the
following code:

.. code:: python

  from nevergrad.optimization import registry
  print(sorted(registry.keys()))


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

User can also provide a custom made sklearn regressor.
`~brian2modelfitting.optimizer.SkoptOptimizer` can be specified in the following way:


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
custom class that inherits from provided abstract class `~brian2modelfitting.optimizer.Optimizer`

User can plug in different optimization tool, as long as it follows an ``ask() / tell``
interface. The abstract class `~brian2modelfitting.optimizer.Optimizer` is
prepared for different back-end libraries. All of the optimizer specific
arguments have to be provided upon optimizers initialization.


The ``ask() / tell`` interface is used as follows:

.. code:: python

  parameters = optimizer.ask()

  errors = simulator.run(parameters)

  optimizer.tell(parameters, errors)
  results = optimizer.recommend()
