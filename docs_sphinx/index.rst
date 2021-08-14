brian2modelfitting
==================

The package `.brian2modelfitting` is a tool for parameter identification of
neuron models defined in the `Brian 2 simulator <https://brian2.readthedocs.org>`_.

Please report bugs at the `GitHub issue tracker <https://github.com/brian-team/brian2modelfitting/issues>`_
or at the `Brian 2 discussion forum <https://groups.google.com/forum/#!forum/briansupport>`_.
The latter is also a place to discuss feature requests or potential
contributions.


Model fitting
-------------

This toolbox allows the user to find the best fit of the unknown free
parameters for recorded traces and spike trains.
It also supports simulation-based inference, where instead of point-estimated
parameter values, a full posterior distribution over the parameters is
computed.

By default, the toolbox supports a range of global derivative-free
optimization methods, that include popular methods for model fitting:
differential evolution, particle swarm optimization and covariance matrix
adaptation (provided by the ``Nevergrad``, a gradient-free optimization
platform) as well as Bayesian optimization for black box functions (provided
by ``scikit-optimize``, a sequential model-based optimization library). On the
other hand, simulation-based inference is the process of finding parameters of
a simulator from observations by taking a Bayesian approach via sequential
neural posterior estimation, likelihood estimation or ration estimation
(provided by the ``sbi``), where neural densitiy estimator, a deep neural
network allowing probabilistic association between the data and underlying
parameter space, is trained. After the network is trained, the approximated
posterior distribution is available.

Just like Brian itself, the `brian2modelfitting` toolbox is designed to be
easy to use and to save time through automatic parallelization of the
simulations using code generation.


Contents
--------

.. toctree::
   :maxdepth: 2
   :titlesonly:

   introduction/index
   optimizer/index
   metric/index
   inferencer/index
   features/index
   examples/index

API reference
-------------
.. toctree::
   :maxdepth: 5
   :titlesonly:

   api/brian2modelfitting

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
