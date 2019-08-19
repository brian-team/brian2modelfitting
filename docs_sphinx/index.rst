brian2modelfitting
==================

The `brian2modelfitting` is a package for parameter fitting for neuron models
in the `Brian 2 simulator <https://brian2.readthedocs.org>`_.

Please contact us at
``brian-development@googlegroups.com`` (https://groups.google.com/forum/#!forum/brian-development)
if you are interested in contributing.

Please report bugs at the `github issue tracker <https://github.com/brian-team/brian2modelfitting/issues>`_ or to
``briansupport@googlegroups.com`` (https://groups.google.com/forum/#!forum/briansupport).


Model fitting
-------------

The ``brian2modelfitting`` offers model fitting package, that allows for data driven optimization of custom
models.

The toolbox allows the user to find the best fit of the parameters for recorded traces and
spike trains. Just like Brian the Model Fitting Toolbox is designed to be easily used and
save time through automatic parallelization of the simulations using code generation.

Model Fitting provides three optimization classes:
 - ``TraceFitter()``
 - ``SpikeFitter()``
 - ``OnlineTraceFitter()``

The class accept the a model and data as an input and returns best fit of parameters
and corresponding error. Proposed solution can accept multiple traces to optimize over
at the same time.

.. contents::
    Overview
    :local:


In following documentation we assume that ``brian2modelfitting`` has been imported like this:

.. code:: python

    from brian2modelfitting import *


Installation
------------

To install Model Fitting alongside Brian2 you can use pip, by using
a pip utility:

.. code:: python

    pip install brian2


Contents
--------

.. toctree::
   :maxdepth: 2
   :titlesonly:

   introduction/index
   optimizer/index
   metric/index
   features/index
   examples/index

API reference
-------------
.. toctree::
   :maxdepth: 5
   :titlesonly:

   api/index

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
