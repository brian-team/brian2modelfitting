Introduction
============

The `brian2modelfitting` offers model fitting package, that allows for data driven optimization of custom
models.

The toolbox allows the user to find the best fit of the parameters for recorded traces and
spike trains. Just like Brian the Model Fitting Toolbox is designed to be easily used and
save time through automatic parallelization of the simulations using code generation.

Model provides three optimization classes:
 - `TraceFitter()`
 - `SpikeFitter()`
 - `OnlineTraceFitter()`

The class accept the a model and data as an input and returns best fit of parameters
and corresponding error. Proposed solution can accept multiple traces to optimize over
at the same time.

In following documentation we assume that ``brian2modelfitting`` has been imported like this:

.. code:: python

    from brian2modelfitting import *


Installation
------------

To install Model Fitting alongside Brian2 you can use pip, by using
a pip utility:

.. code:: python

    pip install brian2


Testing Model Fitting
---------------------

Version on master branch gets automatically tested with Travis services.
To test the code yourself, you will need to have `pytest` installed run a command:


.. code:: python

    pytest
