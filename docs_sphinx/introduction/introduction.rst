Introduction
============

The ``brian2modelfitting`` offers model fitting package, that allows for data driven optimization of custom
models.

Model provides three optimization classes:
 - `~brian2modelfitting.fitter.TraceFitter`
 - `~brian2modelfitting.fitter.SpikeFitter`
 - `~brian2modelfitting.fitter.OnlineTraceFitter`

The class accepts a model and data as an input and returns best fit of parameters
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

  pip install brian2modelfitting


Testing Model Fitting
---------------------

Version on master branch gets automatically tested with Travis services.
To test the code yourself, you will need to have ``pytest`` installed run a command:


.. code:: python

    pytest
