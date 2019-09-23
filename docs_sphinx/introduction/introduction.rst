Introduction
============

The `.brian2modelfitting` toolbox provides three optimization classes:
 - `~brian2modelfitting.fitter.TraceFitter`
 - `~brian2modelfitting.fitter.SpikeFitter`
 - `~brian2modelfitting.fitter.OnlineTraceFitter`

The classes expect a model and data as an input and returns the best fit of
parameters and the corresponding error. The toolbox can optimize over multiple
traces (e.g. input currents) at the same time.

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
To test the code yourself, you will need to have ``pytest`` installed and run
the command:


.. code:: python

    pytest
