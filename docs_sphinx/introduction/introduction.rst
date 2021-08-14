Introduction
============

The `.brian2modelfitting` toolbox provides three optimization classes:
 - `~brian2modelfitting.fitter.TraceFitter`
 - `~brian2modelfitting.fitter.SpikeFitter`
 - `~brian2modelfitting.fitter.OnlineTraceFitter`

and a simulation-based inference class:
 - `~brian2modelfitting.inferencer.Inferencer`

All classes expect a model and the data as an input and return either the best
fit of each parameter with the corresponding error, or a posterior
distribution over unknown parameters.
The toolbox can optimize over multiple traces (e.g. input currents) at the
same time.
It also allows the possiblity of simultaneous fitting/inferencing by taking
into account multiple output variables including spike trains.

In following documentation we assume that `.brian2modelfitting` has been
installed and imported as follows:

.. code:: python

    from brian2modelfitting import *


Installation
------------

To install the toolbox alongside Brian 2 simulator, use ``pip`` as follows:

.. code::

  pip install brian2modelfitting


Testing Model Fitting
---------------------

Version on master branch gets automatically tested with Travis services.
To test the code yourself, you will need to have ``pytest`` installed and run
the following command inside the `.brian2modelfitting` root directory:


.. code::

    pytest
