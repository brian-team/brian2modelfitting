brian2modelfitting
==================

Model fitting toolbox for Brian 2 simulator.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4601961.svg)](https://doi.org/10.5281/zenodo.4601961)



The package [`brian2modelfitting`](https://brian2modelfitting.readthedocs.io) allows the user to find the best fit of the unknown free parameters for recorded traces and spike trains. It also supports simulation-based inference, where instead of point-estimated parameter values, a full posterior distribution over the parameters is computed.

By default, the toolbox supports a range of global derivative-free optimization methods, that include popular methods for model fitting: differential evolution, particle swarm optimization and covariance matrix adaptation (provided by the  [`Nevergrad`](https://facebookresearch.github.io/nevergrad/), a gradient-free optimization platform) as well as Bayesian optimization for black box functions (provided by [`scikit-optimize`](https://scikit-optimize.github.io/stable/), a sequential model-based optimization library). On the other hand, simulation-based inference is the process of finding parameters of a simulator from observations by taking a Bayesian approach via sequential neural posterior estimation, likelihood estimation or ration estimation (provided by the [`sbi`](https://www.mackelab.org/sbi/)), where neural densitiy estimator, a deep neural network allowing probabilistic association between the data and underlying parameter space, is trained. After the network is trained, the approximated posterior distribution is available.

Documentation
-------------
[![Documentation Status](https://readthedocs.org/projects/ansicolortags/badge/?version=latest)](https://brian2modelfitting.readthedocs.io)

The full documentation is available at http://brian2modelfitting.readthedocs.org.

Testing status
--------------
[![Build Status](https://github.com/brian-team/brian2modelfitting/workflows/Tests/badge.svg)](https://github.com/brian-team/brian2modelfitting/actions) 
[![Coverage Status](https://coveralls.io/repos/github/brian-team/brian2modelfitting/badge.svg?branch=master)](https://coveralls.io/github/brian-team/brian2modelfitting?branch=master)

Installation
------------
Install `brian2modelfitting` from the Python package index via pip:
```
pip install brian2modelfitting
```

License
-------
The model fitting toolbox is released under the terms of the CeCILL 2.1 license and is available [here](https://github.com/brian-team/brian2modelfitting/blob/master/LICENSE).

Use
---
Please report issues at the GitHub [issue tracker](https://github.com/brian-team/brian2modelfitting/issues) or at the [Brian 2 discussion forum](https://brian.discourse.group).
