brian2modelfitting
==================


Model Fitting Toolbox for Brian 2 simulator, to allow the user to find the best fit of the parameters for recorded traces and spike trains.

By default, we support a range of global derivative-free optimization methods, that include popular methods for model fitting, such as: Differential Evolution, Particle Swarm Optimization and Covariance Matrix Adaptation (provided by the Nevergrad) as well as Bayesian Optimization for black box functions (provided by Scikit-Optimize).

Documentation for Brian2modelfitting can be found at http://brian2modelfitting.readthedocs.org

Brian2modelfitting is released under the terms of the CeCILL 2.1 license.

Please report issues at the github issue tracker (https://github.com/brian-team/brian2modelfitting/issues) or at the brian discussion forum (https://brian.discourse.group)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4601961.svg)](https://doi.org/10.5281/zenodo.4601961)

Installation
------------
Module can be installed with pypi:
```
pip install brian2modelfitting
```


Testing status
--------------
[![Build Status](https://github.com/brian-team/brian2modelfitting/workflows/Tests/badge.svg)](https://github.com/brian-team/brian2modelfitting/actions) 
[![Coverage Status](https://coveralls.io/repos/github/brian-team/brian2modelfitting/badge.svg?branch=master)](https://coveralls.io/github/brian-team/brian2modelfitting?branch=master)
