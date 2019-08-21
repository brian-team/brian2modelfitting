brian2modelfitting
==================


Model Fitting Toolbox for Brian 2 simulator, to allow the user to find the best fit of the parameters for recorded traces and spike trains.

By default, we support a range of global derivative-free optimization methods, that include popular methods for model fitting, such as: Differential Evolution, Particle Swarm Optimization and Covariance Matrix Adaptation (provided by the Nevergrad) as well as Bayesian Optimization for black box functions (provided by Scikit-Optimize).

Documentation for Brian2modelfitting can be found at http://brian2modelfitting.readthedocs.org

Brian2modelfitting is released under the terms of the CeCILL 2.1 license.

Please report issues at the github issue tracker (https://github.com/brian-team/brian2/issues) or at the brian support mailing list (http://groups.google.com/group/briansupport/)




Testing status
--------------
master
[![Build Status](https://travis-ci.org/brian-team/brian2modelfitting.svg?branch=master)](https://travis-ci.org/brian-team/brian2modelfitting)

dev
[![Build Status](https://travis-ci.org/brian-team/brian2modelfitting.svg?branch=dev)](https://travis-ci.org/brian-team/brian2modelfitting)

[![Coverage Status](https://coveralls.io/repos/github/brian-team/brian2modelfitting/badge.svg?branch=dev)](https://coveralls.io/github/brian-team/brian2modelfitting?branch=dev)
