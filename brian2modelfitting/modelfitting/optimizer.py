import abc
from numpy import array, all
from brian2 import asarray

from skopt.space import Real
from skopt import Optimizer as skoptOptimizer
from sklearn.base import RegressorMixin
from nevergrad import instrumentation as inst
from nevergrad.optimization import optimizerlib, registry


class Optimizer(object):
    """
    Optimizer class created as a base for optimization initialization and
    performance with different libraries. To be used with modelfitting
    fit_traces.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, method='DE', **kwds):
        """
        Initialize the given optimizator with method and specific arguments

        Parameters
        ----------
        method: string
            optimization method
        **kwds:
            parameters to be passed to optimization init
        """
        pass

    def calc_bounds(self, parameter_names, **params):
        """
        Verify and get the provided for parameters bounds

        Parameters
        ----------
        parameter_names: list
            list of parameter names in use
        **params:
            bounds for each parameter
        """
        for param in parameter_names:
            if (param not in params):
                raise Exception("Bounds must be set for parameter %s" % param)

        bounds = []
        for name in parameter_names:
            bounds.append(params[name])

        return bounds

    @abc.abstractmethod
    def initialize(self, parameter_names, **params):
        """
        Initialize the instrumentation for the optimization, based on
        parameters, creates bounds for variables and attaches them to the
        optimizer

        Parameters
        ----------
        parameter_names: list
            list of parameter names in use
        **params:
            bounds for each parameter
        """
        pass

    @abc.abstractmethod
    def ask(self, n_samples):
        """
        Returns the requested number of samples of parameter sets

        Parameters
        ----------
        n_samples: int
            number of samples to be drawn

        Returns
        -------
        parameters: list
            list of drawn parameters [n_samples x n_params]
        """
        pass

    @abc.abstractmethod
    def tell(self, parameters, errors):
        """
        Provides the evaluated errors from parameter sets to optimizer

        Parameters
        ----------
        parameters: list
            list of parameters [n_samples x n_params]
        errors:
            list of errors [n_samples]
        """
        pass

    @abc.abstractmethod
    def recommend(self):
        """
        Returns best recomentation provided by the method

        Returns
        -------
        result: list
            list of best fit parameters[n_params]
        """
        pass


class NevergradOptimizer(Optimizer):
    """
    NevergradOptimizer instance creates all the tools necessary for the user
    to use it with Nevergrad library.

    Parameters
    ----------
    parameter_names : (list, dict)
        List/Dict of strings with parameters to be used as instruments.
    bounds : (list)
        List with appropiate bounds for each parameter.
    method : (str), optional
        The optimization method. By default differential evolution, can be
        chosen from any method in Nevergrad registry

    TODO: specify kwds
    budget: int/None
        number of allowed evaluations
    num_workers: int
        number of evaluations which will be run in parallel at once
    """

    def __init__(self,  method='DE', popsize=30, **kwds):
        super(Optimizer, self).__init__()

        if method not in list(registry.keys()):
            raise AssertionError("Unknown to Nevergrad optimization method:"
                                 + method)

        self.method = method
        self.popsize = popsize
        self.kwds = kwds  # TODO: check if kwds are a valible arguemnt

    def initialize(self, parameter_names, **params):
        for param in params.keys():
            if (param not in parameter_names):
                raise Exception("Parameter %s must be defined as a parameter \
                                 in the model" % param)

        bounds = self.calc_bounds(parameter_names, **params)

        instruments = []
        for i, name in enumerate(parameter_names):
            vars()[name] = inst.var.Array(1).bounded(*bounds[i]).asscalar()
            instruments.append(vars()[name])

        instrum = inst.Instrumentation(*instruments)
        self.optim = optimizerlib.registry[self.method](instrumentation=instrum,
                                                        **self.kwds)

        self.optim._llambda = self.popsize  # TODO: more elegant way once possible

    def ask(self, n_samples):
        self.candidates, parameters = [], []

        for _ in range(n_samples):
            cand = self.optim.ask()
            self.candidates.append(cand)
            parameters.append(list(cand.args))

        return parameters

    def tell(self, parameters, errors):
        if not(all(parameters == [list(v.args) for v in self.candidates])):
            raise AssertionError("Parameters and Candidates don't have \
                                  identical values")

        for i, candidate in enumerate(self.candidates):
            self.optim.tell(candidate, errors[i])

    def recommend(self):
        res = self.optim.provide_recommendation()
        return res.args


class SkoptOptimizer(Optimizer):
    """
    SkoptOptimizer instance creates all the tools necessary for the user
    to use it with scikit-optimize library.

    Parameters
    ----------
    parameter_names : (list, dict)
        List/Dict of strings with parameters to be used as instruments.
    bounds : (list)
        List with appropiate bounds for each parameter.
    method : (str), optional
        The optimization method. Possibilities: "GP", "RF", "ET", "GBRT" or
        sklearn regressor, default="GP"

    TODO: specify kwds
    n_calls [int, default=100]:
        Number of calls to `func`.
    """
    def __init__(self, method='GP', **kwds):
        super(Optimizer, self).__init__()
        if not(method.upper() in ["GP", "RF", "ET", "GBRT"] or
               isinstance(method, RegressorMixin)):
            raise AssertionError("Provided method: {} is not an skopt \
                                  optimization or a regressor".format(method))

        self.method = method
        self.kwds = kwds  # TODO: check if kwds are a valible arguemnt

    def initialize(self, parameter_names, **params):
        for param in params.keys():
            if (param not in parameter_names):
                raise Exception("Parameter %s must be defined as a parameter \
                                 in the model" % param)

        bounds = self.calc_bounds(parameter_names, **params)

        instruments = []
        for i, name in enumerate(parameter_names):
            vars()[name] = Real(*asarray(bounds[i]), transform='normalize')
            instruments.append(vars()[name])

        self.optim = skoptOptimizer(
            dimensions=instruments,
            base_estimator=self.method,
            **self.kwds
        )

    def ask(self, n_samples):
        return self.optim.ask(n_points=n_samples)

    def tell(self, parameters, errors):
        self.optim.tell(parameters, errors.tolist());

    def recommend(self):
        xi = self.optim.Xi
        yii = array(self.optim.yi)
        return xi[yii.argmin()]
