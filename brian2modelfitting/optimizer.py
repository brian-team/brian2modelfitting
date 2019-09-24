import abc
from numpy import array, all, ndarray
from brian2 import asarray

from skopt.space import Real
from skopt import Optimizer as skoptOptimizer
from sklearn.base import RegressorMixin
from nevergrad import instrumentation as inst
from nevergrad.optimization import optimizerlib, registry


def calc_bounds(parameter_names, **params):
    """
    Verify and get the provided for parameters bounds

    Parameters
    ----------
    parameter_names: list[str]
        list of parameter names in use
    **params
        bounds for each parameter
    """
    for param in parameter_names:
        if param not in params:
            raise TypeError("Bounds must be set for parameter %s" % param)

    bounds = []
    for name in parameter_names:
        bounds.append(params[name])

    return bounds


class Optimizer(metaclass=abc.ABCMeta):
    """
    Optimizer class created as a base for optimization initialization and
    performance with different libraries. To be used with modelfitting
    Fitter.
    """
    @abc.abstractmethod
    def initialize(self, parameter_names, popsize, **params):
        """
        Initialize the instrumentation for the optimization, based on
        parameters, creates bounds for variables and attaches them to the
        optimizer

        Parameters
        ----------
        parameter_names: list[str]
            list of parameter names in use
        popsize: int
            population size
        **params
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
        errors: list
            list of errors [n_samples]
        """
        pass

    @abc.abstractmethod
    def recommend(self):
        """
        Returns best recommendation provided by the method

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
    parameter_names: `list` or `dict`
        List/Dict of strings with parameters to be used as instruments.
    bounds: `list`
        List with appropriate bounds for each parameter.
    method: `str`, optional
        The optimization method. By default differential evolution, can be
        chosen from any method in Nevergrad registry
    budget: int or None
        number of allowed evaluations
    num_workers: int
        number of evaluations which will be run in parallel at once
    """

    def __init__(self,  method='DE', **kwds):
        super(Optimizer, self).__init__()

        if method not in list(registry.keys()):
            raise AssertionError("Unknown to Nevergrad optimization method:"
                                 + method)

        self.method = method
        self.kwds = kwds

    def initialize(self, parameter_names, popsize, **params):
        for param in params.keys():
            if param not in parameter_names:
                raise ValueError("Parameter %s must be defined as a parameter "
                                 "in the model" % param)

        bounds = calc_bounds(parameter_names, **params)

        instruments = []
        for i, name in enumerate(parameter_names):
            vars()[name] = inst.var.Array(1).bounded(*bounds[i]).asscalar()
            instruments.append(vars()[name])

        instrum = inst.Instrumentation(*instruments)
        self.optim = optimizerlib.registry[self.method](instrumentation=instrum,
                                                        **self.kwds)

        self.optim._llambda = popsize  # TODO: more elegant way once possible

    def ask(self, n_samples):
        self.candidates, parameters = [], []

        for _ in range(n_samples):
            cand = self.optim.ask()
            self.candidates.append(cand)
            parameters.append(list(cand.args))

        return parameters

    def tell(self, parameters, errors):
        if not(all(parameters == [list(v.args) for v in self.candidates])):
            raise AssertionError("Parameters and Candidates don't have "
                                 "identical values")

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
    parameter_names: list[str]
        Parameters to be used as instruments.
    bounds : list
        List with appropiate bounds for each parameter.
    method : `str`, optional
        The optimization method. Possibilities: "GP", "RF", "ET", "GBRT" or
        sklearn regressor, default="GP"
    n_calls: `int`
        Number of calls to ``func``. Defaults to 100.
    """
    def __init__(self, method='GP', **kwds):
        super(Optimizer, self).__init__()
        if not(method.upper() in ["GP", "RF", "ET", "GBRT"] or
               isinstance(method, RegressorMixin)):
            raise AssertionError("Provided method: {} is not an skopt "
                                 "optimization or a regressor".format(method))

        self.method = method
        self.kwds = kwds

    def initialize(self, parameter_names, popsize, **params):
        for param in params.keys():
            if param not in parameter_names:
                raise ValueError("Parameter %s must be defined as a parameter "
                                 "in the model" % param)

        bounds = calc_bounds(parameter_names, **params)

        instruments = []
        for i, name in enumerate(parameter_names):
            vars()[name] = Real(*asarray(bounds[i]), transform='normalize')
            instruments.append(vars()[name])

        self.optim = skoptOptimizer(
            dimensions=instruments,
            base_estimator=self.method,
            **self.kwds)

    def ask(self, n_samples):
        return self.optim.ask(n_points=n_samples)

    def tell(self, parameters, errors):
        if type(errors) is ndarray:
            errors = errors.tolist()

        self.optim.tell(parameters, errors)

    def recommend(self):
        xi = self.optim.Xi
        yii = array(self.optim.yi)
        return xi[yii.argmin()]
