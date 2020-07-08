import abc
import numpy as np
import warnings

from brian2.utils.logger import get_logger

# Prevent sklearn from adding a filter by monkey-patching the warnings module
# TODO: Remove when we depend on a newer version of scikit-learn (with
#       https://github.com/scikit-learn/scikit-learn/pull/15080 merged)
_filterwarnings = warnings.filterwarnings
warnings.filterwarnings = lambda *args, **kwds: None
from skopt.space import Real
from skopt import Optimizer as skoptOptimizer
from sklearn.base import RegressorMixin
warnings.filterwarnings = _filterwarnings

from nevergrad import instrumentation as inst
from nevergrad.optimization import optimizerlib, registry

logger = get_logger(__name__)

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

        Returns
        -------
        popsize : int
            The actual population size that will be used by the algorithm.
            Does not always correspond to ``popsize``, since some algorithms
            have minimal/maximal population sizes.
        """
        return popsize

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
    use_nevergrad_recommendation: bool, optional
        Whether to use Nevergrad's recommendation as the "best result". This
        recommendation takes several evaluations of the same parameters (for
        stochastic simulations) into account. The alternative is to simply
        return the parameters with the lowest error so far (the default). The
        problem with Nevergrad's recommendation is that it can give wrong result
        for errors that are very close in magnitude due (see github issue #16).
    budget: int or None
        number of allowed evaluations
    num_workers: int
        number of evaluations which will be run in parallel at once
    """

    def __init__(self,  method='DE', use_nevergrad_recommendation=False,
                 **kwds):
        super(Optimizer, self).__init__()

        if method not in registry:
            raise AssertionError("Unknown to Nevergrad optimization method:"
                                 + method)
        self.tested_parameters = []
        self.errors = []
        self.method = method
        self.use_nevergrad_recommendation = use_nevergrad_recommendation
        self.kwds = kwds

    def initialize(self, parameter_names, popsize, **params):
        self.tested_parameters = []
        self.errors = []
        for param in params:
            if param not in parameter_names:
                raise ValueError("Parameter %s must be defined as a parameter "
                                 "in the model" % param)

        bounds = calc_bounds(parameter_names, **params)

        instruments = []
        for i, name in enumerate(parameter_names):
            assert len(bounds[i]) == 2
            instrumentation = inst.var.Array(1).asscalar().bounded(np.array([bounds[i][0]]),
                                                                   np.array([bounds[i][1]]))
            instruments.append(instrumentation)

        instrum = inst.Instrumentation(*instruments)
        nevergrad_method = optimizerlib.registry[self.method]
        if nevergrad_method.no_parallelization and popsize > 1:
            logger.warn(f'Sample size {popsize} requested, but Nevergrad\'s '
                        f'\'{self.method}\' algorithm does not support '
                        f'parallelization. Will run the algorithm '
                        f'sequentially.',
                        name_suffix='no_parallelization')
            popsize = 1
        self.optim = nevergrad_method(instrumentation=instrum,
                                      num_workers=popsize,
                                      **self.kwds)
        if hasattr(self.optim, 'llambda'):
            optimizer_pop_size = self.optim.llambda
        elif hasattr(self.optim, 'es') and hasattr(self.optim.es, 'popsize'):
            # For CMA algorithm
            optimizer_pop_size = self.optim.es.popsize
        else:
            optimizer_pop_size = popsize
        if optimizer_pop_size != popsize:
            logger.warn(f'Sample size {popsize} requested, but Nevergrad\'s '
                        f'\'{self.method}\' algorithm will use '
                        f'{optimizer_pop_size}.', name_suffix='sample_size')

        return optimizer_pop_size

    def ask(self, n_samples):
        self.candidates, parameters = [], []

        for _ in range(n_samples):
            cand = self.optim.ask()
            self.candidates.append(cand)
            parameters.append(list(cand.args))

        return parameters

    def tell(self, parameters, errors):
        if not(np.all(parameters == [list(v.args) for v in self.candidates])):
            raise AssertionError("Parameters and Candidates don't have "
                                 "identical values")

        for i, candidate in enumerate(self.candidates):
            self.optim.tell(candidate, errors[i])
        self.tested_parameters.extend(parameters)
        self.errors.extend(errors)

    def recommend(self):
        if self.use_nevergrad_recommendation:
            res = self.optim.provide_recommendation()
            return res.args
        else:
            best = np.argmin(self.errors)
            return self.tested_parameters[best]


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
        self.tested_parameters = []
        self.errors = []

    def initialize(self, parameter_names, popsize, **params):
        self.tested_parameters = []
        self.errors = []
        for param in params.keys():
            if param not in parameter_names:
                raise ValueError("Parameter %s must be defined as a parameter "
                                 "in the model" % param)

        bounds = calc_bounds(parameter_names, **params)

        instruments = []
        for i, name in enumerate(parameter_names):
            instrumentation = Real(*np.asarray(bounds[i]), transform='normalize')
            instruments.append(instrumentation)

        self.optim = skoptOptimizer(
            dimensions=instruments,
            base_estimator=self.method,
            **self.kwds)

        return popsize

    def ask(self, n_samples):
        return self.optim.ask(n_points=n_samples)

    def tell(self, parameters, errors):
        if isinstance(errors, np.ndarray):
            errors = errors.tolist()

        self.tested_parameters.extend(parameters)
        self.errors.extend(errors)
        self.optim.tell(parameters, errors)

    def recommend(self):
        xi = self.optim.Xi
        yii = np.array(self.optim.yi)
        return xi[yii.argmin()]
