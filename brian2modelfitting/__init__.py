'''
Tools for use with the Brian 2 simulator.
'''
import warnings

from .tests import run as run_test
from .fitter import *
from .inferencer import Inferencer
from .optimizer import *
from .metric import *
from .simulator import *
from .utils import *

try:
    from ._version import __version__, __version_tuple__
except ImportError:
    try:
        from setuptools_scm import get_version

        __version__ = get_version(
            root="..",
            relative_to=__file__,
            version_scheme="post-release",
            local_scheme="no-local-version",
        )
        __version_tuple__ = tuple(int(x) for x in __version__.split(".")[:3])
    except ImportError:
        warnings.warn(
            "Cannot determine brian2modelfitting version, running from source and "
            "setuptools_scm is not installed."
        )
        __version__ = "unknown"
        __version_tuple__ = (0, 0, 0)


__all__ = ['callback_setup', 'callback_text', 'make_dic', 'callback_none',
           'Optimizer', 'NevergradOptimizer', 'SkoptOptimizer', 'calc_bounds',
           'Simulator', 'RuntimeSimulator', 'CPPStandaloneSimulator',
           'MSEMetric', 'Metric', 'GammaFactor', 'FeatureMetric',
           'SpikeMetric', 'TraceMetric', 'get_gamma_factor', 'firing_rate',
           'Fitter', 'SpikeFitter', 'TraceFitter', 'OnlineTraceFitter',
           'Inferencer']
