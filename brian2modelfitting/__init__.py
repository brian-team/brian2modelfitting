'''
Tools for use with the Brian 2 simulator.
'''
import os

from .tests import run as run_test
from .modelfitting import *
from .optimizer import *
from .metric import *
from .simulation import *
from .utils import *

__all__ = ['callback_setup', 'callback_text', 'make_dic', 'callback_none',
           'Optimizer', 'NevergradOptimizer', 'SkoptOptimizer', 'calc_bounds',
           'Simulation', 'RuntimeSimulation', 'CPPStandaloneSimulation',
           'MSEMetric', 'Metric', 'GammaFactor', 'FeatureMetric',
           'SpikeMetric', 'TraceMetric', 'get_gamma_factor', 'firing_rate',
           'Fitter', 'SpikeFitter', 'TraceFitter', 'OnlineTraceFitter']

try:
    # Use version written out by setuptools
    from .version import version
    __version__ = version
except ImportError:
    # Apparently we are running directly from a git clone, let
    # setuptools_scm fetch the version from git
    from setuptools_scm import get_version
    __version__ = get_version(relative_to=os.path.dirname(__file__))
    version = __version__
