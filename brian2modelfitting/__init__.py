'''
Tools for use with the Brian 2 simulator.
'''
from .version import version
__version__ = version

from .tests import run as run_test
from .fitter import *
from .optimizer import *
from .metric import *
from .simulator import *
from .utils import *

__all__ = ['callback_setup', 'callback_text', 'make_dic', 'callback_none',
           'Optimizer', 'NevergradOptimizer', 'SkoptOptimizer', 'calc_bounds',
           'Simulator', 'RuntimeSimulator', 'CPPStandaloneSimulator',
           'MSEMetric', 'Metric', 'GammaFactor', 'FeatureMetric',
           'SpikeMetric', 'TraceMetric', 'get_gamma_factor', 'firing_rate',
           'Fitter', 'SpikeFitter', 'TraceFitter', 'OnlineTraceFitter']
