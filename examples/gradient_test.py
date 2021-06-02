from collections import defaultdict

import pandas as pd
import numpy as np
from brian2 import *
from brian2modelfitting import *

# For nicer unit display
from brian2modelfitting import SCALAR_METHODS_GRADIENT_SUPPORT

mV**-2, uvolt**-2, nvolt**-2

set_device('cpp_standalone')  # recommend for speed
dt = defaultclock.dt

# Generate ground truth data
area = 20000*umetre**2
El = -65*mV
EK = -90*mV
ENa = 50*mV
VT = -63*mV

eqs= '''
dx1/dt = (x1 - p1**2 - p2*I)/ms : 1
dx2/dt = (x1 + (p1*p2)**0.5)/ms : 1
p1 : 1 (constant)
p2 : 1 (constant)
'''
inp_ar = np.random.uniform(size=(10, 2)) * [2, 5]

inp = TimedArray(inp_ar, dt=dt)
ground_truth = NeuronGroup(2, eqs + 'I = inp(t, i) : 1',
                           method='euler')
ground_truth.p1 = 7
ground_truth.p2 = 3
mon = StateMonitor(ground_truth, ['x1', 'x2'], record=True)
run(10*defaultclock.dt)
ground_truth_x1 = mon.x1[:]
ground_truth_x2 = mon.x2[:]

class StoreResults:
    def __init__(self):
        self._results = []

    def __call__(self, parameters, errors, best_parameters, best_error,
                 index, additional_info):
        self._results.append((index, best_error))
        return False

    def results(self):
        result_array = np.array(self._results)
        self._results.clear()
        return result_array

seed(172522925)
start_p = np.random.uniform(0, 10, size=(10, 2))

full_results = defaultdict(list)

METHODS = dict(SCALAR_METHODS_GRADIENT_SUPPORT)
METHODS['least_squares'] = ('least_squares', False, False)
METHODS['leastsq'] = ('leastsq', True, False)

for method, (_, supports_grad, needs_grad) in METHODS.items():
    print(method)
    for calc_gradient in [True, False]:
        if needs_grad and not calc_gradient:
            continue
        if not supports_grad and calc_gradient:
            continue
        store_results = StoreResults()
        for idx, (start_p1, start_p2) in enumerate(start_p):
            fitter = TraceFitter(model=eqs, input={'I': inp_ar.T},
                                 output={'x1': ground_truth_x1,
                                         'x2': ground_truth_x2},
                                 dt=dt, n_samples=60,
                                 method='euler')
            try:
                refined_params, _ = fitter.refine(params={'p1': start_p1, 'p2': start_p1},
                                                  calc_gradient=calc_gradient,
                                                  method=method,
                                                  max_nfev=100,
                                                  callback=store_results,
                                                  p1=[0, 10], p2=[0, 10])
            except Exception as ex:
                print(ex)
            finally:
                full_results[method, calc_gradient].append(store_results.results())

import pickle
with open('results.pickle', 'wb') as f:
    pickle.dump(full_results, f)
