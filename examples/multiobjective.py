import pandas as pd
import numpy as np
from brian2 import *
from brian2modelfitting import *

# For nicer unit display
mV**-2, uvolt**-2, nvolt**-2

# set_device('cpp_standalone')  # recommend for speed
dt = 0.01*ms
defaultclock.dt = dt

# Generate ground truth data
area = 20000*umetre**2
El = -65*mV
EK = -90*mV
ENa = 50*mV
VT = -63*mV
dt = 0.01*ms
eqs='''
dv/dt = (gl*(El-v) - g_na*(m*m*m)*h*(v-ENa) - g_kd*(n*n*n*n)*(v-EK) + I)/Cm : volt
dm/dt = 0.32*(mV**-1)*(13.*mV-v+VT)/
    (exp((13.*mV-v+VT)/(4.*mV))-1.)/ms*(1-m)-0.28*(mV**-1)*(v-VT-40.*mV)/
    (exp((v-VT-40.*mV)/(5.*mV))-1.)/ms*m : 1
dn/dt = 0.032*(mV**-1)*(15.*mV-v+VT)/
    (exp((15.*mV-v+VT)/(5.*mV))-1.)/ms*(1.-n)-.5*exp((10.*mV-v+VT)/(40.*mV))/ms*n : 1
dh/dt = 0.128*exp((17.*mV-v+VT)/(18.*mV))/ms*(1.-h)-4./(1+exp((40.*mV-v+VT)/(5.*mV)))/ms*h : 1
g_na : siemens (constant)
g_kd : siemens (constant)
gl   : siemens (constant)
Cm   : farad (constant)
'''
inp_ar = np.zeros((10000, 5))*nA
inp_ar[1000:, :] = 1.*nA
inp_ar *= (np.arange(5)*0.25)
inp = TimedArray(inp_ar, dt=dt)
ground_truth = NeuronGroup(5, eqs + 'I = inp(t, i) : amp',
                           method='exponential_euler')
ground_truth.v = El
ground_truth.Cm = (1*ufarad*cm**-2) * area
ground_truth.gl = (5e-5*siemens*cm**-2) * area
ground_truth.g_na = (100*msiemens*cm**-2) * area
ground_truth.g_kd = (30*msiemens*cm**-2) * area
mon = StateMonitor(ground_truth, ['v', 'm'], record=True)
run(100*ms)
ground_truth_v = mon.v[:]
ground_truth_m = mon.m[:]
## Optimization and Metric Choice
n_opt = NevergradOptimizer()

# The metrics' normalization can be used to weigh the errors produced by the
# two metrics. It is also necessary for making the units of the two errors
# compatible.
metric_v = MSEMetric(t_start=5*ms, normalization=10*mV)
metric_m = MSEMetric(t_start=5*ms, normalization=0.1)

## Fitting
fitter = TraceFitter(model=eqs, input={'I': inp_ar.T},
                     output={'v': ground_truth_v,
                             'm': ground_truth_m},
                     dt=dt, n_samples=60, param_init={'v': 'El'},
                     method='exponential_euler')

res, error = fitter.fit(n_rounds=20,
                        optimizer=n_opt, metric=[metric_v, metric_m],
                        callback='text',
                        gl=[1e-09 *siemens, 1e-07 *siemens],
                        g_na=[2e-06*siemens, 2e-04*siemens],
                        g_kd=[6e-07*siemens, 6e-05*siemens],
                        Cm=[0.1*ufarad*cm**-2 * area, 2*ufarad*cm**-2 * area])

refined_params, _ = fitter.refine(calc_gradient=True)

## Visualization of the results
fits = fitter.generate_traces(params=None, param_init={'v': -65*mV})
refined_fits = fitter.generate_traces(params=refined_params, param_init={'v': -65*mV})

fig, ax = plt.subplots(2, ncols=5, figsize=(20, 5), sharex=True, sharey='row')
for idx in range(5):
    ax[0][idx].plot(ground_truth_v[idx]/mV, 'k:', alpha=0.75,
                    label='ground truth')
    ax[0][idx].plot(fits['v'][idx].transpose()/mV, alpha=0.75, label='fit')
    ax[0][idx].plot(refined_fits['v'][idx].transpose() / mV, alpha=0.75,
                    label='refined')
    ax[1][idx].plot(ground_truth_m[idx], 'k:', alpha=0.75)
    ax[1][idx].plot(fits['m'][idx].transpose(), alpha=0.75)
    ax[1][idx].plot(refined_fits['m'][idx].transpose(), alpha=0.75)
ax[0][0].legend(loc='best')
plt.show()
