import pandas as pd
import numpy as np
from brian2 import *
from brian2modelfitting import *

# Load Input and Output Data
df_inp_traces = pd.read_csv('input_traces_hh.csv')
df_out_traces = pd.read_csv('output_traces_hh.csv')

inp_traces = df_inp_traces.to_numpy()
inp_traces = inp_traces[:, 1:]

out_traces = df_out_traces.to_numpy()
out_traces = out_traces[:, 1:]

# Model Fitting
## Parameters
area = 20000*umetre**2
El = -65*mV
EK = -90*mV
ENa = 50*mV
VT = -63*mV
dt = 0.01*ms
defaultclock.dt = dt

## Modle Definition
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

## Optimization and Metric Choice
n_opt = NevergradOptimizer()
metric = MSEMetric(t_start=5*ms)

## Fitting
fitter = TraceFitter(model=eqs, input_var='I', output_var='v',
                     input=inp_traces*amp, output=out_traces*mV, dt=dt,
                     n_samples=20,
                     param_init={'v': 'VT'},
                     method='exponential_euler',)

res, error = fitter.fit(n_rounds=4,
                        optimizer=n_opt, metric=metric,
                        callback='text',
                        gl = [1e-09 *siemens, 1e-07 *siemens],
                        g_na = [2e-06*siemens, 2e-04*siemens],
                        g_kd = [6e-07*siemens, 6e-05*siemens],
                        Cm=[0.1*ufarad*cm**-2 * area, 2*ufarad*cm**-2 * area])

## Show results
all_output = fitter.results(format='dataframe')
print(all_output)

## Visualization of the results
start_scope()
fits = fitter.generate_traces(params=None, param_init={'v': -65*mV})

fig, ax = plt.subplots(ncols=5, figsize=(20,5))
ax[0].plot(out_traces[0].transpose())
ax[0].plot(fits[0].transpose()/mV)

ax[1].plot(out_traces[1].transpose())
ax[1].plot(fits[1].transpose()/mV)
ax[2].plot(out_traces[2].transpose())
ax[2].plot(fits[2].transpose()/mV)
ax[3].plot(out_traces[3].transpose())
ax[3].plot(fits[3].transpose()/mV)
ax[4].plot(out_traces[4].transpose())
ax[4].plot(fits[4].transpose()/mV)
plt.show()
