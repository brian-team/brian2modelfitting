import numpy as np
from brian2 import *
from brian2modelfitting import *

# Generate data
duration = 100*ms
dt = 0.01 * ms
defaultclock.dt = dt
input_current = np.hstack([np.zeros(int(5*ms/dt)), np.ones(int(5*ms/dt)), np.zeros(5*int(5*ms/dt))])* 5 * nA

C = 1*nF; gL = 30*nS; EL = -70*mV; VT = -50*mV; DeltaT = 2*mV
eqs = '''
    dv/dt = (gL*(EL-v)+gL*DeltaT*exp((v-VT)/DeltaT) + I(t))/C : volt
    '''

I = TimedArray(input_current, dt=dt)

group = NeuronGroup(1, eqs,
                    threshold='v > -50*mV',
                    reset='v = -70*mV',
                    method='exponential_euler')

group.v = -70 *mV

monitor = StateMonitor(group, 'v', record=True)
run(duration)
voltage = monitor.v[0]/mV
voltage += np.random.randn(len(voltage))*1/2

inp_trace = np.array([input_current])
n0, n1 = inp_trace.shape

out_trace = np.array(voltage[:n1])

# Model Fitting
eqs_fit = Equations('''
    dv/dt = (gL*(EL-v)+gL*DeltaT*exp((v-VT)/DeltaT) + I)/C : volt
    gL: siemens (constant)
    EL:  volt (constant)
    VT: volt (constant)
    ''',
    C = 1*nF,
    DeltaT = 2*mV,
    )

n_opt = NevergradOptimizer('DE')
metric = MSEMetric()

res, error = fit_traces(model=eqs_fit, input_var='I', output_var='v',
                        input=inp_trace * amp, output=[out_trace]*mV, dt=dt,
                        param_init={'v': -70*mV},
                        method='exponential_euler',
                        gL=[1*nS, 100*nS],
                        EL=[-100*mV, 0*mV],
                        VT=[-100*mV, 0*mV],
                        n_rounds=3, n_samples=30, optimizer=n_opt,
                        metric=metric,
                        threshold='v > -50*mV', reset='v = -70*mV')

print(res)

# generate fits
fits = generate_fits(model=eqs_fit, method='exponential_euler', params=res,
                     input=inp_trace * amp, input_var='I', output_var='v',
                     dt=dt, param_init={'v':-70*mV},threshold='v > -50*mV', reset='v = -70*mV')


plot(np.arange(len(out_trace))*dt/ms, out_trace)
plot(np.arange(len(fits[0]))*dt/ms, fits[0]/mV)
plt.show()
