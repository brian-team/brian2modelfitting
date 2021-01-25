from brian2 import *
from brian2modelfitting import *

# For nicer unit display
mV**-2, uvolt**-2, nvolt**-2

set_device('cpp_standalone')

# A trivial model where we know the ground truth
C = 2*nF
g_L = 10*nS
inp_ar = np.array([[0, 0.4, 0.4, 0.4, 0],
                   [0, 0.8, 0.8, 0.8, 0],
                   [0, 1.2, 1.2, 1.2, 0]])
inp = TimedArray(inp_ar.T*nA, dt=20*ms)
E_L = -70*mV
G = NeuronGroup(3, '''dv/dt = (g_L*(E_L - v) + I)/C : volt (unless refractory)
                      I = inp(t, i): amp''',
                threshold='v > -50*mV', reset='v=E_L', refractory=1*ms)
G.v = E_L
mon = StateMonitor(G, ['v', 'I'], record=True)
run(100*ms)
ground_truth = Quantity(mon.v.T)
inp_ar = Quantity(mon.I.T)

# Fit the model
eqs = '''dv/dt = (g_L*(E_L - v) + I)/C : volt (unless refractory)
         C : farad (constant)
         g_L : siemens (constant)
         E_L : volt (constant)'''
fitter = TraceFitter(model=eqs, input={'I': inp_ar.T},
                     output={'v': ground_truth.T},
                     dt=defaultclock.dt, n_samples=60,
                     param_init={'v': E_L},
                     method='exact', threshold='v > -50*mV',
                     reset='v=E_L', refractory=1*ms)
n_opt = NevergradOptimizer()

res, error = fitter.fit(n_rounds=0,
                        optimizer=n_opt, metric=MSEMetric(normalization=1*mV),
                        callback='text',
                        C=[1*nF, 3*nF],
                        g_L=[1*nS, 30*nS],
                        E_L=[-80*mV, -50*mV])

refined, _ = fitter.refine(params={'C': 1.2*nF, 'g_L': 15*nS, 'E_L': -60*mV},
                           method='basinhopping',
                           calc_gradient=True)
print(refined)
