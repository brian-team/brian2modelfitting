from brian2 import *


defaultclock.dt = 0.1*ms


I1 = 0.1*nA
I2 = 0.5*nA

t_on = 100*ms
t_total = 1000*ms
t_off = t_total - t_on

C = 200*pF
VT = -60.0*mV
E_Na = 53*mV
g_Na = 32*uS
E_K = -107*mV
g_K = 1*uS
E_l = -70*mV
g_l = 10*nS
E_l = -70*mV

eqs = '''
    dVm/dt = - (g_Na * m ** 3 * h * (Vm - E_Na) 
                + g_K * n ** 4 * (Vm - E_K)
                + g_l * (Vm - E_l) - I_inj) / C : volt
    I_inj = ((I2 - I1) / (t_off - t_on) * (t - t_on) + I1) * int((t >= t_on) and (t < t_off)) : amp (shared)
    dm/dt = alpha_m * (1-m) - beta_m * m : 1
    dn/dt = alpha_n * (1-n) - beta_n * n : 1
    dh/dt = alpha_h * (1-h) - beta_h * h : 1
    alpha_m = ((-0.32/mV) * (Vm - VT - 13.*mV)
               / (exp((-(Vm - VT - 13.*mV)) / (4.*mV)) - 1))/ms : Hz
    beta_m = ((0.28/mV) * (Vm - VT - 40.*mV)
              / (exp((Vm - VT - 40.*mV) / (5.*mV)) - 1))/ms : Hz
    alpha_h = 0.128 * exp(-(Vm - VT - 17.*mV) / (18.*mV))/ms : Hz
    beta_h = 4 / (1 + exp((-(Vm - VT - 40.*mV)) / (5.*mV)))/ms : Hz
    alpha_n = ((-0.032/mV) * (Vm - VT - 15.*mV)
               / (exp((-(Vm - VT - 15.*mV)) / (5.*mV)) - 1))/ms : Hz
    beta_n = 0.5 * exp(-(Vm - VT - 10.*mV) / (40.*mV))/ms : Hz
    '''
neurons = NeuronGroup(1, eqs, threshold='m>0.5', refractory='m>0.5',
                      method='exponential_euler', name='neurons')
neurons.Vm = 'E_l'
neurons.m = '1 / (1 + beta_m / alpha_m)'
neurons.h = '1 / (1 + beta_h / alpha_h)'
neurons.n = '1 / (1 + beta_n / alpha_n)'
Vm_mon = StateMonitor(neurons, ['Vm', 'I_inj'], record=True, name='Vm_mon')
run(t_total)

save('input_traces_synthetic.npy', Vm_mon.I_inj.ravel()/amp)
save('output_traces_synthetic.npy', Vm_mon.Vm.ravel()/mV)
