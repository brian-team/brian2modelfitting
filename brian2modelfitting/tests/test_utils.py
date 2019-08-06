'''
Test the modelfitting utils
'''
import numpy as np
from brian2modelfitting import generate_fits
from brian2 import (Equations, start_scope, defaultclock, ms, mV, nF, nS, amp,
                    nA, set_device)
from brian2.devices import reinit_devices
from numpy.testing.utils import assert_equal

dt = 0.01 * ms
defaultclock.dt = dt

input_current1 = np.hstack([np.zeros(int(5*ms/dt)),
                            np.ones(int(5*ms/dt))*5,
                            np.zeros(int(5*ms/dt))]) * 5 * nA
input_current0 = np.hstack([np.zeros(int(5*ms/dt)),
                            np.ones(int(5*ms/dt))*10,
                            np.zeros(int(5*ms/dt))]) * 5 * nA

inp_trace = np.concatenate((np.array([input_current0]),
                            np.array([input_current1])))

eqs_fit = Equations('''
                    dv/dt = (gL*(EL-v)+gL*DeltaT*exp((v-VT)/DeltaT) + I)/C : volt
                    gL: siemens (constant)
                    C: farad (constant)
                    ''',
                    EL=-70*mV, VT=-50*mV, DeltaT=2*mV)

params = {'gL': 30*nS, 'C': 1*nF}


def test_genetate_fits():
    generate_fits


def test_genetate_spikes():
    reinit_devices()
    start_scope()
    spikes = generate_fits(model=eqs_fit,
                           params=params,
                           input=inp_trace * amp,
                           input_var='I',
                           output_var='spikes',
                           dt=dt,
                           threshold='v > -50*mV',
                           reset='v = -70*mV',
                           method='exponential_euler',
                           param_init={'v': -70*mV})

    assert_equal(np.shape(spikes[0]), (12,))
    assert_equal(np.shape(spikes[1]), (6,))


def test_generate_traces():
    reinit_devices()
    start_scope()
    fits = generate_fits(model=eqs_fit,
                         params=params,
                         input=inp_trace * amp,
                         input_var='I',
                         output_var='v',
                         dt=dt,
                         threshold='v > -50*mV',
                         reset='v = -70*mV',
                         method='exponential_euler',
                         param_init={'v': -70*mV})

    assert_equal(np.shape(fits[0]), np.shape(input_current0))
    assert_equal(np.shape(fits[1]), np.shape(input_current1))


def test_generate_standalone():
    set_device('cpp_standalone', directory='parallel', clean=False)
    start_scope()
    fits = generate_fits(model=eqs_fit,
                         params=params,
                         input=inp_trace * amp,
                         input_var='I',
                         output_var='v',
                         dt=dt,
                         threshold='v > -50*mV',
                         reset='v = -70*mV',
                         method='exponential_euler',
                         param_init={'v': -70*mV})

    assert_equal(np.shape(fits[0]), np.shape(input_current0))
    assert_equal(np.shape(fits[1]), np.shape(input_current1))

    reinit_devices()
