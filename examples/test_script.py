from brian2 import *
from brian2modelfitting import calc_eFEL

import brian2
import efel
print(brian2.__version__)
print(efel.__version__)

# "voltage traces" that are constant at -70*mV, -60mV, -50mV, -40mV for
# 50ms each.
dt = 1 * ms
voltage = np.ones((2, 200)) * np.repeat([-70, -60, -50, -40], 50) * mV
print(np.min(voltage), np.max(voltage))
# Note that calcEFL takes times in ms
inp_times = [[99, 150], [49, 150]]
results = calc_eFEL(voltage, inp_times, ['voltage_base'], dt=dt)
assert len(results) == 2
assert all(res.keys() == {'voltage_base'} for res in results)
print(results)
