from brian2 import *
from brian2modelfitting import *
import pandas as pd
from matplotlib.animation import FuncAnimation


# load data
df_inp_traces = pd.read_csv('input_traces_hh.csv')
df_out_traces = pd.read_csv('output_traces_hh.csv')
inp_traces = df_inp_traces.to_numpy()
inp_traces = inp_traces[:-1, 1:]
out_traces = df_out_traces.to_numpy()
out_traces = out_traces[:-1, 1:]

# model parameters
area = 20000. * umetre ** 2
Cm = 1. * ufarad * cm ** -2 * area
E_l = -65. * mV
E_k = -90. * mV
E_na = 50. * mV
Vt = -63. * mV
init_v = {'v': -65. * mV}
dt = 0.01 * ms
defaultclock.dt = dt

# model definition
hodgkin_huxley = Equations(
'''dv/dt = (
        (g_l * (E_l - v)
        - g_na * (m ** 3) * h * (v - E_na)
        - g_k * (n ** 4) * (v - E_k) + I) / Cm) : volt
    dm/dt = (
        0.32 * (mV ** -1 ) * (13.0 * mV - v + Vt) 
        / (exp((13.0 * mV - v + Vt) / (4.0 * mV)) - 1.0) / ms * (1 - m)
        - 0.28 * (mV ** -1) * (v - Vt - 40.0 * mV) 
        / (exp((v - Vt - 40.0 * mV) / (5.0 * mV)) - 1.0) / ms * m) : 1
    dn/dt = (
        0.032 * (mV ** -1) * (15.0 * mV - v + Vt) 
        / (exp((15.0 * mV - v + Vt) / (5.0 * mV)) - 1.0) / ms * (1.0 - n)
        - 0.5 * exp((10.0 * mV - v + Vt) / (40.0 * mV)) / ms * n) : 1
    dh/dt = (
        0.128 * exp((17.0 * mV - v + Vt) / (18.0 * mV)) / ms * (1.0 - h)
        - 4.0 / (1 + exp((40.0 * mV - v + Vt) / (5.0 * mV))) / ms * h) : 1
    g_na : siemens (constant)
    g_k : siemens (constant)
    g_l : siemens (constant)''')

# optimizer instantiation
optimizer = NevergradOptimizer()

# metric instantiation
metric = MSEMetric()

# fitter definition and fitting procedure
n_samples = 40
fitter = TraceFitter(
    model=hodgkin_huxley,
    input_var='I', input=inp_traces * amp,
    output_var='v', output=out_traces * mV,
    dt=dt,
    n_samples=n_samples,
    method='exponential_euler',
    param_init=init_v)


def callback(params, errors, best_params, best_error, index):
    """Custom callback.
    
    Print the best error for each optimization round."""
    print(f'[round {index + 1}]\tbest error: {np.min(errors)}')


# fitting procedure
n_rounds = 25
res, error = fitter.fit(
    optimizer=optimizer,
    metric=metric,
    n_rounds=n_rounds,
    callback=callback,
    g_l=[1.e-09 * siemens, 1.e-07 * siemens],
    g_na=[2.e-06 * siemens, 2.e-04 * siemens],
    g_k=[6.e-07 * siemens, 6.e-05 * siemens])

# visualization of best fitted traces
fit_traces = fitter.generate_traces(params=res, param_init=init_v)

nrows = 2
ncols = fit_traces.shape[0]
fig, axs = plt.subplots(
    nrows=nrows, ncols=ncols, sharex=True,
    gridspec_kw={'height_ratios': [3, 1]}, figsize=(15, 4))
for idx in range(ncols):
    axs[0, idx].plot(out_traces[idx, :].T, 'k-', label='$V_m^{measured}(t)$')
    axs[0, idx].plot(fit_traces[idx, :].T / mV, 'r--', label='$V_m^{fit}(t)$')
    axs[1, idx].plot(inp_traces[idx, :].T / amp, 'k-', label='$I(t)$')
    axs[0, idx].grid()
    axs[1, idx].grid()
    axs[1, idx].set_xlabel('t [ms]')
    if idx == 0:
        axs[0, idx].set_ylabel('$V_m$ [mV]')
        axs[1, idx].set_ylabel('$I$ [A/cm$^2$]')
handles, labels = [
    (h + l) for h, l
    in zip(axs[0, idx].get_legend_handles_labels(),
           axs[1, idx].get_legend_handles_labels())]
fig.legend(handles, labels, loc='upper right')
plt.tight_layout()
plt.show()

# visualization of errors and parameters evolving over time
full_output = fitter.results(format='dict', use_units=False)
g_k = full_output['g_k']
g_na = full_output['g_na']
g_l = full_output['g_l']
error = full_output['error']

fig = plt.figure(figsize=(15, 5))
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.set(xlabel='$g_K$ [S]', ylabel='$g_{Na}$ [S]', zlabel='$g_l$ [S]',
        xlim=(0, g_k.max() * 1.01), ylim=(0, g_na.max() * 1.01),
        zlim=(0, g_l.max() * 1.01))
ax1.ticklabel_format(useOffset=True, style='scientific', scilimits=(0, 0))
ax2 = fig.add_subplot(1, 2, 2)
ax2.set(xlabel='round', ylabel='error')
ax2.grid()


def init():
    """Scatter plot of the initial population of parameters and the
    best associated error."""
    ax1.plot3D(g_k[:n_samples], g_na[:n_samples], g_l[:n_samples],
               'b*', markersize=4, label='init population')
    ax2.plot([0], np.min(error[:n_samples]),
             'b*', markersize=8, label='init best error')
    ax1.legend()
    ax2.legend()


def animate(frame):
    """Scatter plot current population of parameters for each frame,
    starting from the second round of optimization.
    
    Number of frames should correspond to the number of optimization
    rounds."""
    istart = frame * n_samples
    iend = istart + n_samples
    if (res['g_k'] / siemens in g_k[istart:iend]
            and res['g_na'] / siemens in g_na[istart:iend]
            and res['g_l'] / siemens in g_l[istart:iend]):
        ax1.plot3D([res['g_k']], [res['g_na']], [res['g_l']],
                  'r*', markersize=8, label='best params')
        ax2.plot(frame, np.min(error[istart:iend]),
                'r*', markersize=8, label='best error')
        ax1.legend()
        ax2.legend()
    else:
        ax1.plot3D(g_k[istart:iend], g_na[istart:iend], g_l[istart:iend],
                   'ko', markersize=4, zorder=-1, alpha=0.3)
        ax2.plot(frame, np.min(error[istart:iend]), 'ko',
                 markersize=4, zorder=-1)


anim = FuncAnimation(
    fig, animate, init_func=init, frames=np.arange(1, n_rounds), repeat=False)
plt.tight_layout()
plt.show()
